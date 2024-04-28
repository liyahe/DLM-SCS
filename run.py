"""Finetuning the library models for sequence classification on GLUE."""

import os
import logging
import os
import sys
from typing import Callable, Dict
import torch
import pickle
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    ElectraConfig,
    ElectraTokenizer,
    GPT2Config,
    GPT2Tokenizer,
)
from transformers import HfArgumentParser, set_seed
from src.dataset import FewShotDataset
from src.models import (
    BertForPromptFinetuning,
    RobertaForPromptFinetuning,
    resize_token_type_embeddings,
    ElectraForPromptFinetuning_multiTokenDiscrimination,
    ElectraForPromptFinetuning_maskOnlyDiscrimination,
    GPT2ForPromptFinetuning,
    T5ForPromptFinetuning,
)
from src.trainer import Trainer
from src.processors import (
    num_labels_mapping,
    output_modes_mapping,
    compute_metrics_mapping,
    bound_mapping,
)
from datetime import datetime
from src.args import ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (
        training_args.save_at_last and training_args.evaluate_during_training
    ), "para 'evaluate_during_training' and 'save_at_last' must not be both true"

    model_args.cache_dir = f"../pretrained_model_cache/{model_args.model_name_or_path}/"
    data_args.compute_tf_idf = bool(data_args.compute_tf_idf)
    model_args.output_each_part_logits = bool(model_args.output_each_part_logits)
    logger.info(f"eval_steps: {training_args.eval_steps}")

    data_args.few_shot_type = model_args.few_shot_type  # using in dataset.py

    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split("\t")
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info(
                "Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping)
            )
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[: data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None  # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode)
        )
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Create config
    if "electra" in model_args.model_name_or_path:
        config = ElectraConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
    elif "gpt2" in model_args.model_name_or_path:
        config = GPT2Config.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        config.ori_vocab_size = config.vocab_size
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
    logger.info(f"model type: {config.model_type}")
    data_args.model_type = config.model_type

    if "multiTokenDiscrimination" in model_args.few_shot_type:
        assert config.model_type == "electra"
        model_fn = ElectraForPromptFinetuning_multiTokenDiscrimination
    elif "maskOnlyDiscrimination" in model_args.few_shot_type:
        assert config.model_type == "electra"
        model_fn = ElectraForPromptFinetuning_maskOnlyDiscrimination
    else:
        if "prompt" in model_args.few_shot_type:
            if config.model_type == "roberta":
                model_fn = RobertaForPromptFinetuning
            elif config.model_type == "bert":
                model_fn = BertForPromptFinetuning
            elif config.model_type == "gpt2":
                model_fn = GPT2ForPromptFinetuning
            elif config.model_type == "t5":
                model_fn = T5ForPromptFinetuning
            else:
                raise NotImplementedError
        elif model_args.few_shot_type == "finetune":
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError

    # additional_special_tokens = [f"<virtual_{i}>" for i in range(50)]
    additional_special_tokens = []

    # Create tokenizer
    if "electra" in model_args.model_name_or_path:
        tokenizer = ElectraTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=additional_special_tokens,
            cache_dir=model_args.cache_dir,
        )
    elif "gpt2" in model_args.model_name_or_path:
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=additional_special_tokens,
            cache_dir=model_args.cache_dir,
        )
        # add special token， mask, sep, pad, cls
        additional_special_tokens = ["[MASK]", "[SEP]", "[PAD]", "[CLS]"]
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
        )
        config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=additional_special_tokens,
            cache_dir=model_args.cache_dir,
        )
    logger.info(f"Token number: {len(tokenizer)}, special token number: {len(additional_special_tokens)}")

    # load idf dict
    idf_dict = None
    if data_args.compute_tf_idf:  # load idf dict
        logger.info("using TF-IDF...")
        idf_save_dir = f"data/idf_dict/{type(tokenizer).__name__}_5270734_wiki_articles.pkl"
        if not os.path.exists(idf_save_dir):
            raise ValueError(f"Please pre compute idf dict and save at {idf_save_dir}")
        idf_dict = pickle.load(open(idf_save_dir, "rb"))

    # Get our special datasets.
    train_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", idf_dict=idf_dict)
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", idf_dict=idf_dict) if training_args.do_eval else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", idf_dict=idf_dict)
        if training_args.do_predict
        else None
    )

    logger.info(
        f"train data={len(train_dataset) if train_dataset is not None else 0} | "
        f"eval data={len(eval_dataset) if eval_dataset is not None else 0} | "
        f"test data={len(test_dataset) if test_dataset is not None else 0}"
    )

    set_seed(training_args.seed)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if len(additional_special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == "bert":
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    model = model.to(training_args.device)

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == "regression":
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    n_params = sum([p.numel() for p in model.parameters()])
    n_learnable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info(f"# of params = {n_params} ({n_learnable_params} learnable)")

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            predictions = p.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # some models return tuple(output)
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([-1, num_logits])  # [batch_size, -1]

            if model_args.output_each_part_logits:
                num_part = num_logits // num_labels - 1
                assert num_logits == num_labels * (num_part + 1)
                num_logits = num_labels
                logits = logits[:, :num_logits]  # [batch_size, num_labels]

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([1, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # set hyperparameter based on dev dataset
    if (
        "multiTokenDiscrimination" in model_args.few_shot_type
        and hasattr(model, "loss_weight")
        and all(model.loss_weight.detach().cpu() == torch.zeros(3))
    ):
        logger.info("grid search on dev dataset")
        trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
        best_a, best_b, best_c = -9999, -9999, -9999
        best_eval_acc, best_loss = -999999, 999999
        eval_key = "eval_acc" if data_args.task_name not in ["mrpc", "qqp"] else "eval_f1"
        if data_args.task_name == "mnli":
            eval_key = "eval_mnli/acc"
        if data_args.task_name == "cola":
            eval_key = "eval_mcc"
        if data_args.task_name in ["snli", "qnli", "mnli", "rte", "mrpc", "qqp"]:
            for b in np.linspace(0, 1, 30):
                a = (1.0 - b) / 2
                c = a
                if c >= 0:
                    model.loss_weight = torch.nn.Parameter(torch.tensor([a, b, c]), requires_grad=False)
                    output = trainer.evaluate(eval_dataset=eval_dataset)
                    eval_result = output.metrics
                    if eval_result[eval_key] > best_eval_acc:
                        best_eval_acc = eval_result[eval_key]
                        best_a, best_b, best_c = a, b, c
                    # elif eval_result[eval_key] == best_eval_acc and eval_result['eval_loss'] < best_loss:
                    #     best_loss = eval_result['eval_loss']
                    #     best_a, best_b, best_c = a,b,c
        elif data_args.task_name in ["sst-2", "sst-5", "mr", "cr", "trec", "subj", "cola"]:
            for a in np.linspace(0, 1, 30):
                b = 1.0 - a
                if b >= 0:
                    model.loss_weight = torch.nn.Parameter(torch.tensor([a, b, -9999]), requires_grad=False)
                    output = trainer.evaluate(eval_dataset=eval_dataset)
                    eval_result = output.metrics
                    if eval_result[eval_key] > best_eval_acc:
                        best_eval_acc = eval_result[eval_key]
                        best_a, best_b = a, b
                    # elif eval_result[eval_key] == best_eval_acc and eval_result['eval_loss'] < best_loss:
                    #     best_loss = eval_result['eval_loss']
                    #     best_a, best_b = a, b
        else:
            raise NotImplementedError
        logger.info(
            f"best dev {eval_key}={best_eval_acc} | best a={best_a} | best b={best_b} | best c={best_c} | best weight={[best_a, best_b, best_c]}"
        )
        model.loss_weight = torch.nn.Parameter(torch.tensor([best_a, best_b, best_c]), requires_grad=False)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))

    # eval and test
    postfix = "best" if training_args.evaluate_during_training else "last"
    logger.info(f"########################## Evaluating on {postfix} model ##########################")
    if training_args.do_train:
        # Reload the saved model
        # assert config.vocab_size == len(tokenizer)
        model = model_fn.from_pretrained(training_args.output_dir, config=config)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == "regression":
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer
    if hasattr(model, "loss_weight"):
        logger.info(f"loss_weight:{model.loss_weight}")
    # Evaluation
    final_result = {
        "time": str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [train_dataset, eval_dataset]
        dataset_name = ["_train_", "_dev_"]

        for k, eval_dataset in enumerate(eval_datasets):
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** {} results {} *****".format(dataset_name[k], eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + dataset_name[k] + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        dataset_name = ["_test_"]
        test_datasets = [test_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     test_datasets.append(
        #         FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type))
        #     )
        for k, test_dataset in enumerate(test_datasets):
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt")
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** {} results {} *****".format(dataset_name[k], test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + dataset_name[k] + key] = value

            test_results.update(test_result)

    if "multiTokenDiscrimination" in model_args.few_shot_type and hasattr(model, "loss_weight"):
        final_result["best_loss_weight_init"] = str(model.loss_weight.detach().cpu().numpy().tolist())

    final_result.update(vars(model_args))
    final_result.update(vars(training_args))
    final_result.update(vars(data_args))
    # with FileLock(f'{data_args.log_dir}.lock'):
    with open(f"{data_args.log_dir}", "a") as f:
        if "evaluation_strategy" in final_result:
            final_result.pop("evaluation_strategy")  # this may be too long
        f.write(str(final_result) + "\n")
    return


if __name__ == "__main__":
    main()
