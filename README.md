# DLM-SCS (**D**iscriminative **L**anguage **M**odel as **S**emantic **C**onsistency **S**corer for Prompt-based Few-Shot Text Classification)

This is the implementation of the LREC-COLING 2024 paper [Discriminative Language Model as Semantic Consistency Scorer for Prompt-based Few-Shot Text Classification](the_paper_will_be_published_after_the_conference). DLM-SCS is short for **D**iscriminative **L**anguage **M**odel as **S**emantic **C**onsistency **S**corer.

*The project is based on the previous repo [LM-BFF](https://github.com/princeton-nlp/LM-BFF)*

## Quick links

- [DLM-SCS (**D**iscriminative **L**anguage **M**odel as **S**emantic **C**onsistency **S**corer for Prompt-based Few-Shot Text Classification)](#dlm-scs-discriminative-language-model-as-semantic-consistency-scorer-for-prompt-based-few-shot-text-classification)
  - [Quick links](#quick-links)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Prepare the data](#prepare-the-data)
  - [Run DLM-SCS](#run-dlm-scs)
    - [Quick start](#quick-start)
    - [Experiments with multiple runs](#experiments-with-multiple-runs)
      - [Run original DLM-SCS](#run-original-dlm-scs)
      - [Run DLM-SCS ablation (w.o. token weight)](#run-dlm-scs-ablation-wo-token-weight)
      - [Run DLM-SCS ablation (only label word)](#run-dlm-scs-ablation-only-label-word)
      - [Run finetune](#run-finetune)
      - [Run LM-BFF](#run-lm-bff)
    - [Gather Results](#gather-results)
  - [Bugs or questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

![comparision of DLM-SCS and previous work](figs/framework.svg)

A successful prompt-based finetuning method should have three prerequisites: task compatibility, input compatibility, and evidence abundance. Bearing this belief in mind, this paper designs a novel prompt-based method (called DLM-SCS) for few-shot text classification, which utilizes the discriminative language model ELECTRA that is pretrained to distinguish whether a token is original or replaced. The method is built upon the intuitive idea that the prompt instantiated with the true label should have higher semantic consistency score than other
prompts with false labels.

## Requirements

To run our code, please install all the dependency packages by using the following command:

```bash
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

The [LM-BFF](https://github.com/princeton-nlp/LM-BFF) pack the original datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP, STS-B) [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original` or just run the script

```bash
cd data
bash download_dataset.sh 
```

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples. You can use the following command to check whether the generated data are exactly the same as ours:

```bash
cd data/k-shot
md5sum -c ../checksum
```

## Run DLM-SCS

### Quick start

Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `3.4.0` version. Other versions of `transformers` might cause unexpected errors.

Before running any experiments, create the result folder by `mkdir result` to save checkpoints. Then you can run our code with the following example:

```bash
python run.py \
    --task_name $TASK \
    --data_dir $DATA_DIR \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path $MODEL \
    --few_shot_type $TYPE \
    --num_k $K \
    --max_seq_length 128 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size $REAL_BS \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps $GS \
    --learning_rate $LR \
    --logging_steps $EVAL_STEP \
    --eval_steps $EVAL_STEP \
    --num_train_epochs $TRAIN_EPOCH \
    --warmup_ratio $WARMUP_RATIO \
    --loss_weight_lr $LOSS_WEIGHT_LR \
    --output_dir $OUTPUT/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
    --seed $SEED \
    --tag $TAG \
    --overwrite_cache \
    --template $TEMPLATE \
    --mapping $MAPPING \
    --fix_layers $FIX_LAYERS \
    --output_each_part_logits $OUTPUT_EACH_PART_LOGITS \
    --compute_tf_idf $COMPUTE_TF_IDF \
    --log_dir $LOG_DIR \
```

Most arguments are inherited from `transformers` and are easy to understand. We further explain some of the DLM-SCS's arguments:

- `few_shot_type`: There are four modes
  - `finetune`: Standard fine-tuning
  - `prompt`: Prompt-based fine-tuning for RoBERTa or BERT.
  - `prompt-multiTokenDiscrimination`: Prompt-based fine-tuning with ELECTRA, multi token discrimination, this is exactly the proposed DLM-SCS.
  - `prompt-maskOnlyDiscrimination`: Prompt-based fine-tuning with ELECTRA, label word token discrimination, this is DLM-SCS (-only label word).
- `num_k`: Number of training instances for each class. We take `num_k`=16 in our paper. This argument is mainly used for indexing logs afterwards (because the training example numbers are actually decided by the data split you use).
- `template`: Template for prompt-based fine-tuning. We will introduce the template format later.
- `mapping`: Label word mapping for prompt-based fine-tuning. It is a string of dictionary indicating the mapping from label names to label words. **NOTE**: For RoBERTa, the model will automatically add space before the word. See the paper appendix for details.
- `loss_weight_lr`: only for DLM-SCS, loss_weight is the weight of each sentence part, this is the learning rate of the weight.
- `output_each_part_logits`: only for DLM-SCS, weather to output predicted logits of each sentence part. 1 for True and 0 for False.
- `compute_tf_idf`: only for DLM-SCS. Whether to use TF-IDF weighting for different tokens. If set to 1, it indicates the use of TF-IDF weighting. If set to 0, it means that TF-IDF weighting is not used, which corresponds to the ablation experiment of DLM-SCS (-w.o. token weight).

The above mainly explains the program arguments. For actual execution, it is recommended to use the pre-written batch running script, which will be introduced in the next section.

### Experiments with multiple runs

#### Run original DLM-SCS

```bash
for task in SST-2 sst-5 mr cr SNLI MNLI QNLI RTE MRPC QQP
do
  for shot in 16
  do
      for seed in 13 21 42 87 100
      do
          for bs in 2
          do
              TAG=DLMSCS-bs$bs \
              TYPE=prompt-multiTokenDiscrimination \
              MODE=k-shot \
              SHOT=$shot \
              TASK=$task \
              LOG_DIR=logs/log_dlmscs.log \
              OUTPUT=result \
              BS=$bs \
              LR=1e-5 \
              LOSS_WEIGHT_LR=0.0 \
              WARMUP_RATIO=0.0 \
              SEED=$seed \
              MODEL=google/electra-large-discriminator \
              FIX_LAYERS=0 \
              COMPUTE_TF_IDF=1 \
              OUTPUT_EACH_PART_LOGITS=0 \
              bash scripts/run_experiment.sh
          done
      done
  done
done
```

of simply run

```bash
bash scripts/batch_run_dlmscs.sh
```

#### Run DLM-SCS ablation (w.o. token weight)

In this ablation experiment, we do not apply TF-IDF weighting and instead treat all tokens equally with uniform weighting.The enviroment variables are identical to DLM-SCS except for setting COMPUTE_TF_IDF to 0.

```bash
bash scripts/batch_run_dlmscs_woTokenWeight.sh
```

#### Run DLM-SCS ablation (only label word)

In this ablation experiment, we only utilize the label word token to determine semantic consistency and perform classification.

```bash
bash scripts/batch_run_dlmscs_maskOnly.sh
```

#### Run finetune

The traditional finetuning approach without prompt.

```bash
bash scripts/batch_run_finetune.sh
```

#### Run LM-BFF

[LM-BFF](https://github.com/princeton-nlp/LM-BFF) is a prompt method based on the masked language model.

```bash
bash scripts/batch_run_lmbff.sh
```

### Gather Results

All the results will be stored in `./log`. To gather all the results, run the following command:

```bash
python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs', 'task_name': '$task', 'few_shot_type': 'prompt-multiTokenDiscrimination'}"
```

- `--log`: log file path
- `--condition`: A dictionary string containing various conditions is provided. The program will automatically search for results that meet the specified conditions from the specified log file and calculate the statistics.

Then the program will find all the trials that satisfy the condition in log file, and print the mean/std of the final results. Note that the task names are all lower-cased and if the task has more than one metric, you need to specify the major metric (used for taking the best validation trial) in the name (e.g., `mnli`, `mnli-mm`, `mrpc/acc`, `mrpc/f1`, `qqp/acc`, `qqp/f1`, `sts-b/pearson`, `sts-b/spearman`).

Simply, you can also run

```bash
bash scripts/gather_result.sh
```

the script will collect the results of all methods.

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to contact [email](yaheli21@m.fudan.edu.cn). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use DLM-SCS in your work:

```bibtex
@inproceedings{xie2024discriminative,
   title={Discriminative Language Model as Semantic Consistency Scorer for Prompt-based Few-Shot Text Classification},
   author={Xie, Zhipeng and Li, Yahe},
   booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation(LREC-COLING 2024)},
   year={2024}
}
```
