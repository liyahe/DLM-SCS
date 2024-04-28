from dataclasses import dataclass, field
from typing import Optional

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='google/electra-large-discriminator',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-multiTokenDiscrimination: prompt-based fine-tuning with multi-token discrimination
    #   - prompt-maskOnlyDiscrimination: prompt-based fine-tuning with mask-only discrimination
    few_shot_type: str = field(
        default='prompt-multiTokenDiscrimination',
        metadata={
            "help": "Few-shot learning model type. Choice: finetune, prompt, prompt-multiTokenDiscrimination, prompt-maskOnlyDiscrimination."}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    output_each_part_logits: int = field(
        default=0,
        metadata={
            "help": 'whether to output all two(three) parts\' logits, note that the parameter only works for'
                    ' multiTokenDiscrimination model'}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )
    # For prompting
    template: str = field(
        default='*cls**sent_0*_It_is*mask*.*sep+*',
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default="{'0':'terrible','1':'great'}",  # '{"0":"terrible","1":"great"}',  # Default mapping for SST-2
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )

    compute_tf_idf: int = field(
        default=0,
    )

    log_dir: str = field(
        default='logs/log_reproduce.log'
    )

    data_dir: str = field(
        default='data/k-shot/SST-2/16-42'
    )

    task_name: str = field(
        default='sst-2'
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "warm up ratio"}
    )

    loss_weight_lr: float = field(
        default=0.0,
        metadata={"help": "Learning rate for loss weight of DLM-SCS, default 0.0 means no learning"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )

    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    output_dir: str = field(
        default='result/'
    )
