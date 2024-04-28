# Required environment variables (NOT ALL):
# TAG: tag for the trail, describing the experiment
# TYPE: finetune / prompt / prompt-multiTokenDiscrimination / prompt-maskOnlyDiscrimination
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list
# SHOT: number of training instances per label (16 / 32 / 64 / 128 / 256)
# MODE: 'origianl' or 'k-shot', determine the data directory
# WARMUP_RATIO: warmup ratio for the learning rate scheduler
# COMPUTE_TF_IDF: whether to use TF-IDF features to gather multi token discrimination results

# Number of training instances per label
K=$SHOT

# Validation steps
case $SHOT in
16)
    EVAL_STEP=50
    TRAIN_EPOCH=15
    ;;
32)
    EVAL_STEP=100
    TRAIN_EPOCH=15
    ;;
64)
    EVAL_STEP=200
    TRAIN_EPOCH=15
    ;;
128)
    EVAL_STEP=400
    TRAIN_EPOCH=10
    ;;
256)
    EVAL_STEP=800
    TRAIN_EPOCH=10
    ;;
*)
    EVAL_STEP=-2 # -x means evaluate x times at each epoch
    TRAIN_EPOCH=10
    ;;
esac

echo "EVAL_STEP: $EVAL_STEP"
echo "TRAIN_EPOCH: $TRAIN_EPOCH"

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
case $TASK in
CoLA)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{'0':'incorrect','1':'correct'}"
    ;;
SST-2)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{'0':'terrible','1':'great'}"
    ;;
MRPC)
    TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
    MAPPING="{'0':'No','1':'Yes'}"
    TASK_EXTRA="--max_seq_len 128"
    ;;
QQP)
    TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
    MAPPING="{'0':'No','1':'Yes'}"
    TASK_EXTRA="--max_seq_len 256 --first_sent_limit 220"
    ;;
STS-B)
    TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
    MAPPING="{'0':'No','1':'Yes'}"
    ;;
MNLI)
    TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
    MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
    TASK_EXTRA="--max_seq_len 128 --first_sent_limit 110 --other_sent_limit 50"
    ;;
SNLI)
    TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
    MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
    TASK_EXTRA="--max_seq_len 128"
    ;;
QNLI)
    TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
    MAPPING="{'not_entailment':'No','entailment':'Yes'}"
    TASK_EXTRA="--max_seq_len 128 --first_sent_limit 110 --other_sent_limit 50"
    ;;
RTE)
    TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
    MAPPING="{'not_entailment':'No','entailment':'Yes'}"
    TASK_EXTRA="--max_seq_len 256 --first_sent_limit 220"
    ;;
mr)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{0:'terrible',1:'great'}"
    TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
    ;;
sst-5)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
    TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --eval_steps 100"
    ;;
subj)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{0:'subjective',1:'objective'}"
    TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
    ;;
trec)
    TEMPLATE=*cls**mask*:*+sent_0**sep+*
    MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
    TASK_EXTRA="--first_sent_limit 110"
    ;;
cr)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{0:'terrible',1:'great'}"
    TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
    ;;
mpqa)
    TEMPLATE=*cls**sent_0*_It_is*mask*.*sep+*
    MAPPING="{0:'terrible',1:'great'}"
    TASK_EXTRA="--first_sent_limit 110"
    ;;
esac

# The GLM model uses different prompt templates,
# so there is a need to modify the template,
# and the "label_word" should be in the last position.

# gpt only used for single sentence classification tasks
if [ $MODEL == 'gpt2-medium' ]; then
    echo "model is $MODEL, using $MODEL template"
    case $TASK in
    SST-2)
        TEMPLATE=*sent_0*_It_is*mask*
        ;;
    mr)
        TEMPLATE=*sent_0*_It_is*mask*
        ;;
    sst-5)
        TEMPLATE=*sent_0*_It_is*mask*
        ;;
    cr)
        TEMPLATE=*sent_0*_It_is*mask*
        ;;
    esac
fi

if [ $MODEL == 't5-large' -o $MODEL == 't5-base' -o $MODEL == 't5-3b' ]; then
    echo "model is $MODEL, using $MODEL template"
    case $TASK in
    SST-2)
        TEMPLATE=*sent_0*_It_is*\<extra_id_0\>*.*\</s\>*
        ;;
    MRPC)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    QQP)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    MNLI)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    SNLI)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    QNLI)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    RTE)
        TEMPLATE=*sent-_0*?*\<extra_id_0\>*,*+sentl_1*\</s\>*
        ;;
    mr)
        TEMPLATE=*sent_0*_It_is*\<extra_id_0\>*.*\</s\>*
        ;;
    sst-5)
        TEMPLATE=*sent_0*_It_is*\<extra_id_0\>*.*\</s\>*
        ;;
    cr)
        TEMPLATE=*sent_0*_It_is*\<extra_id_0\>*.*\</s\>*
        ;;
    esac
fi

echo "current template: $TEMPLATE", "current mapping: $MAPPING"

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2

GS=$(expr $BS / $REAL_BS)

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM

# data directory
if [ $MODE == "original" ]; then
    DATA_DIR=data/$MODE/$TASK
else
    DATA_DIR=data/$MODE/$TASK/$K-$SEED
fi
echo "Data directory: $DATA_DIR"

# result dir
mkdir -p $OUTPUT
echo "Output root directory: $OUTPUT"

# disable wandb
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0 \
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
    $TASK_EXTRA \
    $1

# Delete the checkpoint
# Since we need to run multiple trials, saving all the checkpoints takes
# a lot of storage space. You can find all evaluation results in `log` file anyway.
rm -r $OUTPUT/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF/
