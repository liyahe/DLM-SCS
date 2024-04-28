for task in SST-2 sst-5 mr cr SNLI MNLI QNLI RTE MRPC QQP; do
    for shot in 16; do
        for seed in 13 21 42 87 100; do
            for bs in 2; do
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
                    WARMUP_RATIO=0.05 \
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
#TYPE: finetune, prompt, prompt-multiTokenDiscrimination, prompt-maskOnlyDiscrimination
