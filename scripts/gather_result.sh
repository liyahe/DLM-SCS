# DLM-SCS
for bs in 2; do
    for task in sst-2 snli mnli qnli qqp/f1 rte mrpc/f1 mr cr sst-5; do
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs', 'task_name': '${task}'}"
    done
done

# DLM-SCS (w.o. token weight)
for bs in 2; do
    for task in sst-2 snli mnli qnli qqp/f1 rte mrpc/f1 mr cr sst-5; do
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs-woTokenWeight', 'task_name': '${task}'}"
    done
done

# DLM-SCS (only label word)
for bs in 2; do
    for task in sst-2 snli mnli qnli qqp/f1 rte mrpc/f1 mr cr sst-5; do
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs-onlyLabelWord', 'task_name': '${task}'}"
    done
done

# finetune
for bs in 2; do
    for task in sst-2 snli mnli qnli qqp/f1 rte mrpc/f1 mr cr sst-5; do
        python tools/gather_result.py --log log_finetune --condition "{'tag': 'finetune-bs$bs', 'task_name': '${task}'}"
    done
done

# lmbff
for bs in 2; do
    for task in sst-2 snli mnli qnli qqp/f1 rte mrpc/f1 mr cr sst-5; do
        python tools/gather_result.py --log log_lmbff --condition "{'tag': 'LMBFF-bs$bs', 'task_name': '${task}'}"
    done
done
