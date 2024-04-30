for bs in 2; do
    for task in snli mnli qnli qqp/f1 mrpc/f1 rte sst-2 sst-5 mr cr; do
        # DLM-SCS
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs', 'task_name': '${task}'}"

        # DLM-SCS (w.o. token weight)
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs-woTokenWeight', 'task_name': '${task}'}"

        # DLM-SCS (only label word)
        python tools/gather_result.py --log log_dlmscs --condition "{'tag': 'DLMSCS-bs$bs-onlyLabelWord', 'task_name': '${task}'}"

        # finetune
        python tools/gather_result.py --log log_finetune --condition "{'tag': 'finetune-bs$bs', 'task_name': '${task}'}"

        # lmbff
        python tools/gather_result.py --log log_lmbff --condition "{'tag': 'LMBFF-bs$bs', 'task_name': '${task}'}"
    done
done
