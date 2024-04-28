""""
code for analysis experiments. 
corresponding to the '5.4 Reject Option: Unanimous vs. Disagreed Examples' section in the paper.
"""

import pickle
from src.processors import compute_metrics_mapping


def analyse(task_name, seed, num_part):
    assert num_part == 2 or num_part == 3
    output_dir = f"each_part_test_output/{task_name}_seed_{seed}_test_output.pkl"
    output = pickle.load(open(output_dir, "rb"))
    original_metric = output.metrics
    print("original result:", original_metric)
    print("output.prediction size:", output.predictions.shape)
    label_ids = output.label_ids
    all_logit = output.predictions  # [num_data, num_labels * num_part]
    num_labels = all_logit.shape[-1] // (num_part + 1)
    assert num_labels * (num_part + 1) == all_logit.shape[-1]

    original_predict = all_logit[:, :num_labels].argmax(-1)
    sent0_predict = all_logit[:, num_labels : num_labels * 2].argmax(-1)
    mask_predict = all_logit[:, num_labels * 2 : num_labels * 3].argmax(-1)
    if num_part == 3:
        sent1_predict = all_logit[:, num_labels * 3 : num_labels * 4].argmax(-1)

    print("original metrics = ", compute_metrics_mapping[task_name](task_name, original_predict, label_ids))
    print("only use sent0 metrics = ", compute_metrics_mapping[task_name](task_name, sent0_predict, label_ids))
    print("only use mask metrics = ", compute_metrics_mapping[task_name](task_name, mask_predict, label_ids))
    if num_part == 3:
        print("only use sent1 metrics = ", compute_metrics_mapping[task_name](task_name, sent1_predict, label_ids))

    if num_part == 2:
        same_condition = mask_predict == sent0_predict
        diff_condition = ~same_condition
    else:
        same_condition = (mask_predict == sent0_predict) & (mask_predict == sent1_predict)
        diff_condition = ~same_condition

    same_part_predict = mask_predict[same_condition]
    same_part_label = label_ids[same_condition]
    print(
        f"same part = {len(same_part_predict)}/{len(mask_predict)} | same part metrics = {compute_metrics_mapping[task_name](task_name, same_part_predict, same_part_label)}"
    )

    diff_part_predict = mask_predict[diff_condition]
    diff_part_label = label_ids[diff_condition]
    print(
        f"diff part = {len(diff_part_predict)}/{len(mask_predict)} | diff part acc = {compute_metrics_mapping[task_name](task_name, diff_part_predict, diff_part_label)}"
    )


task2num_part = {
    "sst-2": 2,
    "sst-5": 2,
    "mr": 2,
    "cr": 2,
    "snli": 3,
    "qnli": 3,
    "rte": 3,
    "mnli": 3,
    "mrpc": 3,
    "qqp": 3,
}

taskSeed2weight = {}
result_list = []
with open("log_analyzeEachPart") as f:
    for line in f:
        result_list.append(eval(line))

for line, item in enumerate(result_list):
    task_name = item["task_name"]
    best_loss_weight_init = [float(num) for num in item["best_loss_weight_init"][1:-1].split(",")]
    seed = item["seed"]
    taskSeed2weight[task_name + str(seed)] = best_loss_weight_init

for task_name in ["sst-2", "sst-5", "mr", "cr", "snli", "qnli", "rte", "mnli", "mrpc", "qqp"]:
    for seed in [13, 21, 42, 87, 100]:
        best_loss_weight_init = taskSeed2weight[task_name + str(seed)]
        if not any([i == 0 for i in best_loss_weight_init]):
            print("*" * 50)
            print("loss weight:", best_loss_weight_init)
            print(f"task={task_name} | seed={seed}")
            analyse(task_name, seed, num_part=task2num_part[task_name])
