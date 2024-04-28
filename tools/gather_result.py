import argparse

import numpy as np

from torch import device  # this line should not be removed


def color(text, color='green'):  # or \033[32m
    color2code = {'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m', 'blue': '\033[34m',
                  'purple': '\033[35m', 'cyan': '\033[36m', 'white': '\033[37m', 'black': '\033[30m'}
    return color2code[color] + text + "\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", '-cd', type=str,
                        help="A dictionary contains conditions that the experiment results need to fulfill"
                             " (e.g., tag, task_name, few_shot_type)")

    # These options should be kept as their default values
    parser.add_argument("--log", type=str, nargs='+', default=["log_reproduce.log"], help="Log path.")
    parser.add_argument("--start_line", '-sl', type=int, default=1,
                        help='from which line to start reading the log file')
    parser.add_argument("--key", type=str, default='', help="Validation metric name")
    parser.add_argument("--test_key", type=str, default="", help="Test metric name")
    parser.add_argument("--test_key2", type=str, default="", help="Second test metric name")

    args = parser.parse_args()

    condition = eval(args.condition)

    if len(args.key) == 0:
        if condition['task_name'] == 'cola':
            args.key = 'cola_dev_eval_mcc'
            args.test_key = 'cola_test_eval_mcc'
        elif condition['task_name'] == 'mrpc/acc':
            args.key = 'mrpc_dev_eval_acc'
            args.test_key = 'mrpc_test_eval_acc'
            args.test_key2 = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'mrpc/f1':
            args.key = 'mrpc_dev_eval_f1'
            args.test_key2 = 'mrpc_test_eval_acc'
            args.test_key = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'qqp/acc':
            args.key = 'qqp_dev_eval_acc'
            args.test_key = 'qqp_test_eval_acc'
            args.test_key2 = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'qqp/f1':
            args.key = 'qqp_dev_eval_f1'
            args.test_key2 = 'qqp_test_eval_acc'
            args.test_key = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'sts-b/pearson':
            args.key = 'sts-b_dev_eval_pearson'
            args.test_key = 'sts-b_test_eval_pearson'
            args.test_key2 = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'sts-b/spearmanr':
            args.key = 'sts-b_dev_eval_spearmanr'
            args.test_key2 = 'sts-b_test_eval_pearson'
            args.test_key = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'qnli':
            args.key = 'qnli_dev_eval_acc'
            args.test_key = 'qnli_test_eval_acc'
        elif condition['task_name'] == 'sst-2':
            args.key = 'sst-2_dev_eval_acc'
            args.test_key = 'sst-2_test_eval_acc'
        elif condition['task_name'] == 'snli':
            args.key = 'snli_dev_eval_acc'
            args.test_key = 'snli_test_eval_acc'
        elif condition['task_name'] == 'mnli':
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli_test_eval_mnli/acc'
        elif condition['task_name'] == 'mnli-mm':
            condition['task_name'] = 'mnli'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli-mm_test_eval_mnli-mm/acc'
        elif condition['task_name'] == 'rte':
            args.key = 'rte_dev_eval_acc'
            args.test_key = 'rte_test_eval_acc'
        elif condition['task_name'] == 'ag_news':
            args.key = 'ag_news_dev_eval_acc'
            args.test_key = 'ag_news_test_eval_acc'
        elif condition['task_name'] == 'yahoo_answers':
            args.key = 'yahoo_answers_dev_eval_acc'
            args.test_key = 'yahoo_answers_test_eval_acc'
        elif condition['task_name'] == 'yelp_review_full':
            args.key = 'yelp_review_full_dev_eval_acc'
            args.test_key = 'yelp_review_full_test_eval_acc'
        elif condition['task_name'] == 'mr':
            args.key = 'mr_dev_eval_acc'
            args.test_key = 'mr_test_eval_acc'
        elif condition['task_name'] == 'sst-5':
            args.key = 'sst-5_dev_eval_acc'
            args.test_key = 'sst-5_test_eval_acc'
        elif condition['task_name'] == 'subj':
            args.key = 'subj_dev_eval_acc'
            args.test_key = 'subj_test_eval_acc'
        elif condition['task_name'] == 'trec':
            args.key = 'trec_dev_eval_acc'
            args.test_key = 'trec_test_eval_acc'
        elif condition['task_name'] == 'cr':
            args.key = 'cr_dev_eval_acc'
            args.test_key = 'cr_test_eval_acc'
        elif condition['task_name'] == 'mpqa':
            args.key = 'mpqa_dev_eval_acc'
            args.test_key = 'mpqa_test_eval_acc'
        else:
            raise NotImplementedError

    assert len(args.log) == 1, 'Only support one log file.'
    result_list = []
    for log_file in args.log:
        if not log_file.startswith('logs/'):
            log_file = 'logs/' + log_file
        if not log_file.endswith('.log'):
            log_file += '.log'
        with open(log_file) as f:
            for line in f:
                result_list.append(eval(line))

    # print(color('Total {} results.'.format(len(result_list)), 'blue'))
    print(color('task_name: {}'.format(condition['task_name']), 'blue'))

    seed_result = {}
    seed_best = {}
    seed_latest = {}
    seed_lines = {}

    for line_idx, item in enumerate(result_list):
        check = True

        if line_idx + 1 < args.start_line:  # check start line
            check = False

        for cond in condition:  # check condition
            if isinstance(condition[cond], list) or isinstance(condition[cond], tuple):
                if cond not in item or (item[cond] not in condition[cond]):
                    check = False
                    break
            else:
                if cond not in item or (item[cond] != condition[cond]):
                    check = False
                    break

        if check:
            seed = str(item['seed'])  # seed
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_best[seed] = item
                seed_latest[seed] = item
                seed_lines[seed] = [str(line_idx + 1) + ': ' + item['time'][:19]]
            else:
                seed_result[seed].append(item)
                if item[args.key] > seed_best[seed][args.key]:
                    seed_best[seed] = item
                if item['time'] > seed_latest[seed]['time']:
                    seed_latest[seed] = item
                seed_lines[seed].append(str(line_idx + 1) + ': ' + item['time'][:19])

    seed_num = len(seed_result)
    assert len(seed_result) == len(seed_best) == len(seed_latest) == seed_num

    for i, seed in enumerate(seed_best):
        print(color(
            "seed %s: best dev (%.5f) test (%.5f) %s | trials: %d | each trial test: %s | result lines: %s" % (
                seed,
                seed_best[seed][args.key],
                seed_best[seed][args.test_key],
                "test2 (%.5f)" % (seed_best[seed][args.test_key2]) if len(args.test_key2) > 0 else "",
                len(seed_result[seed]),
                str([round(x[args.test_key], 5) for x in seed_result[seed]]),
                str(seed_lines[seed]),
            ), 'white'))
        # s = ''
        # for k in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'eval_steps',
        #           'max_steps']:
        #     s += '| {}: {} '.format(k, seed_best[seed][k])
        # print('    ' + s)
    final_result_dev = np.zeros(seed_num)
    final_result_test = np.zeros(seed_num)
    final_result_test2 = np.zeros(seed_num)

    print(color('condition: ' + str(condition), 'blue'))
    for desc, best_or_latest in zip(['best(on dev) result: ', 'latest result: '], [seed_best, seed_latest]):
        for i, seed in enumerate(seed_best):
            final_result_dev[i] = best_or_latest[seed][args.key]
            final_result_test[i] = best_or_latest[seed][args.test_key]
            if len(args.test_key2) > 0:
                final_result_test2[i] = seed_best[seed][args.test_key2]
        s = desc + " mean +- std(avg over %d): %.1f (%.1f) (median %.1f)" % (
            seed_num,
            final_result_test.mean() * 100,
            final_result_test.std() * 100,
            np.median(final_result_test) * 100
        )
        if len(args.test_key2) > 0:
            s += " second metric: %.1f (%.1f) (median %.1f)" % (
                final_result_test2.mean() * 100, final_result_test2.std() * 100,
                np.median(final_result_test2) * 100
            )
        print(color(s, 'green'))
    print('\n')


if __name__ == '__main__':
    main()
