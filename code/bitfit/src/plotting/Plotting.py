import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


masakhaner = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]
datasets = ['wikiann', 'conll2003']

api = wandb.Api()
fully_runs = api.runs(path='antonvlasjuk/MNLP-MasakhanerNER',
                      filters={
                          "display_name": {"$regex": ".*Fully.*"}
                      })
bitfit_runs = api.runs(path='antonvlasjuk/MNLP-MasakhanerNER',
                       filters={
                           "display_name": {"$regex": ".*BitFit.*"}
                       })


def get_all_performances():
    fully = defaultdict(lambda: defaultdict(float))
    bitfit = defaultdict(lambda: defaultdict(float))

    for run in fully_runs:
        summary = run.summary
        for i, l in enumerate(masakhaner):
            key = 'test_micro_f1_' + l + '_epoch/dataloader_idx_' + str(i)
            micro_f1 = summary[key]
            fully[run.name][l] = micro_f1
        train_dataset = 'wikiann' if 'wikiann' in run.name else 'conll2003'
        fully[run.name][train_dataset] += summary['test_micro_f1_' + train_dataset + '_epoch/dataloader_idx_' + str(len(masakhaner))]

    for run in bitfit_runs:
        summary = run.summary
        for i, l in enumerate(masakhaner):
            key = 'test_micro_f1_' + l + '_epoch/dataloader_idx_' + str(i)
            micro_f1 = summary[key]
            bitfit[run.name][l] = micro_f1
        train_dataset = 'wikiann' if 'wikiann' in run.name else 'conll2003'
        bitfit[run.name][train_dataset] += summary['test_micro_f1_' + train_dataset + '_epoch/dataloader_idx_' + str(len(masakhaner))]

    return fully, bitfit


def get_respective_name(set, values):
    for name, performances in set.items():
        if values == list(performances.values()):
            return name
    return None


def get_best_results(fully, bitfit):
    fully_values = defaultdict(lambda: defaultdict(float))
    bitfit_values = defaultdict(lambda: defaultdict(float))

    for name, performances in fully.items():
        for l in performances.keys():
            fully_values[name][l] += performances[l]
    best_performance = max(float(sum(d.values())) for d in fully_values.values())
    fully_best = list(list(filter(lambda d: float(sum(d.values())) == best_performance, fully_values.values()))[0].values())

    for name, performances in bitfit.items():
        for l in performances.keys():
            bitfit_values[name][l] += performances[l]
    best_performance = max(float(sum(d.values())) for d in bitfit_values.values())
    bitfit_best = list(list(filter(lambda d: float(sum(d.values())) == best_performance, bitfit_values.values()))[0].values())

    return fully_best, bitfit_best


def get_worst_results(fully, bitfit):
    fully_values = defaultdict(lambda: defaultdict(float))
    bitfit_values = defaultdict(lambda: defaultdict(float))

    for name, performances in fully.items():
        for l in performances.keys():
            fully_values[name][l] += performances[l]
    worst_performance = min(float(sum(d.values())) for d in fully_values.values())
    fully_worst = list(list(filter(lambda d: float(sum(d.values())) == worst_performance, fully_values.values()))[0].values())

    for name, performances in bitfit.items():
        for l in performances.keys():
            bitfit_values[name][l] += performances[l]
    worst_performance = min(float(sum(d.values())) for d in bitfit_values.values())
    bitfit_worst = list(list(filter(lambda d: float(sum(d.values())) == worst_performance, bitfit_values.values()))[0].values())

    return fully_worst, bitfit_worst


def get_avg(value, name, fully, number_of_runs=18):
    number_of_runs = number_of_runs if not name in datasets \
                     else (number_of_runs / 2 if name == 'conll2003' or not fully
                     else number_of_runs / 3)
    return value / number_of_runs

def get_avg_results(fully, bitfit):
    fully_values = defaultdict(float)
    bitfit_values = defaultdict(float)

    for _, performances in fully.items():
        for l in performances.keys():
            fully_values[l] += performances[l]
    fully_values = dict(map(lambda kv: (kv[0], get_avg(kv[1], kv[0], True)), fully_values.items()))

    for _, performances in bitfit.items():
        for l in performances.keys():
            bitfit_values[l] += performances[l]
    bitfit_values = dict(map(lambda kv: (kv[0], get_avg(kv[1], kv[0], False)), bitfit_values.items()))

    return fully_values, bitfit_values


def get_avg_batch(fully, bitfit):
    fully_values = defaultdict(lambda: defaultdict(float))
    bitfit_values = defaultdict(lambda: defaultdict(float))

    for name, performances in fully.items():
        name = re.search('Batch:(\d{1,})', name).group(1)
        for l in performances.keys():
            fully_values[name][l] += performances[l]

    for batch, batch_performances in fully_values.items():
        for name, performance in batch_performances.items():
            divisor = 3 if batch == '64' or name in datasets else 6
            fully_values[batch][name] = performance / divisor


    for name, performances in bitfit.items():
        name = re.search('Batch:(\d{1,})', name).group(1)
        for l in performances.keys():
            bitfit_values[name][l] += performances[l]

    for batch, batch_performances in bitfit_values.items():
        for name, performance in batch_performances.items():
            divisor = 3 if name in datasets else 6
            bitfit_values[batch][name] = performance / divisor

    return fully_values, bitfit_values


def get_avg_dataset(fully, bitfit):
    fully_values = defaultdict(lambda: defaultdict(float))
    bitfit_values = defaultdict(lambda: defaultdict(float))

    for name, performances in fully.items():
        name = re.search('(conll2003|wikiann)', name).group(0)
        for l in performances.keys():
            fully_values[name][l] += performances[l]

    for dataset, dataset_performances in fully_values.items():
        for name, performance in dataset_performances.items():
            divisor = 6 if dataset == 'wikiann' else 9
            fully_values[dataset][name] = performance / divisor


    for name, performances in bitfit.items():
        name = re.search('(conll2003|wikiann)', name).group(0)
        for l in performances.keys():
            bitfit_values[name][l] += performances[l]

    for dataset, dataset_performances in bitfit_values.items():
        for name, performance in dataset_performances.items():
            bitfit_values[dataset][name] = performance / 9

    return fully_values, bitfit_values


def get_avg_preprocessing(fully, bitfit):
    fully_values = defaultdict(lambda: defaultdict(float))
    bitfit_values = defaultdict(lambda: defaultdict(float))

    for name, performances in fully.items():
        name = re.search('(IGNORE|SAME|INSIDE)', name).group(0)
        for l in performances.keys():
            fully_values[name][l] += performances[l]

    for preprocessing, preprocessing_performances in fully_values.items():
        for name, performance in preprocessing_performances.items():
            divisor = 2 if name == 'wikiann' else 3 if name == 'conll2003' else 5
            fully_values[preprocessing][name] = performance / divisor


    for name, performances in bitfit.items():
        name = re.search('(IGNORE|SAME|INSIDE)', name).group(0)
        for l in performances.keys():
            bitfit_values[name][l] += performances[l]

    for preprocessing, preprocessing_performances in bitfit_values.items():
        for name, performance in preprocessing_performances.items():
            divisor = 3 if name in datasets else 6
            bitfit_values[preprocessing][name] = performance / divisor

    return fully_values, bitfit_values


def get_all_avgs():
    fully, bitfit = get_all_performances()
    fully_avg, bitfit_avg = get_avg_results(fully, bitfit)
    fully_batch_avg, bitfit_batch_avg = get_avg_batch(fully, bitfit)
    fully_dataset_avg, bitfit_dataset_avg = get_avg_dataset(fully, bitfit)
    fully_preprocessing_avg, bitfit_preprocessing_avg = get_avg_preprocessing(fully, bitfit)

    fully_dict = {
        'total': fully_avg,
        'dataset': fully_dataset_avg,
        'preprocessing': fully_preprocessing_avg,
        'batch': fully_batch_avg
    }
    bitfit_dict = {
        'total': bitfit_avg,
        'dataset': bitfit_dataset_avg,
        'preprocessing': bitfit_preprocessing_avg,
        'batch': bitfit_batch_avg
    }
    return fully_dict, bitfit_dict



def plot_total(fully, bitfit):
    total_labels = fully['total'].keys()
    total_fully_values = fully['total'].values()
    total_bitfit_values = bitfit['total'].values()

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(total_fully_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, total_fully_values, color='r', edgecolor='black', width=barWidth, label='Fully')
    plt.bar(br2, total_bitfit_values, color='b', edgecolor='black', width=barWidth, label='BitFit')

    # adding Xticks
    plt.xlabel('Total averages over all runs', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(total_fully_values))], total_labels)

    plt.legend()
    plt.savefig('./plotres/total_avg.png')


def plot_batch(fully, bitfit, batch_size):
    batch_labels = bitfit['batch'][batch_size].keys()
    batch_fully_values = list(fully['batch'][batch_size].values())
    if batch_size == '64': batch_fully_values.insert(len(batch_fully_values)-1, 0)
    batch_bitfit_values = bitfit['batch'][batch_size].values()

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(batch_fully_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, batch_fully_values, color='r', edgecolor='black', width=barWidth, label='Fully')
    plt.bar(br2, batch_bitfit_values, color='b', edgecolor='black', width=barWidth, label='BitFit')

    # adding Xticks
    plt.xlabel(f'Total averages over a batch size of {batch_size}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(batch_fully_values))], batch_labels)

    plt.legend()
    plt.savefig(f'./plotres/batch{batch_size}_avg.png')


def plot_batch_within(set, fully=True):
    batch_labels = set['batch']['16'].keys()

    batch_16 = set['batch']['16'].values()
    batch_32 = set['batch']['32'].values()
    batch_64 = list(set['batch']['64'].values())
    if fully: batch_64.insert(len(batch_64)-1, 0)

    # set width of bar
    barWidth = 0.22
    fig, ax = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(batch_16))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # make the plot
    plt.bar(br1, batch_16, color='r', edgecolor='black', width=barWidth, label='16')
    plt.bar(br2, batch_32, color='b', edgecolor='black', width=barWidth, label='32')
    plt.bar(br3, batch_64, color='g', edgecolor='black', width=barWidth, label='64')

    # adding Xticks
    name = 'Fully' if fully else 'BitFit'
    plt.xlabel(f'Total averages within different batch sizes - {name}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(batch_16))], batch_labels)

    plt.legend()
    plt.savefig(f'./plotres/within_batch_{name}_avg.png')


def plot_preprocessing(fully, bitfit, preprocessing):
    preprocessing_labels = bitfit['preprocessing'][preprocessing].keys()
    preprocessing_fully_values = list(fully['preprocessing'][preprocessing].values())
    preprocessing_bitfit_values = bitfit['preprocessing'][preprocessing].values()

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(preprocessing_fully_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, preprocessing_fully_values, color='r', edgecolor='black', width=barWidth, label='Fully')
    plt.bar(br2, preprocessing_bitfit_values, color='b', edgecolor='black', width=barWidth, label='BitFit')

    # adding Xticks
    plt.xlabel(f'Total averages over the preprocessing variant {preprocessing}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(preprocessing_fully_values))], preprocessing_labels)

    plt.legend()
    plt.savefig(f'./plotres/preprocessing{preprocessing}_avg.png')


def plot_preprocessing_within(set, fully=True):
    preprocessing_labels = set['preprocessing']['SAME'].keys()

    preprocessing_same = set['preprocessing']['SAME'].values()
    preprocessing_inside = set['preprocessing']['INSIDE'].values()
    preprocessing_ignore = list(set['preprocessing']['IGNORE'].values())

    # set width of bar
    barWidth = 0.22
    fig, ax = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(preprocessing_same))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # make the plot
    plt.bar(br1, preprocessing_same, color='r', edgecolor='black', width=barWidth, label='SAME')
    plt.bar(br2, preprocessing_inside, color='b', edgecolor='black', width=barWidth, label='INSIDE')
    plt.bar(br3, preprocessing_ignore, color='g', edgecolor='black', width=barWidth, label='IGNORE')

    # adding Xticks
    name = 'Fully' if fully else 'BitFit'
    plt.xlabel(f'Total averages within different preprocessing variants - {name}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(preprocessing_same))], preprocessing_labels)

    plt.legend()
    plt.savefig(f'./plotres/within_preprocessing_{name}_avg.png')


def plot_dataset(fully, bitfit, dataset):
    dataset_labels = bitfit['dataset'][dataset].keys()
    dataset_fully_values = list(fully['dataset'][dataset].values())
    dataset_bitfit_values = bitfit['dataset'][dataset].values()

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(dataset_fully_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, dataset_fully_values, color='r', edgecolor='black', width=barWidth, label='Fully')
    plt.bar(br2, dataset_bitfit_values, color='b', edgecolor='black', width=barWidth, label='BitFit')

    # adding Xticks
    plt.xlabel(f'Total averages over the dataset {dataset}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(dataset_fully_values))], dataset_labels)

    plt.legend()
    plt.savefig(f'./plotres/dataset{dataset}_avg.png')


def plot_dataset_within(set, fully=True):
    dataset_labels = list(set['dataset']['conll2003'].keys())
    dataset_labels[-1] = 'same'

    dataset_conll = set['dataset']['conll2003'].values()
    dataset_wikiann = set['dataset']['wikiann'].values()

    # set width of bar
    barWidth = 0.35
    fig, ax = plt.subplots(figsize=(18, 12))

    # set position of bar on X axis
    br1 = np.arange(len(dataset_conll))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, dataset_conll, color='r', edgecolor='black', width=barWidth, label='Conll2003')
    plt.bar(br2, dataset_wikiann, color='b', edgecolor='black', width=barWidth, label='WikiAnn')

    # adding Xticks
    name = 'Fully' if fully else 'BitFit'
    plt.xlabel(f'Total averages within different datasets - {name}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(dataset_conll))], dataset_labels)

    plt.legend()
    plt.savefig(f'./plotres/within_dataset_{name}_avg.png')


def plot_best(fully, bitfit):
    best_labels = list(fully[next(iter(fully))].keys())
    best_labels[-1] = 'conll2003'
    fully_best_values, bitfit_best_values = get_best_results(fully, bitfit)

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 13))

    # set position of bar on X axis
    br1 = np.arange(len(fully_best_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, fully_best_values, color='r', edgecolor='black', width=barWidth, label=get_respective_name(fully, fully_best_values))
    plt.bar(br2, bitfit_best_values, color='b', edgecolor='black', width=barWidth, label=get_respective_name(bitfit, bitfit_best_values))

    # adding Xticks
    plt.xlabel(f'Best run for each variant', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(fully_best_values))], best_labels)

    plt.legend()
    plt.savefig(f'./plotres/total_best.png')


def plot_worst(fully, bitfit):
    worst_labels = list(fully[next(iter(fully))].keys())
    worst_labels[-1] = 'wikiann'
    fully_worst_values, bitfit_worst_values = get_worst_results(fully, bitfit)

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 13))

    # set position of bar on X axis
    br1 = np.arange(len(fully_worst_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    plt.bar(br1, fully_worst_values, color='r', edgecolor='black', width=barWidth, label=get_respective_name(fully, fully_worst_values))
    plt.bar(br2, bitfit_worst_values, color='b', edgecolor='black', width=barWidth, label=get_respective_name(bitfit, bitfit_worst_values))

    # adding Xticks
    plt.xlabel(f'Worst run for each variant', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(fully_worst_values))], worst_labels)

    plt.legend()
    plt.savefig(f'./plotres/total_worst.png')


def plot_best_worst_within(fully, bitfit, fully_flag=True):
    labels = list(fully[next(iter(fully))].keys())
    labels[-1] = 'same'
    pick_idx = 0 if fully_flag else 1
    best_values, worst_values = get_best_results(fully, bitfit)[pick_idx], get_worst_results(fully, bitfit)[pick_idx]

    # set width of bar
    barWidth = 0.35
    fig = plt.subplots(figsize=(18, 13))

    # set position of bar on X axis
    br1 = np.arange(len(best_values))
    br2 = [x + barWidth for x in br1]

    # make the plot
    name = 'Fully' if fully_flag else 'BitFit'
    plt.bar(br1, best_values, color='r', edgecolor='black', width=barWidth, label='Best')
    plt.bar(br2, worst_values, color='b', edgecolor='black', width=barWidth, label='Worst')

    # adding Xticks
    plt.xlabel(f'Comparison of best and worst - {name}', fontweight='bold', fontsize=18)
    plt.ylabel('Micro F1', fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(best_values))], labels)

    plt.legend()
    plt.savefig(f'./plotres/within_best_worst_{name}.png')


def plot_all():
    fully, bitfit = get_all_avgs()

    plot_total(fully, bitfit)

    plot_dataset(fully, bitfit, 'conll2003')
    plot_dataset(fully, bitfit, 'wikiann')
    plot_dataset_within(fully, fully=True)
    plot_dataset_within(bitfit, fully=False)

    plot_preprocessing(fully, bitfit, 'SAME')
    plot_preprocessing(fully, bitfit, 'INSIDE')
    plot_preprocessing(fully, bitfit, 'IGNORE')
    plot_preprocessing_within(fully, fully=True)
    plot_preprocessing_within(bitfit, fully=False)

    plot_batch(fully, bitfit, '16')
    plot_batch(fully, bitfit, '32')
    plot_batch(fully, bitfit, '64')
    plot_batch_within(fully, fully=True)
    plot_batch_within(bitfit, fully=False)

    fully, bitfit = get_all_performances()

    plot_best(fully, bitfit)
    plot_worst(fully, bitfit)

    plot_best_worst_within(fully, bitfit, fully_flag=True)
    plot_best_worst_within(fully, bitfit, fully_flag=False)



if __name__ == '__main__':
    plot_all()
