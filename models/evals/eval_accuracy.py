import os
import numpy as np
from collections import defaultdict

from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics as cie

from evals.remove_mask import remove_mask

def get_all_files(path):
    if os.path.isfile(path): return [path]
    return [f for d in os.listdir(path)
              for f in get_all_files(os.path.join(path, d))]


def eval_top1_acc(pred, ref, step):

    all_texts = remove_mask(pred)
    top1 = [all_texts[i] for i in range(0, len(all_texts), step)]
    refs = open(ref, 'r').readlines()
    metrics_dict = compute_metrics(hypothesis=top1, references=refs)
    metrics_dict = {f'top1_{k}': v for k, v in metrics_dict.items()}

    return metrics_dict


def eval_topk_acc(pred, ref, step):

    ref_by_idx = open(ref, 'r').readlines()
    all_texts = remove_mask(pred)
    gen_by_idx = [all_texts[i: i+step] for i in range(0, len(all_texts), step)]
    gens = []
    for i in range(len(ref_by_idx)):
        ref = ref_by_idx[i]
        gen = gen_by_idx[i]
        metric = [cie(ref, g)['Bleu_4'] for g in gen]
        gens.append(gen[np.argmax(metric)])
    metrics_dict = compute_metrics(hypothesis=gens, references=ref_by_idx)
    metrics_dict = {f'topk_{k}': v for k, v in metrics_dict.items()}

    return metrics_dict
