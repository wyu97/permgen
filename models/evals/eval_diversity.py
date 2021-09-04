import os
import sys
import nltk
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool
from itertools import chain
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
cm = SmoothingFunction()

from evals.remove_mask import remove_mask


def get_self_bleu(hyps):
    
    sent_bleu_list = []
    for i in range(len(hyps)):
        h = hyps[i]
        r = hyps[:i] + hyps[i + 1:]
        sent_bleu_list.append(
            sentence_bleu(references=r, 
                          hypothesis=h, 
                          weights=(.25, .25, .25, .25), 
                          smoothing_function=cm.method2))
    return np.mean(sent_bleu_list)


def self_bleu(hyp_list, n_process=4):

    assert len(set([len(h) for h in hyp_list])) == 1

    if n_process > len(hyp_list):
        n_process = len(hyp_list)

    with Pool(n_process) as pool:
        self_bleu_list = list(pool.imap(get_self_bleu, zip(*hyp_list)))

    return np.mean(self_bleu_list)


def cal_entropy(generated):
    
    div_scores = {}
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip('2').split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    
    for n in range(4):
        etp_score = 0
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_scores[f'Div_entropy_{n+1}'] = etp_score

    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        div_scores[f'Div_distint_{n+1}'] = (len(counter[n].values())+0.0) / total

    return div_scores


def eval_diversity(pred, step):

    all_text = remove_mask(pred)
    div_metrics = cal_entropy(all_text)

    # sequence_groups = [all_text[i: i+step] for i in range(0, len(all_text), step)]
    # sum_self_bleu = []
    # for group in sequence_groups:
    #     avg_self_bleu = self_bleu([[line.strip('\n').split()] for line in group])
    #     sum_self_bleu.append(avg_self_bleu)
    # div_metrics['self_bleu'] = np.mean(sum_self_bleu)

    return div_metrics