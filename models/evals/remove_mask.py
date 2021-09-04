import os
import numpy as np
from collections import defaultdict

special_tokens = {'<P{}>'.format(i):i for i in range(20)}
special_tokens_list = ['<E{}>'.format(i) for i in range(20)] + ['<P{}>'.format(i) for i in range(20)]


def remove_mask(fin):

    fin = open(fin, 'r').readlines()

    outputs = []
    for line in fin:
        line = line.strip('\n').split()
        special_token_locs = [idx for idx, token in enumerate(line) if type(special_tokens.get(token)) == int]
        
        mapping_token_to_seg = {}
        for i in range(len(special_token_locs)-1):
            mapping_token_to_seg[line[special_token_locs[i]]] = line[special_token_locs[i]+1: special_token_locs[i+1]] 
        mapping_token_to_seg[line[special_token_locs[-1]]] = line[special_token_locs[-1]+1 :]

        sentence = []
        for j in range(len(mapping_token_to_seg)):
            segment = mapping_token_to_seg.get('<P{}>'.format(j))
            if segment:
                sentence += segment
        sentence = [word for word in sentence if word not in special_tokens_list]
        outputs.append(' '.join(sentence))
    return outputs