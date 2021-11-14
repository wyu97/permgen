import os
import re
import random
from shutil import copyfile
from collections import defaultdict
from itertools import permutations
from functools import reduce

threshold = 5


def preprocess_dailymail(ifolder, ofolder):

    if not os.path.exists(ofolder):
        os.mkdir(ofolder)

    # pre-defined filenames for dailymail dataset
    InTrainSrc = open(os.path.join(ifolder, 'train.source'), 'r').readlines()
    InTrainTgt = open(os.path.join(ifolder, 'train.target'), 'r').readlines()
    InValidSrc = open(os.path.join(ifolder, 'val.source'), 'r').readlines()
    InValidTgt = open(os.path.join(ifolder, 'val.target'), 'r').readlines()
    InTestSrc = open(os.path.join(ifolder, 'test.source'), 'r').readlines()
    InTestTgt = open(os.path.join(ifolder, 'test.target'), 'r').readlines()

    OutTrainSrc = open(os.path.join(ofolder, 'train.source'), 'w')
    OutTrainTgt = open(os.path.join(ofolder, 'train.target'), 'w')
    OutValidSrc = open(os.path.join(ofolder, 'val.source'), 'w')
    OutValidTgt = open(os.path.join(ofolder, 'val.target'), 'w')
    OutTestSrc = open(os.path.join(ofolder, 'test.source'), 'w')
    OutTestTgt = open(os.path.join(ofolder, 'test.target'), 'w')

    def preprocess_train(InSrc, InTgt, OutSrc, OutTgt):

        special_tokens_list = ['<P{}>'.format(i) for i in range(10)]
        for src, tgt in zip(InSrc, InTgt):
            mapping_token_to_seg = defaultdict(list)
            for word in tgt.strip().split():
                if word in special_tokens_list:
                    mapping_token_to_seg[len(mapping_token_to_seg)].append(word)
                else:
                    mapping_token_to_seg[len(mapping_token_to_seg)-1].append(word)
            all_permutations = list(permutations([i for i in range(len(mapping_token_to_seg))]))

            if len(all_permutations) > threshold: 
                all_permutations = random.sample(all_permutations, threshold)
            if tuple([i for i in range(len(mapping_token_to_seg))]) not in all_permutations:
                all_permutations.append(tuple([i for i in range(len(mapping_token_to_seg))]))
            for per in all_permutations:
                permuted_tgt = reduce(lambda a, b: a + b, [mapping_token_to_seg[i] for i in per])

                permuted_tgt = '{}\n'.format(' '.join(permuted_tgt))
                OutSrc.write(src)
                OutTgt.write(permuted_tgt)

    def preprocess_dev_test(InSrc, InTgt, OutSrc, OutTgt):

        for src, tgt in zip(InSrc, InTgt):
            OutSrc.write(src)
            tgt = re.sub(r'<P.?> |<E.?> | <E.?>', '', tgt)
            OutTgt.write(tgt)

    preprocess_train(InTrainSrc, InTrainTgt, OutTrainSrc, OutTrainTgt)
    preprocess_dev_test(InValidSrc, InValidTgt, OutValidSrc, OutValidTgt)
    preprocess_dev_test(InTestSrc, InTestTgt, OutTestSrc, OutTestTgt)


def preprocess_rocstory(ifolder, ofolder):

    if not os.path.exists(ofolder):
        os.mkdir(ofolder)

    # pre-defined filenames for dailymail dataset
    InTrain = open(os.path.join(ifolder, 'ROCStories_all_merge_tokenize.titlesepkeysepstory.train'), 'r').readlines()
    InValid = open(os.path.join(ifolder, 'ROCStories_all_merge_tokenize.titlesepkeysepstory.dev'), 'r').readlines()
    InTest = open(os.path.join(ifolder, 'ROCStories_all_merge_tokenize.titlesepkeysepstory.test'), 'r').readlines()

    OutTrainSrc = os.path.join(ofolder, 'train.source')
    OutTrainTgt = os.path.join(ofolder, 'train.target')
    OutValidSrc = os.path.join(ofolder, 'val.source')
    OutValidTgt = os.path.join(ofolder, 'val.target')
    OutTestSrc = os.path.join(ofolder, 'test.source')
    OutTestTgt = os.path.join(ofolder, 'test.target')

    def preprocess_train(InFile, OutSrc, OutTgt):
        
        OutSrc = open(OutSrc, 'w')
        OutTgt = open(OutTgt, 'w')

        for line in InFile:
            line = line.strip('\n').split()

            eot_index = line.index('<EOT>')
            eol_index = line.index('<EOL>')
            storyline = line[: eol_index]
            target = line[eol_index+1: ]

            if storyline.count('#') != target.count('</s>')-1: continue

            storyline_segment = ' '.join(storyline).split(' # ')
            padded_storyline = [seg for idx, seg in enumerate(storyline_segment)]

            target_segment = ' '.join(target[1:]).split(' </s> ')
            padded_target = ['<P{}> '.format(idx) + seg + ' <E{}>'.format(idx) \
                                for idx, seg in enumerate(target_segment)]
            assert(padded_storyline and padded_target)

            OutSrc.write('{}\n'.format(' '.join(padded_storyline)))
            OutTgt.write('{}\n'.format(' '.join(padded_target)))

    def preprocess_dev_test(InFile, OutSrc, OutTgt):
        
        OutSrc = open(OutSrc, 'w')
        OutTgt = open(OutTgt, 'w')

        for line in InFile:
            line = line.strip('\n').split()

            eot_index = line.index('<EOT>')
            eol_index = line.index('<EOL>')
            storyline = line[: eol_index]
            target = line[eol_index+1: ]

            if storyline.count('#') != target.count('</s>')-1: continue

            storyline_segment = ' '.join(storyline).split(' # ')
            target_segment = ' '.join(target[1:]).split(' </s> ')
            
            OutSrc.write('{}\n'.format(' '.join(storyline_segment)))
            OutTgt.write('{}\n'.format(' '.join(target_segment)))

    def permute_train(InSrc, InTgt, OutSrc, OutTgt):

        InSrc = open(InSrc, 'r').readlines()
        InTgt = open(InTgt, 'r').readlines()
        OutSrc = open(OutSrc, 'w')
        OutTgt = open(OutTgt, 'w')

        special_tokens = ['<P{}>'.format(i) for i in range(10)]
        for _, (src, tgt) in enumerate(zip(InSrc, InTgt)):

            tgt_sent_dict = defaultdict(list)
            for word in tgt.split():
                if word in special_tokens:
                    tgt_sent_dict[len(tgt_sent_dict)].append(word)
                else:
                    tgt_sent_dict[len(tgt_sent_dict)-1].append(word)

            all_permutations = list(permutations([i for i in range(len(tgt_sent_dict))]))
            if len(all_permutations) > threshold: 
                all_permutations = random.sample(all_permutations, threshold)

            if tuple([i for i in range(len(tgt_sent_dict))]) not in all_permutations:
                all_permutations.append(tuple([i for i in range(len(tgt_sent_dict))]))

            for per in all_permutations:
                permuted_tgt = reduce(lambda a, b: a + b, [tgt_sent_dict[i] for i in per])
                permuted_tgt = '{}\n'.format(' '.join(permuted_tgt))
                OutSrc.write(src)
                OutTgt.write(permuted_tgt)

    preprocess_train(InTrain, OutTrainSrc, OutTrainTgt)
    preprocess_dev_test(InValid, OutValidSrc, OutValidTgt)
    preprocess_dev_test(InTest, OutTestSrc, OutTestTgt)
    permute_train(OutTrainSrc, OutTrainTgt, OutTrainSrc, OutTrainTgt)


def preprocess_agenda(ifolder, ofolder):

    if not os.path.exists(ofolder):
        os.mkdir(ofolder)

    # pre-defined filenames for dailymail dataset
    InTrain = open(os.path.join(ifolder, 'preprocessed.train.tsv'), 'r').readlines()
    InValid = open(os.path.join(ifolder, 'preprocessed.val.tsv'), 'r').readlines()
    InTest = open(os.path.join(ifolder, 'preprocessed.test.tsv'), 'r').readlines()

    OutTrainSrc = open(os.path.join(ofolder, 'train.source'), 'w')
    OutTrainTgt = open(os.path.join(ofolder, 'train.target'), 'w')
    OutValidSrc = open(os.path.join(ofolder, 'val.source'), 'w')
    OutValidTgt = open(os.path.join(ofolder, 'val.target'), 'w')
    OutTestSrc = open(os.path.join(ofolder, 'test.source'), 'w')
    OutTestTgt = open(os.path.join(ofolder, 'test.target'), 'w')

    def preprocess_train(InFile, OutSrc, OutTgt):

        for line in InFile:
            title, srcs, _, _, targets, _ = line.split('\t')
            splited_source = srcs.split(' ; ')

            inputs, outputs = [], []
            inputs += [title] + ['<eot>']
            for word in targets.split():
                if word.startswith('<') and word.endswith('>') and '_' in word:
                    num = int(word.strip('>').split('_')[-1])
                    inputs += [splited_source[num] + ' ;']
                    outputs += [splited_source[num]]
                else:
                    outputs += [word]
            tokenized_outputs = ' '.join(outputs).strip('.').split(' . ')
            tokenized_outputs = [i + ' .' for i in tokenized_outputs]

            padded_tokenized_target = []
            for idx, segment in enumerate(tokenized_outputs):
                padded_tokenized_target.append(['<P{}>'.format(idx)] + [segment] + ['<E{}>'.format(idx)])

            padded_tokenized_target = padded_tokenized_target[:3]
            all_permutations = list(permutations([i for i in range(len(padded_tokenized_target))]))
            if len(all_permutations) > threshold: 
                all_permutations = random.sample(all_permutations, threshold)

            if tuple([i for i in range(len(padded_tokenized_target))]) not in all_permutations:
                all_permutations.append(tuple([i for i in range(len(padded_tokenized_target))]))

            for per in all_permutations:
                permuted_tgt = reduce(lambda a, b: a + b, [padded_tokenized_target[i] for i in per])
                OutSrc.write('{}\n'.format(' '.join(inputs)))
                OutTgt.write('{}\n'.format(' '.join(permuted_tgt)))


    def preprocess_dev_test(InFile, OutSrc, OutTgt):

        for line in InFile:
            title, srcs, _, _, targets, _ = line.split('\t')
            splited_source = srcs.split(' ; ')

            inputs, outputs = [], []
            inputs += [title] + ['<eot>']
            for word in targets.split():
                if word.startswith('<') and word.endswith('>') and '_' in word:
                    num = int(word.strip('>').split('_')[-1])
                    inputs += [splited_source[num] + ' ;']
                    outputs += [splited_source[num]]
                else:
                    outputs += [word]
            tokenized_outputs = ' '.join(outputs).strip('.').split(' . ')
            tokenized_outputs = [i + ' .' for i in tokenized_outputs]

            padded_tokenized_target = []
            for idx, segment in enumerate(tokenized_outputs):
                padded_tokenized_target += [segment]
            padded_tokenized_target = padded_tokenized_target[:3]

            OutSrc.write('{}\n'.format(' '.join(inputs)))
            OutTgt.write('{}\n'.format(' '.join(padded_tokenized_target)))

    preprocess_train(InTrain, OutTrainSrc, OutTrainTgt)
    preprocess_dev_test(InValid, OutValidSrc, OutValidTgt)
    preprocess_dev_test(InTest, OutTestSrc, OutTestTgt)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='data preprocessing parameters')
    parser.add_argument('--agenda', type=bool, default=False)
    parser.add_argument('--dailymail', type=bool, default=False)
    parser.add_argument('--rocstory', type=bool, default=False)
    args = parser.parse_args()

    cur_dir = os.path.join(os.getcwd(), 'dataset')
    if args.agenda:
        preprocess_agenda(os.path.join(cur_dir, 'agenda'), os.path.join(cur_dir, 'agenda_permute'))
        print('preprocessed agenda!')
    if args.dailymail:
        preprocess_dailymail(os.path.join(cur_dir, 'dailymail'), os.path.join(cur_dir, 'dailymail_permute'))
        print('preprocessed dailymail!')
    if args.rocstory:
        preprocess_rocstory(os.path.join(cur_dir, 'rocstory'), os.path.join(cur_dir, 'rocstory_permute'))
        print('preprocessed rocstory!')