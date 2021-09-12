# Sentence-Permuted Paragraph Generation

This repository contains the code package for the **EMNLP'2021** paper:

**[Sentence-Permuted Paragraph Generation](https://arxiv.org/pdf/2104.07228.pdf)** \[[arXiv](https://arxiv.org/pdf/2104.07228.pdf)\] \[[slides]()\] \[[video]()\]

[Wenhao Yu](https://wyu97.github.io/) (ND), [Chenguang Zhu](https://www.microsoft.com/en-us/research/people/chezhu/) (MSR), [Tong Zhao](https://tzhao.io/) (ND), [Zhichun Guo](https://scholar.google.com/citations?user=BOFfWR0AAAAJ&hl=en&oi=ao) (ND), [Meng Jiang](http://meng-jiang.com/) (ND).

In this paper, we propose a novel framework PermGen whose objective is to maximize the expected log-likelihood of output paragraph distributions with respect to all possible sentence orders. PermGen uses hierarchical positional embedding and designs new procedures for training, and decoding. Experiments on three paragraph generation benchmarks demonstrate PermGen generates more diverse outputs with a higher quality than existing models.

## Model Usage 

### Step 1: Download datasets
We conducted experiments on three paragraph generation tasks: story generation (ROCStory), news generation (DailyMail), paper abstract generation (AGENDA). For the ROCStory and AGENDA datasets, we directly download them from their official repos. For the DailyMail dataset, We use randomly sampled 53,102 news articles from the original corpus and extract keyphrases from each sentence using [RAKE](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.657.8134&rep=rep1&type=pdf). 


| Dataset Name | Original Link | Paper Link | Our Pre-processed | 
| ---------- | :-----------:  | :-----------: | :-----------: |
| ROCStory | [OL-ROC](https://bitbucket.org/VioletPeng/language-model/src/master/) | [PL-ROC](https://arxiv.org/pdf/1811.05701.pdf) | [OP-ROC](https://drive.google.com/drive/folders/1hQ4OMdJZCe9DhzLpv5Wkg-2rufVFePpE?usp=sharing) |
| AGENDA | [OL-AG](https://github.com/rikdz/GraphWriter) | [PL-AG](https://arxiv.org/pdf/1904.02342.pdf) | [OP-AG](https://drive.google.com/drive/folders/1ydkQSBuHlkteGN07Ul57_Qdz64N2zTJu?usp=sharing) |
| DailyMail | [OL-DM](https://www.tensorflow.org/datasets/catalog/cnn_dailymail) | [PL-DM](https://arxiv.org/pdf/1704.04368.pdf) | [OP-DM](https://drive.google.com/drive/folders/1GXColf7nfNAC5E0NCGBHgijzwR8wQqUj?usp=sharing) |

After downloading the pre-processed datasets, please put them in the `dataset` folder. 


### Step 2: Install packages
The python version should be at least 3.6.0.
```
conda create -n permgen python=3.6
conda activate permgen
pip install transformers==3.3.1
pip install torch==1.7.0
```

### Step 3: Randomly permute sentences
Add/delete `--dataset` to choose the dataset.
```
python dataset/preprocessing.py --agenda --dailymail --rocstory
```

### Step 4: Train the model
```
bash scripts/train_agenda.sh
bash scripts/train_dailymail.sh
bash scripts/train_rocstory.sh
```

### Step 5: Test with saved checkpoints
Do not forget to specify the path for saved checkpoints!
```
bash scripts/test_agenda.sh
bash scripts/test_dailymail.sh
bash scripts/test_rocstory.sh
```

## Easy-to-use baseline implementation 

The baseline BART implementation can be found at [here](https://github.com/wyu97/Easy-use-BART). The repository contains the code to reproduce the baseline performance reported in our paper. All hyperparameters and evaluations are the same as in this repository.

## Output examples

Please find our output examples in the `examples` folder.

## Reference
If you find this repository useful in your research, please consider to cite our paper:

```
@inproceedings{yu2021sentence,
  title={Sentence-Permuted Paragraph Generation},
  author={Yu, Wenhao and Zhu, Chenguang and Zhao, Tong and Guo, Zhichun and Jiang, Meng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```

## Contact
If you have any questions, please contact Wenhao Yu (wyu1@nd.edu)
