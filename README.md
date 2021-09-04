# Sentence-Permuted Paragraph Generation

This repository contains the code package for the **EMNLP'2021** paper:

**[Sentence-Permuted Paragraph Generation](https://arxiv.org/pdf/2104.07228.pdf)** [arXiv]() [slides]() [video]()

[Wenhao Yu](https://wyu97.github.io/) (ND), [Chenguang Zhu](https://www.microsoft.com/en-us/research/people/chezhu/) (MSR), [Tong Zhao](https://tzhao.io/) (ND), [Zhichun Guo](https://scholar.google.com/citations?user=BOFfWR0AAAAJ&hl=en&oi=ao) (ND), [Meng Jiang](http://meng-jiang.com/) (ND).

In this paper, we propose a novel framework PermGen whose objective is to maximize the expected log-likelihood of output paragraph distributions with respect to all possible sentence orders. PermGen uses hierarchical positional embedding and designs new procedures for training, and decoding. Experiments on three paragraph generation benchmarks demonstrate PermGen generates more diverse outputs with a higher quality than existing models.

## Usage 

```
python models/main.py \
    --data_dir $DATA_PATH \
    --model_name_or_path facebook/bart-base \
    --output_dir $OUTPUT_PATH \
    --max_source_length $INPUT_MAX_LENGTH \
    --max_target_length $OUTPUT_MAX_LENGTH \
    --val_max_target_length $OUTPUT_MAX_LENGTH \
    --num_train_epochs 25 \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size 60 \
    --per_device_eval_batch_size 50 \
    --eval_beams $BEAM_SIZE \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --n_sample $SAMPLE_NUMBER \
    --k_out $OUTPUT_NUMBER \
```

TODO, coming very soon!

## Examples

TODO! Coming very soon!


## Citation
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
