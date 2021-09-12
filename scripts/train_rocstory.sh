#!/bin/bash

python -u ../models/main.py \
    --data_dir ../dataset/rocstory_permute \
    --model_name_or_path facebook/bart-base \
    --output_dir ../outputs/rocstory_out \
    --max_source_length 40 \
    --max_target_length 150 \
    --val_max_target_length 150 \
    --num_train_epochs 25 \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 60 \
    --eval_beams 3 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --n_sample 5 \
    --k_out 3 \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \