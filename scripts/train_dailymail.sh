#!/bin/bash

python -u ../models/main.py \
    --data_dir ../dataset/dailymail_permute \
    --model_name_or_path facebook/bart-base \
    --output_dir ../outputs/dailymail_out \
    --max_source_length 60 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --num_train_epochs 25  \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size 80 \
    --per_device_eval_batch_size 35 \
    --eval_beams 4 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --n_sample 4 \
    --k_out 4 \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \