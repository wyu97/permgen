#!/bin/bash

python -u ../models/main.py \
    --data_dir ../dataset/agenda_permute \
    --model_name_or_path facebook/bart-base \
    --output_dir ../outputs/agenda_out \
    --max_source_length 100 \
    --max_target_length 200 \
    --val_max_target_length 200 \
    --num_train_epochs 25 \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size 80 \
    --per_device_eval_batch_size 50 \
    --eval_beams 3 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --n_sample 3 \
    --k_out 3 \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \
