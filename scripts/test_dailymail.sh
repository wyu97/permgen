#!/bin/bash

MODEL_PATH=MODEL_FROM_BEST_DEV_EPOCH

python -u ../models/main.py \
    --data_dir ../dataset/dailymail_permute \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ../outputs/dailymail_out \
    --max_source_length 60 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --fp16 \
    --do_predict \
    --per_device_eval_batch_size 35 \
    --eval_beams 4 \
    --predict_with_generate \
    --overwrite_output_dir \
    --n_sample 4 \
    --k_out 4 \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \