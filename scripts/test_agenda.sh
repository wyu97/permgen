#!/bin/bash

MODEL_PATH=MODEL_FROM_BEST_DEV_EPOCH

python -u ../models/main.py \
    --data_dir ../dataset/agenda_permute \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ../outputs/agenda_out \
    --max_source_length 100 \
    --max_target_length 200 \
    --val_max_target_length 200 \
    --fp16 \
    --do_predict \
    --per_device_eval_batch_size 50 \
    --eval_beams 3 \
    --predict_with_generate \
    --overwrite_output_dir \
    --n_sample 3 \
    --k_out 3 \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \