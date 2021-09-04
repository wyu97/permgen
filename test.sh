#!/bin/bash
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu@qa-v100-002
#$ -pe smp 1
#$ -l gpu=0

CUDA_VISIBLE_DEVICES=3 /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/bart/bin/python -u main.py \
    --data_dir dataset/titlegraphwrite_data_permute \
    --model_name_or_path facebook/bart-base \
    --output_dir outputs/titlegraphwrite_ftest \
    --max_source_length 100 \
    --max_target_length 200 \
    --val_max_target_length 200 \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --eval_beams 3 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --n_sample 3 \
    --k_out 3 \
    --n_train 10 \
    --n_val 10 \
    # --top_k 0 \
    # --top_p 1.0 \
    # --do_sample \
    # --evaluate_during_training \
    # --prediction_loss_only \
    # --n_val 1000 \
    # "$@"
