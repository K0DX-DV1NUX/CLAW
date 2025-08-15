#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi



model_name=CLAW

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

# Best results can be acquired by combinations of the following parameters:
# seq_len: 336, 512, 720
# filter_size: 8, 16, 32
# filters: 1, 2, 4
# extractor_depth: 2, 4, 6
# rank: 15, 25

for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --rank 25 \
      --filters 3 \
      --filter_size 32 \
      --extractor_depth 4 \
      --train_epochs 50 \
      --patience 10 \
      --des 'Exp' \
      --itr 1 \
      --lradj 'type7' \
      --batch_size 32 \
      --num_workers 0 \
      --learning_rate 0.001 \
      --seed 2021
done
done