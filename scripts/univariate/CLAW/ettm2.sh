#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=CLAW

root_path_name=../dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

# Best results can be acquired by combinations of the following parameters:
# seq_len: 336, 512, 720
# filter_size: 8, 16, 32
# filters: 1, 2, 4
# extractor_depth: 2, 4, 6
# rank: 15, 25

for pred_len in 48 96 192 336 512 720
do
for seq_len in 336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --rank 15 \
      --filters 2 \
      --filter_size 8 \
      --extractor_depth 3 \
      --train_epochs 50 \
      --patience 10 \
      --des 'Exp' \
      --itr 1 \
      --lradj 'type7' \
      --batch_size 32 \
      --num_workers 0 \
      --learning_rate 0.001
done
done


