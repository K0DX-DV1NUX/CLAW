#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=ModelX

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

for seq_len in 336 512 720
do
for pred_len in 48 96 192 336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 0 \
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
      --train_epochs 50 \
      --rank 35 \
      --bias 0 \
      --sym_regularizer 1 \
      --decomposer_depth 2 \
      --seasons 3 \
      --kernel_size 70 \
      --patience 10 \
      --des 'Exp' \
      --regularizer 0 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.01
done
done
done

