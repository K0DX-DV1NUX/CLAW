#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=CLAW

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --train_epochs 100 \
      --rank 35 \
      --bias 0 \
      --custom_regularizer 0 \
      --kernel_size 70 \
      --patience 20 \
      --des 'Exp' \
      --itr 1 \
      --lradj type3 \
      --batch_size 32 \
      --learning_rate 0.01
done
done
done
