#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=ModelX

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=512
pred_len=192


for rank in 25 35 45 55 65 75
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
      --train_epochs 30 \
      --rank $rank \
      --bias 0 \
      --sym_regularizer 1 \
      --decomposer_depth 2 \
      --seasons 3 \
      --kernel_size 50 \
      --patience 6 \
      --des 'Exp' \
      --regularizer 0 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.01
done



data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2


for rank in 25 35 45 55 65 75
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
      --train_epochs 30 \
      --rank $rank \
      --bias 1 \
      --sym_regularizer 1 \
      --decomposer_depth 2 \
      --seasons 3 \
      --kernel_size 50 \
      --patience 6 \
      --des 'Exp' \
      --regularizer 0 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.01
done
