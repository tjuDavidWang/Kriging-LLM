model_name=KrigingLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=02000
num_process=4
batch_size=48
d_model=32
d_ff=128

comment='TimeLLM-Metr'
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data Metr-LA \
  --features M \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 24 \
  --factor 3 \
  --enc_in 207 \
  --dec_in 207 \
  --c_out 207 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
