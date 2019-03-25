#!/bin/sh

############## 4.1 ########################
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best

############## 4.2 ########################
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best

############## 4.3 #########################

## TRANSFORMER
# change in seq_len
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=50 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=20 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
# change in hidden_size 
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=768 --num_layers=6 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=6 --dp_keep_prob=0.9 --save_best
# change in num_layers
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=8 --dp_keep_prob=0.9 --save_best
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.9 --save_best

## RNN
# change in seq_len
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=20 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
# change in hidden_size
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1000 --num_layers=2 --dp_keep_prob=0.35 --save_best
# change in num_layers
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=3 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.35 --save_best

## GRU
# change in seq_len
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=20 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
# change in hidden_size
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1000 --num_layers=2 --dp_keep_prob=0.35 --save_best
# change in num_layers
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=3 --dp_keep_prob=0.35 --save_best
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.35 --save_best

############## 5.1 #########################

## RNN
python ptb-lm-val_loss_by_t.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --evaluate --model_path=4_1/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --save_dir=5_1/

## TRANSFORMER
python ptb-lm-val_loss_by_t.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --evaluate --model_path=4_1/TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_save_best_0/best_params.pt --save_dir=5_1/

## GRU
python ptb-lm-val_loss_by_t.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --evaluate --model_path=4_1/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --save_dir=5_1/

############## 5.2 #########################
python3 ptb-lm-grad.py

############## 5.3 #########################
python3 ptb-lm-sample.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=10 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --resume 4_1/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --sample

python3 ptb-lm-sample.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=10 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --resume 4_1/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --sample

python3 ptb-lm-sample.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=10 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --resume 4_1/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --sample

python3 ptb-lm-sample.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=10 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --resume 4_1/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0/best_params.pt --sample

