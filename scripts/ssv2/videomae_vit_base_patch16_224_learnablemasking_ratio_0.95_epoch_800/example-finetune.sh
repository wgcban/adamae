# Set the path to save checkpoints
OUTPUT_DIR='OUTPUT_DIR'
# path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='PATH_TO_TRAIN_VAL_TEST_CVS'
# path to pretrain model 
MODEL_PATH='PATH_TO_PRETRAINED_MODEL'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 --nnodes=8 \
    --node_rank=0 --master_addr=$ip_node_0 \
    finetune_class.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --enable_deepspeed

