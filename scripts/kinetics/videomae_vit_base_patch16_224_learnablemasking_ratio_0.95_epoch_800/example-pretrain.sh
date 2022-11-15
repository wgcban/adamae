# Set the path to save checkpoints
OUTPUT_DIR='OUTPUT_DIR'
# Set the path to SSV2 train set. 
DATA_PATH='PATH_TO_TRAIN_CVS'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)

### mask_type = learnable and mask_ratio = 0.95 ###
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12320 --nnodes=8 \
        --node_rank=0 --master_addr=$your_node_0_ip \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type learnable \
        --mask_ratio 0.95 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}