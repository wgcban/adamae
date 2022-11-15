# Set the path to save checkpoints
OUTPUT_DIR='experiments/ssv2_videomae_pretrain_pretrain_videomae_base_patch16_224_learnable_masking_ratio_0.95_wo_normalize/'
# Set the path to SSV2 train set. 
DATA_PATH='/data/wbandar1/datasets/ss2/something-something-v2-train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=65354\
        pretrain_mae_vit.py \
        --data_path ${DATA_PATH} \
        --mask_type learnable \
        --mask_ratio 0.95 \
        --normlize_target False \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 40 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 100 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
