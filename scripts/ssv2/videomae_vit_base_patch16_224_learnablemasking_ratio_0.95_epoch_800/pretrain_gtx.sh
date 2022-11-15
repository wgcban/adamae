# Set the path to save checkpoints
OUTPUT_DIR='experiments/ssv2_videomae_pretrain_pretrain_videomae_base_patch16_224_learnable_masking_ratio_0.95_e1600_final_ablations/'
# Set the path to SSV2 train set. 
DATA_PATH='datasets/ss2/list_ssv2_gtx/train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        pretrain_mae_vit.py \
        --data_path ${DATA_PATH} \
        --mask_type learnable \
        --mask_ratio 0.95 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 20 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 2400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}

#Use following command to forward the port in the remote server
#Forward port 6006 in server machine to 16006 in local machine
#ssh -L 16006:127.0.0.1:6006 chaminda@10.26.12.241