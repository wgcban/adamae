# Set the path to save checkpoints
OUTPUT_DIR='experiments/kinetics400_videomae_pretrain_pretrain_videomae_base_patch16_224_learnable_masking_ratio_0.95_final/'
# Set the path to SSV2 train set. 
DATA_PATH='/data/wbandar1/datasets/kinetics/k400_320p_lists/kinetics400-320p-train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,6,7,8,9 torchrun --nproc_per_node=8 --master_port=65354\
        pretrain_mae_vit.py \
        --data_path ${DATA_PATH} \
        --mask_type learnable \
        --mask_ratio 0.95 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
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