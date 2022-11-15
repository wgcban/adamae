# Set the path to save checkpoints
OUTPUT_DIR='experiments/test/'
# path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='/media/lidan/ssd2/videodata/ssv2/list_ssv2/'
# path to pretrain model 
MODEL_PATH='/media/lidan/ssd2/VideoMAEZippin/experiments/ssv2_videomae_pretrain_pretrain_videomae_base_patch16_224_learnable_masking_ratio_0.95/checkpoint-799.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    finetune_class.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    # --adaptive
    # --adaptive_mask_ratio 0.95 \

