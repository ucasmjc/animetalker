export WANDB_BASE_URL="https://api.wandb.ai"
#export WANDB_MODE=online
export WANDB_MODE=offline
export WANDB_API_KEY=0f9e4269eec620b5201843ea9fe23e73c8a14b66
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576  # 1MB
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
results_name="face_mask+flow_loss_fps25"
torchrun --nnodes 1 --nproc_per_node 8  --node_rank=0 --master_addr=127.0.0.1 --master_port=24431 \
    fastvideo/train.py \
    --seed 142 \
    --pretrained_model_name_or_path /mnt/data/mjc/Index-anisora/ \
    --model_type "wan" \
    --data_json_path /mnt/data/mjc/Index-anisora/anisoraV2_gpu/data0706.pkl \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 240 \
    --sp_size 1 \
    --master_weight_type fp32 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=10000 \
    --learning_rate=1e-4 \
    --mixed_precision=bf16 \
    --checkpointing_steps=400 \
    --validation_steps 20000 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=results/$results_name \
    --tracker_project_name sp8 \
    --validation_guidance_scale "1.0" #\
    # --group_frame
    #--use_cpu_offload \