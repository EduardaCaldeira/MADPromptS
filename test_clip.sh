export OMP_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_HOME="./cache/"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

declare -a backbone_sizes=("ViT-L/14")

for backbone_size in "${backbone_sizes[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=15924 \
    src/config.py \
    --debug=True \
    --backbone_size="$backbone_size" \
    --model_name="clip"
done
