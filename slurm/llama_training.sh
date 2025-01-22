

# Note 
# 1. This script should be run on the interative mode, after allocated nodes with salloc command
# 2. Everytime you relaunch hvac_server executables, the *.ports.cfg file should be deleted beforehand executing the executable. Otherwise, the server information will be accumulated onto existing ports.cfg. files, leading for client to fail finding the server. 
# 3. You should run it on ROOT folder of llama_recipe
# 4. You should be granted for the llama model from Huggingface before running the script with the specific model: https://huggingface.co/meta-llama/Llama-3.2-1B


# Set environment variable
export HVAC_SERVER_COUNT=2
export HVAC_CLIENT_PER_NODE=1
export HVAC_CHECKPOINT_DIR=/scratch/s5104a20/llama-recipes/chkpt

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llama_env

# Launch HVAC server on the background before launching 
srun -n 2 --ntasks-per-node 1 -c 2 --gpus=0 /scratch/s5104a20/renew_hvac/hvac_f/b/src/hvac_server 2 &

# This is the command that runs training code with hvac_client with two nodes each with one rank. 
LD_PRELOAD=/scratch/s5104a20/renew_hvac/hvac_f/b/src/libhvac_client.so srun -n 2 --ntasks-per-node 1 -c 2 --export=ALL torchrun --nnodes 2 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 recipes/quickstart/finetuning/finetuning.py  --enable-fsdp --dataset grammar_dataset --model_name meta-llama/Llama-3.2-1B --output_dir /scratch/s5104a20/llama-recipes/model/PEFT

# Before running recovery code, the backedup metadata file should be copy to the checkpoint directory 
cp -f .metadata_1B_2_1 /scratch/s5104a20/llama-recipes/chkpt/fine-tuned-meta-llama/Llama-3.2-1B/.metadata


# Training code that loads checkpoint before stepping into training loop (--load_chkpt flag should be set) 
LD_PRELOAD=../renew_hvac/hvac_f/b/src/libhvac_client.so srun -n 1 --ntasks-per-node 1 -c 1 --export=ALL torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 recipes/quickstart/finetuning/finetuning.py  --enable-fsdp --dataset grammar_dataset --model_name meta-llama/Llama-3.2-1B --output_dir /scratch/s5104a20/llama-recipes/model/PEFT --load_chkpt



