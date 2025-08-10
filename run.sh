#!/bin/bash
# conda init bash
# source ~/.bashrc
LOCAL_DIR="/projects/bdrx/azhang14/ReasoningGrid"
OUTPUT_DIR="/projects/bdrx/azhang14/ReasoningGrid/outputs/test"
PARTITION="gpuA100x4"
mkdir -p $OUTPUT_DIR/logs

MODELS=(
    # /projects/bdrx/azhang14/s1/ckpts/DeepSeek-R1-Distill-Qwen-1.5B_s1K-1.1-tokenized-v1-original_bs4_lr2e-5_epoch5_wd1e-4_20250803-195539/checkpoint-1250
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # knoveleng/Open-RS3
    # knoveleng/Open-RS2
    # knoveleng/Open-RS1
    # agentica-org/DeepScaleR-1.5B-Preview
    # simplescaling/s1.1-7B
    # open-thoughts/OpenThinker-7B
    # Intelligent-Internet/II-Thought-1.5B-Preview
    # sail/Qwen2.5-Math-1.5B-Oat-Zero
    # simplescaling/s1.1-32B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # open-thoughts/OpenThinker-32B
    # GAIR/LIMO
    # bespokelabs/Bespoke-Stratos-32B
)

# MAX_NUM_SEQUENCES=(128 256 512)
    

# MAX_NUM_BATCHED_TOKENS=(8192)

TOP_PS=(
    0.8 
    # 0.9 
    # 0.95 
    # 0.98 
    # 1.0
)


TEMPS=(
    0.0 
    # 0.2 
    # 0.4 
    # 0.6 
    # 0.8 
    # 1.0
)

DTYPES=(
    # "bfloat16" 
    # "float16" 
    "float32"
)

MAX_MODEL_LENGTHS=(
    32768
)

MAX_TOKENS_LIST=(
    32768
)

for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
# for MAX_NUM_SEQUENCES in "${MAX_NUM_SEQUENCES[@]}"; do
# for MAX_NUM_BATCHED_TOKENS in "${MAX_NUM_BATCHED_TOKENS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
echo "Submitting $MODEL job for temperature $TEMP, top_p $TOP_P, MAX_MODEL_LENGTH $MAX_MODEL_LENGTH, MAX_TOKENS $MAX_TOKENS, dtype $DTYPE"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval-$MODEL-$SEED-$TEMP-$TOP_P-$MAX_MODEL_LENGTH-$MAX_TOKENS-$DTYPE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --no-requeue
#SBATCH --account=account_name
#SBATCH --mail-user=axzhang2002@outlook.com
#SBATCH --output=$OUTPUT_DIR/logs/%j.out
#SBATCH --error=$OUTPUT_DIR/logs/%j.err
#SBATCH --partition=$PARTITION

# export LD_LIBRARY_PATH=$HOME/glibc-2.34/lib:$LD_LIBRARY_PATH
# export PATH=$HOME/glibc-2.34/lib:$PATH
ldd --version
module load cuda/12.4
conda init
source ~/.bashrc
conda activate sober_reasoning

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

cd $LOCAL_DIR

set -x

SEEDS=(
    0 1 2 3 4 42 100 123 2023 777
)

TASKS=(
    "custom|aime24|0|0"
    # "custom|math_500|0|0"
    # "custom|amc23|0|0"
    # "custom|aime25|0|0"
    # "custom|gpqa:diamond|0|0"
    # "custom|minerva|0|0"
    # "custom|olympiadbench|0|0"
)
for SEED in "\${SEEDS[@]}"; do
for TASK in "\${TASKS[@]}"; do
    python main.py \
        --model $MODEL \
        --task \$TASK \
        --temperature $TEMP \
        --top_p $TOP_P \
        --seed \$SEED \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --use_chat_template \
        --dtype $DTYPE \
        --swap_space 16 
done
done
EOT

done
done
done
done
done
done
# done
# done