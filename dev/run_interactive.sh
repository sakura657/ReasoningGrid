#!/bin/bash
# conda init bash
# source ~/.bashrc
LOCAL_DIR="/projects/bdrx/azhang14/ReasoningGrid/dev"
OUTPUT_DIR="/projects/bdrx/azhang14/ReasoningGrid/test"
mkdir -p $OUTPUT_DIR/logs

module load cuda/12.6
conda init
source ~/.bashrc
conda deactivate
source /projects/bdrx/azhang14/env/test/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi topo -m

# Set path for prompts file and load SYSTEM_PROMPT
export PROMPTS_PATH="$LOCAL_DIR/prompts.json"
SYSTEM_PROMPT=$(python -c "import json; print(json.load(open('$PROMPTS_PATH'))['SYSTEM_PROMPT'])")

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

MAX_NUM_SEQUENCES=(8)
    

MAX_NUM_BATCHED_TOKENS=(65536)

TOP_PS=(
    # 0.8 
    0.9 
    # 0.95 
    # 0.98 
    # 1.0
)


TEMPS=(
    # 0.0 
    # 0.2 
    # 0.4 
    # 0.6 
    0.8 
    # 1.0
)

DTYPES=(
    "bfloat16" 
    # "float16" 
    # "float32"
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
for MAX_NUM_SEQUENCES in "${MAX_NUM_SEQUENCES[@]}"; do
for MAX_NUM_BATCHED_TOKENS in "${MAX_NUM_BATCHED_TOKENS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
cd $LOCAL_DIR

set -x

SEEDS=(
    0
    1
    2
    # 3
    # 4 
    # 42 
    # 100
    # 110
    # 123
    # 666
    # 888
    # 911
    # 999
    # 1000
    # 2023
    # 2025 
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
for SEED in "${SEEDS[@]}"; do
for TASK in "${TASKS[@]}"; do
    python main.py \
        --model $MODEL \
        --task $TASK \
        --temperature $TEMP \
        --top_p $TOP_P \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --system_prompt "$SYSTEM_PROMPT" \
        --use_chat_template \
        --dtype $DTYPE \
        --max_num_seqs $MAX_NUM_SEQUENCES \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --tensor_parallel_size 1 \
        --pipeline_parallel_size 1 \
        --data_parallel_size 1 
done
done

# Analysis and Conversion
echo "Starting analysis and conversion..."

MODEL_NAME_WITH_SLASH="${MODELS[0]}"
MODEL_NAME_WITH_UNDERSCORE=$(echo "$MODEL_NAME_WITH_SLASH" | sed 's/\//_/g')

# Run analysis scripts
python $LOCAL_DIR/analyze_results.py \
    --base_dir "$OUTPUT_DIR" \
    --model_name_pattern "$MODEL_NAME_WITH_UNDERSCORE" \
    --tokenizer_name "$MODEL_NAME_WITH_SLASH"

RESULTS_PATH="$OUTPUT_DIR/$MODEL_NAME_WITH_UNDERSCORE/all_experiments_results.json"
ANALYSIS_OUTPUT_PATH="$OUTPUT_DIR/$MODEL_NAME_WITH_UNDERSCORE/analysis_results.json"

python $LOCAL_DIR/analyze_summary.py \
    --results-path "$RESULTS_PATH" \
    --output-path "$ANALYSIS_OUTPUT_PATH"

# Convert parquet to csv
echo "Converting Parquet files to CSV..."
find "$OUTPUT_DIR" -type f -name "*.parquet" | while read -r parquet_file; do
    python $LOCAL_DIR/convert_parquet_to_csv.py "$parquet_file"
done

echo "Analysis and conversion complete."

done
done
done
done
done
done
done
done
