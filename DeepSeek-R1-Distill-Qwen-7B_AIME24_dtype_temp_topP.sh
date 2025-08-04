#!/bin/bash

# ==== Docker-Compatible Configuration ====
LOCAL_DIR="/app"
OUTPUT_DIR="/app/output"
mkdir -p "$OUTPUT_DIR/logs"

# ==== Experiment Grid ====
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TASK="custom|aime24|0|0"

# Decoding parameters
TOP_PS=(0.8 0.9 0.95 0.98 1.0)
TEMPS=(0.0 0.2 0.4 0.6 0.8 1.0)

# Random seeds for averaging
SEEDS=(0 1 2 3 4 42 100 123 2023 777)

# Numerical precisions
DTYPES=("bfloat16" "float16" "float32")

MAX_MODEL_LENGTH=32768
MAX_TOKENS=32768

# ==== Grid Execution ====
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do

JOB_NAME="eval-aime24-7B-seed${SEED}-temp${TEMP}-topp${TOP_P}-${DTYPE}"
LOG_FILE="$OUTPUT_DIR/logs/${JOB_NAME}.log"

echo "Running job: $JOB_NAME"

# 实际执行任务
python main.py \
  --model "$MODEL" \
  --task "$TASK" \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_model_length "$MAX_MODEL_LENGTH" \
  --custom_tasks_directory lighteval_tasks.py \
  --use_chat_template \
  --dtype "$DTYPE" \
  &> "$LOG_FILE"

done
done
done
done
