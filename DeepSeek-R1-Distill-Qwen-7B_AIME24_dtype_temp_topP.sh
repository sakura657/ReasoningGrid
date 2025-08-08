#!/bin/bash
trap "echo 'Killing all child processes...'; kill 0" SIGINT SIGTERM EXIT

# ==== Docker-Compatible Configuration ====
LOCAL_DIR="/app"
OUTPUT_DIR="/app/output"

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


MODEL_FOLDER_NAME=$(echo "$MODEL" | tr '/' '_')

# Pay Attention to the hardcoded output directory structure (dataset)
RUN_NAME="${SEED}-${TEMP}-${TOP_P}-${DTYPE}-aime24-${MAX_TOKENS}"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_FOLDER_NAME/$RUN_NAME"

# Skip if this run already exists (based on folder existence)
if [ -d "$RUN_OUTPUT_DIR" ]; then
  echo "Skipping existing run: $RUN_NAME"
  continue
fi

mkdir -p "$RUN_OUTPUT_DIR"

LOG_FILE="$RUN_OUTPUT_DIR/run.log"

echo "Running job for parameters: seed=${SEED}, temp=${TEMP}, top_p=${TOP_P}, dtype=${DTYPE}"


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
