#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage: $0 <model_name1> [model_name2 model_name3 ...]"
    echo "Example:"
    echo "  $0 'Qwen/Qwen2-0.5B-Instruct'"
    echo "  $0 'meta-llama/Llama-2-7b-hf' 'Qwen/Qwen2-1.5B-Instruct'"
    echo ""
}

# Check if any model names are provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Activate conda environment
# source activate grpo

# Process each model name provided as an argument
for MODEL_NAME in "$@"; do
    echo "Downloading model: $MODEL_NAME"

    # Disable DeepSpeed and CUDA operations
    export CUDA_VISIBLE_DEVICES=""
    export DISABLE_DEEPSPEED=1

    # First check if the model exists
    echo "Checking model availability..."
    python -c "from huggingface_hub import model_info; model_info('$MODEL_NAME')" || {
        echo "Error: Model '$MODEL_NAME' not found or not accessible"
        continue
    }

    # Download model files
    echo "Downloading model weights..."
    python -c "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['DISABLE_DEEPSPEED'] = '1'; from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$MODEL_NAME', torch_dtype='auto', device_map='cpu')"

    echo "Downloading tokenizer..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME', torch_dtype='auto')"

    echo "Model download complete for $MODEL_NAME"
    echo "----------------------------------------"
done

echo "All requested models have been processed!"
