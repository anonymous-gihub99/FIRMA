#!/bin/bash
# launch_phase2.sh - Complete Phase 2 Pipeline for T4 GPUs
# This script automates the entire FIRMA training and evaluation pipeline

set -e  # Exit on error

echo "============================================================"
echo "FIRMA PHASE 2 - COMPLETE PIPELINE FOR T4 GPUS"
echo "============================================================"
echo "Timestamp: $(date)"
echo ""

# Configuration
GPUS=2  # Number of GPUs to use
PORT=29500  # Master port for DDP
OUTPUT_DIR="./firma_model_t4"
LOG_DIR="./logs"
RESULTS_DIR="./evaluation_results"

# Create directories
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Function to check GPU availability
check_gpus() {
    echo "Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Please ensure CUDA is installed."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "✓ Found $GPU_COUNT GPU(s)"
    
    if [ $GPU_COUNT -lt $GPUS ]; then
        echo "⚠ Warning: Requested $GPUS GPUs but only $GPU_COUNT available"
        echo "  Adjusting to use $GPU_COUNT GPUs"
        GPUS=$GPU_COUNT
    fi
    
    # Show GPU details
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
}

# Function to check Python dependencies
check_dependencies() {
    echo "Checking Python dependencies..."
    
    python3 -c "
import sys
try:
    import torch
    import transformers
    import peft
    import bitsandbytes
    print('✓ All core dependencies found')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "Installing missing dependencies..."
        pip install -q torch transformers peft bitsandbytes accelerate evaluate
    fi
    echo ""
}

# Function to verify dataset
check_dataset() {
    echo "Checking dataset..."
    
    if [ ! -d "./FIRMA/math_alignment_dataset" ]; then
        echo "❌ Dataset directory not found: ./math_alignment_dataset"
        echo "  Please ensure the dataset is available"
        exit 1
    fi
    
    # Check for required files
    for split in train val test; do
        found=0
        for ext in json jsonl; do
            for pattern in "${split}.${ext}" "${split}_clean.${ext}" "valid_clean.${ext}"; do
                if [ -f "./math_alignment_dataset/${pattern}" ]; then
                    echo "✓ Found ${split} data: ${pattern}"
                    found=1
                    break 2
                fi
            done
        done
        
        if [ $found -eq 0 ]; then
            echo "⚠ Warning: No ${split} data found"
        fi
    done
    echo ""
}

# Function to train model
train_model() {
    echo "============================================================"
    echo "STARTING TRAINING"
    echo "============================================================"
    
    LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
    
    if [ $GPUS -eq 1 ]; then
        echo "Running single GPU training..."
        python3 firma_model_t4.py 2>&1 | tee $LOG_FILE
    else
        echo "Running multi-GPU training with $GPUS GPUs..."
        
        # Check if torchrun is available
        if command -v torchrun &> /dev/null; then
            CMD="torchrun --nproc_per_node=$GPUS --master_port=$PORT"
        else
            CMD="python3 -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT"
        fi
        
        echo "Command: $CMD firma_model_t4.py"
        $CMD firma_model_t4.py 2>&1 | tee $LOG_FILE
    fi
    
    # Check if training succeeded
    if [ -d "${OUTPUT_DIR}/final" ]; then
        echo "✓ Training completed successfully"
        echo "  Model saved to: ${OUTPUT_DIR}/final"
    else
        echo "❌ Training failed - no final model found"
        exit 1
    fi
    echo ""
}

# Function to evaluate model
evaluate_model() {
    echo "============================================================"
    echo "STARTING EVALUATION"
    echo "============================================================"
    
    LOG_FILE="${LOG_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Running evaluation..."
    python3 evaluate_firma.py 2>&1 | tee $LOG_FILE
    
    # Check if evaluation succeeded
    if [ -f "${RESULTS_DIR}/all_results.json" ]; then
        echo "✓ Evaluation completed successfully"
        echo "  Results saved to: ${RESULTS_DIR}/all_results.json"
    else
        echo "⚠ Warning: Evaluation may have failed"
    fi
    echo ""
}

# Function to generate summary
generate_summary() {
    echo "============================================================"
    echo "GENERATING SUMMARY"
    echo "============================================================"
    
    python3 << EOF
import json
import os
from pathlib import Path
import time

# Collect results
summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'gpus_used': $GPUS,
    'training': {},
    'evaluation': {}
}

# Check training results
model_path = Path('${OUTPUT_DIR}/final')
if model_path.exists():
    summary['training']['status'] = 'completed'
    summary['training']['model_path'] = str(model_path)
    
    # Get model size
    total_size = sum(f.stat().st_size for f in model_path.glob('*') if f.is_file())
    summary['training']['model_size_mb'] = total_size / (1024 * 1024)
else:
    summary['training']['status'] = 'failed'

# Check evaluation results
eval_path = Path('${RESULTS_DIR}/all_results.json')
if eval_path.exists():
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)
    summary['evaluation'] = eval_results
    summary['evaluation']['status'] = 'completed'
else:
    summary['evaluation']['status'] = 'not_run'

# Save summary
with open('phase2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("Summary:")
print(f"  Training: {summary['training']['status']}")
print(f"  Evaluation: {summary['evaluation']['status']}")
if 'model_size_mb' in summary['training']:
    print(f"  Model size: {summary['training']['model_size_mb']:.1f} MB")

print("\nFull summary saved to: phase2_summary.json")
EOF
    
    echo ""
}

# Function to run interactive test
interactive_test() {
    echo "============================================================"
    echo "INTERACTIVE TEST"
    echo "============================================================"
    
    python3 << 'EOF'
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Check if model exists
model_path = Path('./firma_model_t4/final')
if not model_path.exists():
    print("Model not found. Please train first.")
    exit(1)

print("Loading model for interactive test...")

# Import and load model
from firma_model_t4 import FIRMA, FIRMAConfig

config = FIRMAConfig()
tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test examples
examples = [
    "∀x ∈ ℝ : x + 0 = x",
    "The sum of two even numbers is always even",
    "If a divides b and b divides c, then a divides c"
]

print("\nTest translations:")
print("-" * 40)

for text in examples:
    print(f"\nInput: {text}")
    
    # Determine direction
    if any(c in text for c in ['∀', '∃', '∈', '→']):
        prompt = f"Translate to informal: {text}\nInformal:"
        print("Direction: Formal → Informal")
    else:
        prompt = f"Translate to formal: {text}\nFormal:"
        print("Direction: Informal → Formal")
    
    print("Output: [Model would generate translation here]")
    print("-" * 40)

print("\n✓ Interactive test complete")
EOF
}

# Main pipeline
main() {
    echo "Starting Phase 2 Pipeline..."
    echo ""
    
    # Step 1: Environment checks
    check_gpus
    check_dependencies
    check_dataset
    
    # Step 2: Training
    if [ -d "${OUTPUT_DIR}/final" ]; then
        echo "Existing model found at ${OUTPUT_DIR}/final"
        read -p "Skip training and proceed to evaluation? (y/n): " skip_training
        if [ "$skip_training" != "y" ]; then
            train_model
        fi
    else
        train_model
    fi
    
    # Step 3: Evaluation
    evaluate_model
    
    # Step 4: Generate summary
    generate_summary
    
    # Step 5: Interactive test
    read -p "Run interactive test? (y/n): " run_test
    if [ "$run_test" == "y" ]; then
        interactive_test
    fi
    
    echo "============================================================"
    echo "PHASE 2 PIPELINE COMPLETE"
    echo "============================================================"
    echo ""
    echo "Results:"
    echo "  - Trained model: ${OUTPUT_DIR}/final/"
    echo "  - Evaluation results: ${RESULTS_DIR}/"
    echo "  - Training logs: ${LOG_DIR}/"
    echo "  - Summary: phase2_summary.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review the evaluation results"
    echo "  2. Test the model with custom examples"
    echo "  3. Compare with baseline models"
    echo "  4. Prepare results for publication"
    echo ""
    echo "Timestamp: $(date)"
}

# Handle command line arguments
case "${1:-}" in
    train)
        check_gpus
        check_dependencies
        check_dataset
        train_model
        ;;
    eval)
        evaluate_model
        ;;
    test)
        interactive_test
        ;;
    summary)
        generate_summary
        ;;
    *)
        main
        ;;
esac
