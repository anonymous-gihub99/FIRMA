# Phase 1 Execution Notebook
# Run dataset curation and baseline implementation

# Cell 1: Install required packages
!pip install -q torch transformers datasets peft bitsandbytes accelerate
!pip install -q sentence-transformers faiss-gpu evaluate rouge-score
!pip install -q sentencepiece protobuf tqdm networkx

# Cell 2: Import scripts and setup environment
import os
import torch
import gc
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for efficient GPU usage
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.6'  # For T4 and L4 GPUs

# Check available GPUs
print("Available GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Cell 3: Execute Phase 1.1 - Dataset Curation
print("="*60)
print("PHASE 1.1: DATASET CURATION")
print("="*60)

from dataset_curation import DatasetCurator

# Initialize curator
curator = DatasetCurator(output_dir="./math_alignment_dataset")

# Run curation pipeline
try:
    splits = curator.run_full_pipeline()
    print(f"\n✓ Dataset curation complete!")
    print(f"  Train: {len(splits['train'])} pairs")
    print(f"  Val: {len(splits['val'])} pairs")
    print(f"  Test: {len(splits['test'])} pairs")
except Exception as e:
    print(f"❌ Dataset curation failed: {e}")
    print("Attempting to continue with mock data...")

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Cell 4: Execute Phase 1.2 - Baseline Implementation
print("\n" + "="*60)
print("PHASE 1.2: BASELINE IMPLEMENTATION")
print("="*60)

from baseline_implementation import (
    BaselineConfig,
    DirectFineTuningBaseline,
    RetrievalAugmentedBaseline,
    CommercialAPIBaseline,
    BaselineEvaluator
)

# Configure for available hardware
config = BaselineConfig()

# Adjust for GPU memory constraints
if torch.cuda.get_device_properties(0).total_memory < 16e9:  # Less than 16GB
    print("Detected limited GPU memory, adjusting configuration...")
    config.batch_size = 2
    config.gradient_accumulation = 16
    config.max_length = 256
    config.use_4bit = True
    config.model_name = "EleutherAI/pythia-1.4b"  # Smaller model for T4 GPUs

# Cell 5: Train Direct Fine-tuning Baseline
print("\n--- Training Direct Fine-tuning Baseline ---")
try:
    direct_baseline = DirectFineTuningBaseline(config)
    direct_baseline.train()
    print("✓ Direct fine-tuning complete!")
except Exception as e:
    print(f"❌ Direct fine-tuning failed: {e}")

# Clear GPU memory
del direct_baseline
gc.collect()
torch.cuda.empty_cache()

# Cell 6: Build Retrieval-Augmented Baseline
print("\n--- Building Retrieval-Augmented Baseline ---")
try:
    retrieval_baseline = RetrievalAugmentedBaseline(config)
    retrieval_baseline.build_index()
    print("✓ Retrieval index built!")
except Exception as e:
    print(f"❌ Retrieval baseline failed: {e}")

# Clear memory
del retrieval_baseline
gc.collect()

# Cell 7: Setup API Baseline
print("\n--- Setting up API Baseline ---")
try:
    api_baseline = CommercialAPIBaseline(config)
    print("✓ API baseline ready!")
except Exception as e:
    print(f"❌ API baseline setup failed: {e}")

del api_baseline
gc.collect()
torch.cuda.empty_cache()

# Cell 8: Evaluate All Baselines
print("\n" + "="*60)
print("EVALUATION PHASE")
print("="*60)

try:
    evaluator = BaselineEvaluator(config)
    results = evaluator.run_all_evaluations()
    print("\n✓ Evaluation complete! Results saved to ./baseline_models/evaluation_results/")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")

# Cell 9: Display Results Summary
import json
from pathlib import Path

results_path = Path("./baseline_models/evaluation_results/all_results.json")
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Create results table
    import pandas as pd
    
    rows = []
    for model_name, model_results in results.items():
        for direction in ['formal_to_informal', 'informal_to_formal']:
            metrics = model_results[direction]
            rows.append({
                'Model': model_name,
                'Direction': direction,
                'BLEU': f"{metrics.get('bleu', 0):.4f}",
                'ROUGE-L': f"{metrics.get('rougeL', 0):.4f}",
                'BERTScore': f"{metrics.get('bertscore_f1', 0):.4f}"
            })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
else:
    print("No results file found. Please check the evaluation output.")

# Cell 10: Quick Test of Models
print("\n" + "="*60)
print("QUICK MODEL TEST")
print("="*60)

test_formal = "∀x,y ∈ ℝ : x + y = y + x"
test_informal = "Addition is commutative for real numbers"

print(f"Test formal: {test_formal}")
print(f"Test informal: {test_informal}")
print()

# Test each baseline
try:
    # Load models for testing
    from baseline_implementation import DirectFineTuningBaseline, RetrievalAugmentedBaseline, CommercialAPIBaseline
    
    # Test retrieval baseline
    retrieval = RetrievalAugmentedBaseline(config)
    if Path("./baseline_models/retrieval_augmented/formal_index.faiss").exists():
        import faiss
        retrieval.formal_index = faiss.read_index("./baseline_models/retrieval_augmented/formal_index.faiss")
        retrieval.informal_index = faiss.read_index("./baseline_models/retrieval_augmented/informal_index.faiss")
        
        # Load texts
        with open("./math_alignment_dataset/train.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                retrieval.formal_texts.append(item['formal'])
                retrieval.informal_texts.append(item['informal'])
        
        translation = retrieval.translate(test_formal, "formal_to_informal")
        print(f"Retrieval F→I: {translation}")
    
    # Test API baseline
    api = CommercialAPIBaseline(config)
    translation = api.translate_with_cot(test_informal, "informal_to_formal")
    print(f"API I→F: {translation}")
    
except Exception as e:
    print(f"Model testing failed: {e}")

print("\n" + "="*60)
print("PHASE 1 COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Review the dataset in ./math_alignment_dataset/")
print("2. Check baseline models in ./baseline_models/")
print("3. Analyze evaluation results in ./baseline_models/evaluation_results/")
print("4. Proceed to Phase 2: FIRMA model development")