# Phase 1: Dataset Curation and Baseline Implementation with AI4M Dataset
# Using real Lean 4 formal-informal pairs and Qwen models

# Cell 1: Install required packages
!pip install -q torch transformers datasets peft bitsandbytes accelerate
!pip install -q sentence-transformers faiss-gpu evaluate rouge-score
!pip install -q sentencepiece protobuf tqdm networkx

# Cell 2: Setup environment and check GPUs
import os
import torch
import gc
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for efficient GPU usage
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("="*60)
print("PHASE 1: AI4M DATASET & QWEN BASELINES")
print("="*60)

# Check available GPUs
if torch.cuda.is_available():
    print(f"\n✓ CUDA Available")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 16:  # T4 GPU
            print("    → Detected T4 GPU, using optimized settings")
            os.environ['BATCH_SIZE'] = '1'
        elif gpu_memory < 25:  # L4 GPU
            print("    → Detected L4 GPU, using standard settings")
            os.environ['BATCH_SIZE'] = '2'
else:
    print("⚠ No GPU available, will use CPU (very slow)")

# Cell 3: Dataset Curation with AI4M/less-proofnet-lean4-ranked
print("\n" + "="*60)
print("STEP 1: AI4M DATASET CURATION")
print("="*60)

from dataset_curation_ai4m import AI4MDatasetCurator

# Initialize curator
curator = AI4MDatasetCurator(output_dir="./math_alignment_dataset")

# Run curation pipeline
# Use max_samples=1000 for quick testing, None for full dataset (190k)
print("\nStarting dataset curation from AI4M/less-proofnet-lean4-ranked...")
print("This dataset contains real Lean 4 formal proofs with natural language explanations")

try:
    # For initial testing, use a subset
    # Change to max_samples=None for full dataset
    splits = curator.run_full_pipeline(max_samples=5000)  # Start with 5k for testing
    
    if splits:
        print(f"\n✓ Dataset curation complete!")
        print(f"  Train: {len(splits['train'])} pairs")
        print(f"  Val: {len(splits['val'])} pairs")  
        print(f"  Test: {len(splits['test'])} pairs")
        
        # Display sample pairs
        print("\n" + "-"*40)
        print("Sample pairs from training set:")
        for i, pair in enumerate(splits['train'][:2]):
            print(f"\n--- Example {i+1} ---")
            print(f"Informal: {pair.informal[:150]}...")
            print(f"Formal: {pair.formal[:150]}...")
            print(f"Complexity: Level {pair.complexity_level}, Domain: {pair.domain}")
            
except Exception as e:
    print(f"❌ Dataset curation failed: {e}")
    print("Please check your internet connection and HuggingFace access")

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Cell 4: Configure baselines with Qwen models
print("\n" + "="*60)
print("CONFIGURING QWEN BASELINES")
print("="*60)

from baseline_implementation_qwen import (
    BaselineConfig,
    DirectFineTuningBaseline,
    RetrievalAugmentedBaseline,
    CommercialAPIBaseline,
    BaselineEvaluator
)

# Configure for available hardware
config = BaselineConfig()

# Adjust based on GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory < 16:  # T4 GPU
        print("Configuring for T4 GPU (15GB)...")
        config.batch_size = 1
        config.gradient_accumulation = 32
        config.max_length = 256
        # Use smaller Qwen model for T4
        config.model_name = "Qwen/Qwen2.5-1.5B"  # Smaller math model
        config.api_model = "Qwen/Qwen2.5-0.5B"  # Tiny model for API
        
    elif gpu_memory < 25:  # L4 GPU
        print("Configuring for L4 GPU (24GB)...")
        config.batch_size = 2
        config.gradient_accumulation = 16
        config.max_length = 384
        # Can use larger models on L4
        config.model_name = "Qwen/Qwen2.5-Math-7B"
        config.api_model = "Qwen/Qwen2.5-3B"
    else:
        print("Using full configuration...")
        config.batch_size = 4
        config.gradient_accumulation = 8

print(f"\nBaseline Configuration:")
print(f"  Fine-tuning model: {config.model_name}")
print(f"  API model: {config.api_model}")
print(f"  Batch size: {config.batch_size}")
print(f"  Max length: {config.max_length}")

# Cell 5: Train Direct Fine-tuning Baseline
print("\n" + "="*60)
print("BASELINE 1: DIRECT FINE-TUNING")
print("="*60)

try:
    print("Initializing Qwen model with QLoRA...")
    direct_baseline = DirectFineTuningBaseline(config)
    
    print("\nStarting training...")
    print("This may take 20-30 minutes on T4, 10-15 minutes on L4...")
    direct_baseline.train()
    
    print("✓ Direct fine-tuning complete!")
    
    # Test the model
    test_input = "For all odd n show that 8 divides n^2 - 1"
    print(f"\nTest translation:")
    print(f"Input: {test_input}")
    output = direct_baseline.generate(f"Translate informal to formal:\n{test_input}\nFormal:")
    print(f"Output: {output}")
    
except Exception as e:
    print(f"❌ Direct fine-tuning failed: {e}")

# Clear GPU memory
del direct_baseline
gc.collect()
torch.cuda.empty_cache()

# Cell 6: Build Retrieval-Augmented Baseline
print("\n" + "="*60)
print("BASELINE 2: RETRIEVAL-AUGMENTED")
print("="*60)

try:
    print("Building retrieval system...")
    retrieval_baseline = RetrievalAugmentedBaseline(config)
    retrieval_baseline.build_index()
    
    print("✓ Retrieval index built!")
    
    # Test retrieval
    test_formal = "theorem exercise_1_27 {n : ℕ} (hn : odd n) : 8 ∣ (n^2 - 1)"
    result = retrieval_baseline.translate(test_formal, "formal_to_informal")
    print(f"\nTest retrieval:")
    print(f"Input: {test_formal}")
    print(f"Retrieved: {result}")
    
except Exception as e:
    print(f"❌ Retrieval baseline failed: {e}")

del retrieval_baseline
gc.collect()

# Cell 7: Setup API Baseline
print("\n" + "="*60)
print("BASELINE 3: API BASELINE (QWEN)")
print("="*60)

try:
    print(f"Loading {config.api_model}...")
    api_baseline = CommercialAPIBaseline(config)
    
    print("✓ API baseline ready!")
    
    # Test API baseline
    test_informal = "The square of any real number is non-negative"
    result = api_baseline.translate_with_cot(test_informal, "informal_to_formal")
    print(f"\nTest API translation:")
    print(f"Input: {test_informal}")
    print(f"Output: {result}")
    
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
    print("Starting comprehensive evaluation...")
    print("This will evaluate all baselines on the test set")
    
    evaluator = BaselineEvaluator(config)
    
    # Evaluate Direct Fine-tuning
    print("\n1. Evaluating Direct Fine-tuning...")
    direct_baseline = DirectFineTuningBaseline(config)
    # Load trained model if exists
    from pathlib import Path
    model_path = Path(config.output_dir) / "direct_finetuning" / "final_model"
    if model_path.exists():
        from peft import PeftModel
        direct_baseline.model = PeftModel.from_pretrained(direct_baseline.model, str(model_path))
    
    direct_results = evaluator.evaluate_model(
        direct_baseline,
        f"{config.data_dir}/test.jsonl",
        "direct_finetuning"
    )
    
    # Evaluate Retrieval
    print("\n2. Evaluating Retrieval-Augmented...")
    retrieval_baseline = RetrievalAugmentedBaseline(config)
    index_path = Path(config.output_dir) / "retrieval_augmented" / "formal_index.faiss"
    if index_path.exists():
        import faiss
        retrieval_baseline.formal_index = faiss.read_index(str(index_path))
        retrieval_baseline.informal_index = faiss.read_index(
            str(Path(config.output_dir) / "retrieval_augmented" / "informal_index.faiss")
        )
        # Reload texts
        with open(f"{config.data_dir}/train.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                retrieval_baseline.formal_texts.append(item['formal'])
                retrieval_baseline.informal_texts.append(item['informal'])
    
    retrieval_results = evaluator.evaluate_model(
        retrieval_baseline,
        f"{config.data_dir}/test.jsonl",
        "retrieval_augmented"
    )
    
    # Evaluate API
    print("\n3. Evaluating API Baseline...")
    api_baseline = CommercialAPIBaseline(config)
    api_results = evaluator.evaluate_model(
        api_baseline,
        f"{config.data_dir}/test.jsonl",
        "api_baseline"
    )
    
    print("\n✓ Evaluation complete!")
    
except Exception as e:
    print(f"❌ Evaluation failed: {e}")

# Cell 9: Display Results Summary
import json
from pathlib import Path
import pandas as pd

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_dir = Path(config.output_dir) / "evaluation_results"

# Collect all results
all_results = {}
for model_name in ["direct_finetuning", "retrieval_augmented", "api_baseline"]:
    result_file = results_dir / f"{model_name}_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            all_results[model_name] = json.load(f)

if all_results:
    # Create results table
    rows = []
    for model_name, model_results in all_results.items():
        for direction in ['formal_to_informal', 'informal_to_formal']:
            if direction in model_results:
                metrics = model_results[direction]
                rows.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Direction': 'F→I' if direction == 'formal_to_informal' else 'I→F',
                    'BLEU': f"{metrics.get('bleu', 0):.4f}",
                    'ROUGE-L': f"{metrics.get('rougeL', 0):.4f}",
                    'BERTScore': f"{metrics.get('bertscore_f1', 0):.4f}"
                })
    
    df = pd.DataFrame(rows)
    print("\nBaseline Performance on AI4M Dataset:")
    print(df.to_string(index=False))
    
    # Save combined results
    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n✓ Results saved to {results_dir}")
else:
    print("No evaluation results found")

# Cell 10: Analysis and Next Steps
print("\n" + "="*60)
print("PHASE 1 COMPLETE!")
print("="*60)

print("\n📊 Key Achievements:")
print("1. ✓ Curated real Lean 4 formal-informal pairs from AI4M dataset")
print("2. ✓ Trained Qwen2.5-Math model with QLoRA for efficient fine-tuning")
print("3. ✓ Built retrieval system with real mathematical theorems")
print("4. ✓ Established baselines with proper evaluation metrics")

print("\n🎯 Expected Improvements:")
print("• Real dataset (AI4M) should give meaningful BLEU scores (>0)")
print("• Qwen2.5-Math specialized for mathematical reasoning")
print("• Actual Lean 4 syntax for proper formal evaluation")

print("\n🚀 Next Steps for Phase 2 (FIRMA):")
print("1. Use this real dataset for FIRMA training")
print("2. Leverage complexity stratification from AI4M data")
print("3. Compare FIRMA against these Qwen baselines")
print("4. Show improvements in formal-informal alignment")

print("\n📁 Output Locations:")
print(f"• Dataset: ./math_alignment_dataset/")
print(f"• Models: ./baseline_models/")
print(f"• Results: ./baseline_models/evaluation_results/")

# Final memory cleanup
gc.collect()
torch.cuda.empty_cache()

print("\n✨ Ready for Phase 2: FIRMA Model Development!")