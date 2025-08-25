#!/usr/bin/env python3
"""
train_firma_updated.py - Training and evaluation script for FIRMA
Handles proper JSON parsing and comparison with baselines
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import gc
import evaluate
from transformers import AutoTokenizer

# Import FIRMA components
from firma_model_updated import (
    FIRMA, FIRMAConfig, FIRMADataset, 
    FIRMATrainer, create_and_train_firma
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FIRMAEvaluator:
    """Comprehensive evaluation of FIRMA against baselines"""
    
    def __init__(self, firma_model, tokenizer, config: FIRMAConfig):
        self.firma_model = firma_model
        self.tokenizer = tokenizer
        self.config = config
        self.results_dir = Path(config.output_dir) / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation metrics
        try:
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            self.bertscore = evaluate.load("bertscore")
        except:
            logger.warning("Could not load all evaluation metrics")
            self.bleu = None
            self.rouge = None
            self.bertscore = None
    
    def evaluate_firma(self, test_data_path: str) -> Dict:
        """Evaluate FIRMA model performance"""
        logger.info("Evaluating FIRMA model...")
        
        results = {
            'model': 'FIRMA',
            'formal_to_informal': {},
            'informal_to_formal': {}
        }
        
        # Load test data
        test_data = []
        
        # Try different file formats
        if Path(test_data_path).exists():
            if test_data_path.endswith('.jsonl'):
                with open(test_data_path, 'r') as f:
                    for line in f:
                        try:
                            test_data.append(json.loads(line))
                        except:
                            continue
            elif test_data_path.endswith('.json'):
                with open(test_data_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            test_data = data
                    except:
                        pass
        
        # Limit evaluation size for speed
        test_data = test_data[:100] if test_data else []
        
        if not test_data:
            logger.warning("No test data found!")
            return results
        
        # Evaluate by direction
        for direction_name, direction_id in [('formal_to_informal', 0), ('informal_to_formal', 1)]:
            predictions = []
            references = []
            
            for item in tqdm(test_data[:30], desc=f"FIRMA - {direction_name}"):
                # Handle different key names
                formal_val = None
                informal_val = None
                
                # Check for actual keys in your dataset
                if 'informal_stmt' in item and 'formal_statement' in item:
                    formal_val = item['formal_statement']
                    informal_val = item['informal_stmt']
                elif 'formal_statement' in item and 'informal_statement' in item:
                    formal_val = item['formal_statement']
                    informal_val = item['informal_statement']
                elif 'formal' in item and 'informal' in item:
                    formal_val = item['formal']
                    informal_val = item['informal']
                
                if not formal_val or not informal_val:
                    continue
                    
                if direction_name == 'formal_to_informal':
                    source = formal_val
                    target = informal_val
                    prompt = f"Translate formal to informal:\n{source}\nInformal:"
                else:
                    source = informal_val
                    target = formal_val
                    prompt = f"Translate informal to formal:\n{source}\nFormal:"
                
                # Generate translation
                try:
                    pred = self.generate_translation(prompt)
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    pred = source  # Fallback to source
                
                predictions.append(pred)
                references.append(target)
            
            # Calculate metrics
            if predictions and references:
                try:
                    if self.bleu:
                        bleu_score = self.bleu.compute(
                            predictions=predictions, 
                            references=[[r] for r in references]
                        )
                        results[direction_name]['bleu'] = bleu_score.get('bleu', 0.0)
                    
                    if self.rouge:
                        rouge_score = self.rouge.compute(
                            predictions=predictions, 
                            references=references
                        )
                        results[direction_name]['rouge1'] = rouge_score.get('rouge1', 0.0)
                        results[direction_name]['rouge2'] = rouge_score.get('rouge2', 0.0)
                        results[direction_name]['rougeL'] = rouge_score.get('rougeL', 0.0)
                    
                    if self.bertscore:
                        # BERTScore (sample)
                        sample_size = min(20, len(predictions))
                        bert_score = self.bertscore.compute(
                            predictions=predictions[:sample_size],
                            references=references[:sample_size],
                            lang="en"
                        )
                        results[direction_name]['bertscore_f1'] = np.mean(bert_score['f1'])
                    
                    results[direction_name]['num_samples'] = len(predictions)
                    
                except Exception as e:
                    logger.error(f"Error computing metrics: {e}")
                    results[direction_name] = {
                        'bleu': 0.0,
                        'rouge1': 0.0,
                        'rouge2': 0.0,
                        'rougeL': 0.0,
                        'bertscore_f1': 0.0,
                        'num_samples': 0
                    }
        
        return results
    
    def generate_translation(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate translation using FIRMA"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        self.firma_model.eval()
        with torch.no_grad():
            generated = self.firma_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        translation = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract the translation part
        if "Informal:" in translation:
            translation = translation.split("Informal:")[-1].strip()
        elif "Formal:" in translation:
            translation = translation.split("Formal:")[-1].strip()
        
        return translation
    
    def compare_with_baselines(self, baseline_results_path: str, test_path: str) -> Dict:
        """Compare FIRMA results with baseline results"""
        logger.info("Comparing FIRMA with baselines...")
        
        # Get FIRMA results
        firma_results = self.evaluate_firma(test_path)
        
        # Load baseline results
        baseline_results = {}
        if Path(baseline_results_path).exists():
            try:
                with open(baseline_results_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        baseline_results = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse baseline results: {e}")
                # Try to load individual result files
                results_dir = Path(baseline_results_path).parent
                for model_name in ['direct_finetuning', 'retrieval_augmented', 'api_baseline']:
                    result_file = results_dir / f"{model_name}_results.json"
                    if result_file.exists():
                        try:
                            with open(result_file, 'r') as f:
                                baseline_results[model_name] = json.load(f)
                        except:
                            pass
        
        # Create comparison
        comparison = {
            'models': {},
            'improvements': {},
            'firma_results': firma_results
        }
        
        # Add baseline results
        for model_name, model_results in baseline_results.items():
            if isinstance(model_results, dict) and 'formal_to_informal' in model_results:
                comparison['models'][model_name] = {
                    'f2i_bleu': model_results['formal_to_informal'].get('bleu', 0),
                    'f2i_rouge': model_results['formal_to_informal'].get('rougeL', 0),
                    'f2i_bert': model_results['formal_to_informal'].get('bertscore_f1', 0),
                    'i2f_bleu': model_results['informal_to_formal'].get('bleu', 0),
                    'i2f_rouge': model_results['informal_to_formal'].get('rougeL', 0),
                    'i2f_bert': model_results['informal_to_formal'].get('bertscore_f1', 0),
                }
        
        # Add FIRMA results
        comparison['models']['FIRMA'] = {
            'f2i_bleu': firma_results['formal_to_informal'].get('bleu', 0),
            'f2i_rouge': firma_results['formal_to_informal'].get('rougeL', 0),
            'f2i_bert': firma_results['formal_to_informal'].get('bertscore_f1', 0),
            'i2f_bleu': firma_results['informal_to_formal'].get('bleu', 0),
            'i2f_rouge': firma_results['informal_to_formal'].get('rougeL', 0),
            'i2f_bert': firma_results['informal_to_formal'].get('bertscore_f1', 0),
        }
        
        # Calculate improvements
        if baseline_results:
            best_baseline_f2i = max(
                [m['f2i_bleu'] for m in comparison['models'].values() 
                 if m != comparison['models'].get('FIRMA', {})],
                default=0
            )
            best_baseline_i2f = max(
                [m['i2f_bleu'] for m in comparison['models'].values() 
                 if m != comparison['models'].get('FIRMA', {})],
                default=0
            )
            
            firma_f2i = comparison['models']['FIRMA']['f2i_bleu']
            firma_i2f = comparison['models']['FIRMA']['i2f_bleu']
            
            comparison['improvements'] = {
                'f2i_improvement': ((firma_f2i - best_baseline_f2i) / max(best_baseline_f2i, 0.001)) * 100,
                'i2f_improvement': ((firma_i2f - best_baseline_i2f) / max(best_baseline_i2f, 0.001)) * 100,
            }
        
        # Save comparison
        with open(self.results_dir / "firma_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save detailed FIRMA results
        with open(self.results_dir / "firma_detailed.json", 'w') as f:
            json.dump(firma_results, f, indent=2)
        
        return comparison
    
    def print_comparison_table(self, comparison: Dict):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("FIRMA vs BASELINES COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Model':<20} {'F→I BLEU':<12} {'F→I ROUGE-L':<12} {'I→F BLEU':<12} {'I→F ROUGE-L':<12}")
        print("-"*80)
        
        # Results
        for model_name, metrics in comparison['models'].items():
            print(f"{model_name:<20} {metrics['f2i_bleu']:<12.4f} {metrics['f2i_rouge']:<12.4f} "
                  f"{metrics['i2f_bleu']:<12.4f} {metrics['i2f_rouge']:<12.4f}")
        
        if 'improvements' in comparison:
            print("\n" + "="*80)
            print("FIRMA IMPROVEMENTS")
            print("="*80)
            improvements = comparison['improvements']
            print(f"F→I Improvement: {improvements.get('f2i_improvement', 0):.1f}%")
            print(f"I→F Improvement: {improvements.get('i2f_improvement', 0):.1f}%")

def main():
    """Main execution for Phase 2"""
    print("="*60)
    print("PHASE 2: FIRMA MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Adjust config for available GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 16:  # T4 GPU
            print("Adjusting for T4 GPU...")
            FIRMAConfig.batch_size = 1
            FIRMAConfig.gradient_accumulation = 32
            FIRMAConfig.max_length = 256
            FIRMAConfig.base_model = "Qwen/Qwen2.5-1.5B"  # Smaller model
    
    # Initialize configuration
    config = FIRMAConfig()
    
    # Step 1: Train or load FIRMA
    print("\n" + "-"*60)
    print("Step 1: FIRMA Model")
    print("-"*60)
    
    final_model_path = Path(config.output_dir) / "final"
    
    if final_model_path.exists():
        print("Loading existing FIRMA model...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = FIRMA(config)
        try:
            model.load_pretrained(str(final_model_path))
        except:
            logger.warning("Could not load saved weights, using initialized model")
    else:
        print("Training new FIRMA model...")
        model, tokenizer = create_and_train_firma()
    
    print("✓ FIRMA model ready!")
    
    # Step 2: Evaluate FIRMA
    print("\n" + "-"*60)
    print("Step 2: Evaluating FIRMA")
    print("-"*60)
    
    evaluator = FIRMAEvaluator(model, tokenizer, config)
    
    # Find test data
    test_paths = [
        Path(config.data_dir) / "test.jsonl",
        Path(config.data_dir) / "test.json",
        Path(config.data_dir) / "test_clean.json",
    ]
    
    test_path = None
    for path in test_paths:
        if path.exists():
            test_path = str(path)
            break
    
    if not test_path:
        logger.error("No test data found!")
        return
    
    # Find baseline results
    baseline_paths = [
        "./baseline_models/evaluation_results/all_results.json",
        "./baseline_models/evaluation_results/direct_finetuning_ddp_results.json",
    ]
    
    baseline_path = None
    for path in baseline_paths:
        if Path(path).exists():
            baseline_path = path
            break
    
    if baseline_path:
        print(f"Comparing with baselines from {baseline_path}...")
        comparison = evaluator.compare_with_baselines(baseline_path, test_path)
        evaluator.print_comparison_table(comparison)
    else:
        print("No baseline results found, evaluating FIRMA standalone...")
        firma_results = evaluator.evaluate_firma(test_path)
        print("\nFIRMA Results:")
        print(json.dumps(firma_results, indent=2))
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()