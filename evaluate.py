#!/usr/bin/env python3
"""
evaluate_firma_real.py - Actual evaluation script for FIRMA model
Multi-GPU support for 4x T4 GPUs (60GB total memory)
Fixed data loading and model name
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FIRMAEvaluator:
    """Real evaluation for FIRMA model with multi-GPU support"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Check available GPUs
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Found {self.num_gpus} GPUs")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        else:
            self.num_gpus = 0
            logger.warning("No GPUs found, using CPU")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Load base model and adapter with multi-GPU support
        logger.info("Loading FIRMA model...")
        
        # UPDATED: Use the correct base model
        base_model_name = "Qwen/Qwen3-4B"  # Updated from config
        
        try:
            # Try loading from config file if exists
            config_path = self.model_path.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    if 'base_model_name_or_path' in config_data:
                        base_model_name = config_data['base_model_name_or_path']
                        logger.info(f"Using base model from config: {base_model_name}")
        except:
            pass
        
        # Load base model with automatic device mapping across GPUs
        if self.num_gpus > 0:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatically distribute across all GPUs
                trust_remote_code=True,
                max_memory={i: "14GB" for i in range(self.num_gpus)}  # Limit per GPU
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        # Load LoRA adapter if exists
        try:
            logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path),
                torch_dtype=torch.float16 if self.num_gpus > 0 else torch.float32
            )
            logger.info("LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter: {e}")
            logger.info("Using base model without adapter")
        
        self.model.eval()
        
        if self.num_gpus > 0:
            # Log memory usage
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        logger.info("Model loaded successfully")
    
    def load_test_data(self, data_dir: str = "./math_alignment_dataset", limit: int = 00) -> List[Dict]:
        """Load actual test data with better debugging"""
        test_samples = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_path}")
            return []
        
        # Try different test file names
        test_files = ["test_clean.json", "test.json", "test.jsonl", 
                     "valid_clean.json", "val.json", "validation.json"]
        
        loaded_file = None
        raw_data = []
        
        for filename in test_files:
            filepath = data_path / filename
            if filepath.exists():
                logger.info(f"Found file: {filename}")
                logger.info(f"File size: {filepath.stat().st_size / 1024:.1f} KB")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if filename.endswith('.jsonl'):
                            for line_num, line in enumerate(f, 1):
                                if line.strip():
                                    try:
                                        item = json.loads(line)
                                        raw_data.append(item)
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Error parsing line {line_num}: {e}")
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                raw_data = data
                            else:
                                raw_data = [data]
                    
                    loaded_file = filename
                    logger.info(f"Successfully loaded {len(raw_data)} items from {filename}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue
        
        if not raw_data:
            logger.error("No data could be loaded from any test file")
            logger.info(f"Files checked: {test_files}")
            logger.info(f"Directory contents: {list(data_path.glob('*.json*'))}")
            return []
        
        # Debug: Check keys in first item
        if raw_data:
            logger.info(f"Sample item keys: {list(raw_data[0].keys())}")
            logger.info(f"First item preview: {str(raw_data[0])[:200]}...")
        
        # Process samples - try different key combinations
        processed = []
        
        # Possible key combinations
        informal_keys = ["informal_stmt", "informal_statement", "informal", "informal_text"]
        formal_keys = ["formal_statement", "formal_stmt", "formal", "formal_text"]
        
        # Find which keys exist
        informal_key = None
        formal_key = None
        
        if raw_data:
            sample_item = raw_data[0]
            for key in informal_keys:
                if key in sample_item:
                    informal_key = key
                    break
            for key in formal_keys:
                if key in sample_item:
                    formal_key = key
                    break
        
        logger.info(f"Using keys: informal='{informal_key}', formal='{formal_key}'")
        
        if not informal_key or not formal_key:
            logger.error(f"Could not find required keys in data")
            logger.info(f"Available keys: {list(raw_data[0].keys()) if raw_data else 'None'}")
            return []
        
        # Process items
        count = 0
        for item in raw_data[:limit//2]:  # Take half the limit since we add both directions
            if informal_key in item and formal_key in item:
                informal_text = str(item[informal_key])
                formal_text = str(item[formal_key])
                
                if informal_text and formal_text:
                    # Add both directions
                    processed.append({
                        'input': formal_text[:200],  # Truncate for safety
                        'expected': informal_text[:200],
                        'direction': 'formal_to_informal'
                    })
                    processed.append({
                        'input': informal_text[:200],
                        'expected': formal_text[:200],
                        'direction': 'informal_to_formal'
                    })
                    count += 1
        
        logger.info(f"Processed {len(processed)} test samples from {count} item pairs")
        
        if not processed:
            logger.error("No samples could be processed")
            logger.info("Check that the data files contain the expected fields")
        
        return processed
    
    def translate(self, text: str, direction: str) -> str:
        """Translate a single text"""
        # Create prompt based on direction
        if direction == 'formal_to_informal':
            prompt = f"Translate this formal mathematical statement to informal language:\n{text}\n\nInformal translation:"
        else:
            prompt = f"Translate this informal statement to formal mathematical notation:\n{text}\n\nFormal translation:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        # Move to GPU if available
        if self.num_gpus > 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation
        if "translation:" in full_output.lower():
            translation = full_output.split("translation:")[-1].strip()
        else:
            # Remove the prompt part
            translation = full_output[len(prompt):].strip()
            if not translation:
                # Fallback: take everything after newline
                parts = full_output.split('\n')
                translation = parts[-1] if len(parts) > 1 else full_output
        
        return translation
    
    def evaluate(self, test_samples: List[Dict]) -> Dict:
        """Evaluate model on test samples"""
        results = {
            'translations': [],
            'by_direction': defaultdict(list),
            'by_complexity': defaultdict(list),
            'metrics': {}
        }
        
        if not test_samples:
            logger.error("No test samples to evaluate")
            return results
        
        logger.info(f"Starting evaluation of {len(test_samples)} samples...")
        
        batch_size = 1  # Process one at a time for stability
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
            try:
                # Translate
                gen_start = time.time()
                translation = self.translate(sample['input'], sample['direction'])
                gen_time = time.time() - gen_start
                
                # Compute complexity (simple heuristic)
                complexity = 1
                if any(c in sample['input'] for c in ['∀', '∃', '∈', '⊂', '⊆']):
                    complexity = 2
                if any(c in sample['input'] for c in ['∧', '∨', '→', '⇒', '⇔']):
                    complexity = 3
                if any(c in sample['input'] for c in ['∫', '∂', '∑', '∏', 'lim']):
                    complexity = 4
                
                result = {
                    'input': sample['input'][:200],
                    'expected': sample['expected'][:200],
                    'output': translation[:200] if translation else '',
                    'direction': sample['direction'],
                    'complexity': complexity,
                    'gen_time': gen_time
                }
                
                results['translations'].append(result)
                results['by_direction'][sample['direction']].append(result)
                results['by_complexity'][f'level_{complexity}'].append(result)
                
            except Exception as e:
                logger.debug(f"Translation failed for sample {i}: {e}")
                continue
            
            # Clear cache periodically
            if i % 50 == 0 and self.num_gpus > 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        results['metrics'] = self._calculate_metrics(results)
        results['metrics']['total_time'] = total_time
        
        logger.info(f"Evaluation complete. Processed {len(results['translations'])} samples in {total_time:.1f}s")
        
        return results
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        all_translations = results['translations']
        if all_translations:
            metrics['total_samples'] = len(all_translations)
            gen_times = [r['gen_time'] for r in all_translations if r['gen_time'] > 0]
            metrics['avg_gen_time'] = np.mean(gen_times) if gen_times else 0
            
            # By direction
            for direction in ['formal_to_informal', 'informal_to_formal']:
                dir_results = results['by_direction'].get(direction, [])
                if dir_results:
                    dir_times = [r['gen_time'] for r in dir_results if r['gen_time'] > 0]
                    metrics[direction] = {
                        'count': len(dir_results),
                        'avg_time': np.mean(dir_times) if dir_times else 0
                    }
            
            # By complexity
            for level in range(1, 5):
                level_results = results['by_complexity'].get(f'level_{level}', [])
                if level_results:
                    level_times = [r['gen_time'] for r in level_results if r['gen_time'] > 0]
                    metrics[f'complexity_{level}'] = {
                        'count': len(level_results),
                        'avg_time': np.mean(level_times) if level_times else 0
                    }
        else:
            metrics['total_samples'] = 0
            metrics['avg_gen_time'] = 0
        
        return metrics
    
    def generate_report(self, results: Dict) -> str:
        """Generate evaluation report"""
        report = []
        report.append("="*60)
        report.append("FIRMA EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Model: {self.model_path}")
        report.append(f"GPUs used: {self.num_gpus}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        metrics = results.get('metrics', {})
        
        # Overall stats
        report.append("OVERALL STATISTICS:")
        report.append("-"*40)
        report.append(f"Total samples evaluated: {metrics.get('total_samples', 0)}")
        report.append(f"Total evaluation time: {metrics.get('total_time', 0):.1f}s")
        report.append(f"Average generation time: {metrics.get('avg_gen_time', 0):.3f}s per sample")
        
        if metrics.get('total_time', 0) > 0:
            throughput = metrics.get('total_samples', 0) / metrics.get('total_time', 1)
            report.append(f"Throughput: {throughput:.1f} samples/sec")
        report.append("")
        
        # Direction analysis
        report.append("BY DIRECTION:")
        report.append("-"*40)
        for direction in ['formal_to_informal', 'informal_to_formal']:
            if direction in metrics:
                stats = metrics[direction]
                report.append(f"{direction}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        report.append("")
        
        # Complexity analysis
        report.append("BY COMPLEXITY:")
        report.append("-"*40)
        for level in range(1, 5):
            key = f'complexity_{level}'
            if key in metrics:
                stats = metrics[key]
                report.append(f"Level {level}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        
        # Sample translations
        report.append("")
        report.append("SAMPLE TRANSLATIONS:")
        report.append("-"*40)
        
        for i, trans in enumerate(results['translations'][:5]):
            report.append(f"\nExample {i+1} ({trans['direction']}):")
            report.append(f"Input: {trans['input'][:100]}...")
            if trans['output']:
                report.append(f"Output: {trans['output'][:100]}...")
            else:
                report.append("Output: [No translation generated]")
            report.append(f"Time: {trans['gen_time']:.3f}s")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    """Main evaluation function"""
    
    # Model path
    model_path = Path("firma_model_t4/final")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize evaluator
    evaluator = FIRMAEvaluator(str(model_path))
    
    # Load test data
    test_samples = evaluator.load_test_data(limit=100)  # Start with 100 for testing
    
    if not test_samples:
        logger.error("No test data found - cannot proceed with evaluation")
        return
    
    # Run evaluation
    results = evaluator.evaluate(test_samples)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    results_path = evaluator.results_dir / "evaluation_results.json"
    
    # Convert for JSON serialization
    save_results = {
        'metrics': results['metrics'],
        'sample_translations': results['translations'][:20],
        'by_direction_counts': {k: len(v) for k, v in results['by_direction'].items()},
        'by_complexity_counts': {k: len(v) for k, v in results['by_complexity'].items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

class FIRMAEvaluator:
    """Real evaluation for FIRMA model with multi-GPU support"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Check available GPUs
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Found {self.num_gpus} GPUs")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model and adapter with multi-GPU support
        logger.info("Loading FIRMA model across multiple GPUs...")
        base_model_name = "Qwen/Qwen3-8B"  # From your config
        
        # Load base model with automatic device mapping across GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatically distribute across all GPUs
            trust_remote_code=True,
            max_memory={i: "14GB" for i in range(self.num_gpus)}  # Limit per GPU
        )
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        
        # Log memory usage
        for i in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        logger.info("Model loaded successfully across GPUs")
    
    def load_test_data(self, data_dir: str = "./math_alignment_dataset", limit: int = 200) -> List[Dict]:
        """Load actual test data - increased limit for multi-GPU"""
        test_samples = []
        data_path = Path(data_dir)
        
        # Try different test file names
        test_files = ["test_clean.json", "test.json", "test.jsonl"]
        
        for filename in test_files:
            filepath = data_path / filename
            if filepath.exists():
                logger.info(f"Loading test data from {filename}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if filename.endswith('.json'):
                            for line in f:
                                item = json.loads(line)
                                test_samples.append(item)
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                test_samples.extend(data)
                    break
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Process samples
        processed = []
        informal_key = "informal_stmt" if "test_clean" in str(filepath) else "informal_statement"
        formal_key = "formal_statement"
        
        for item in test_samples[:limit//2]:  # Take more samples with multi-GPU
            if informal_key in item and formal_key in item:
                # Add both directions
                processed.append({
                    'input': item[formal_key],
                    'expected': item[informal_key],
                    'direction': 'formal_to_informal'
                })
                processed.append({
                    'input': item[informal_key],
                    'expected': item[formal_key],
                    'direction': 'informal_to_formal'
                })
        
        logger.info(f"Loaded {len(processed)} test samples")
        return processed
    
    def translate_batch(self, texts: List[str], directions: List[str], batch_size: int = 8) -> List[str]:
        """Translate a batch of texts - optimized for multi-GPU"""
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_directions = directions[i:i+batch_size]
            
            # Create prompts
            prompts = []
            for text, direction in zip(batch_texts, batch_directions):
                if direction == 'formal_to_informal':
                    prompt = f"Translate this formal mathematical statement to informal language:\n{text}\n\nInformal translation:"
                else:
                    prompt = f"Translate this informal statement to formal mathematical notation:\n{text}\n\nFormal translation:"
                prompts.append(prompt)
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            # Move to first GPU (model handles distribution internally)
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_beams=1  # Faster generation
                )
            
            # Decode batch
            batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract translations
            for prompt, full_output in zip(prompts, batch_outputs):
                if "translation:" in full_output.lower():
                    translation = full_output.split("translation:")[-1].strip()
                else:
                    translation = full_output[len(prompt):].strip()
                translations.append(translation)
        
        return translations
    
    def evaluate(self, test_samples: List[Dict]) -> Dict:
        """Evaluate model on test samples with batch processing"""
        results = {
            'translations': [],
            'by_direction': defaultdict(list),
            'by_complexity': defaultdict(list),
            'metrics': {}
        }
        
        logger.info(f"Evaluating {len(test_samples)} samples with batch processing...")
        
        # Process in larger batches for efficiency
        batch_size = 8  # Adjust based on GPU memory
        total_batches = (len(test_samples) + batch_size - 1) // batch_size
        
        all_texts = [s['input'] for s in test_samples]
        all_directions = [s['direction'] for s in test_samples]
        
        # Batch translation
        start_time = time.time()
        all_translations = []
        
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Evaluating batches"):
            batch_texts = all_texts[i:i+batch_size]
            batch_directions = all_directions[i:i+batch_size]
            
            try:
                batch_translations = self.translate_batch(batch_texts, batch_directions, batch_size=batch_size)
                all_translations.extend(batch_translations)
            except Exception as e:
                logger.warning(f"Batch translation failed: {e}")
                # Fallback to empty translations for failed batch
                all_translations.extend([''] * len(batch_texts))
            
            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                for gpu_id in range(self.num_gpus):
                    torch.cuda.empty_cache()
                gc.collect()
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_samples) if test_samples else 0
        
        # Process results
        for sample, translation in zip(test_samples, all_translations):
            # Compute complexity
            complexity = 1
            if any(c in sample['input'] for c in ['∀', '∃', '∈']):
                complexity = 2
            if any(c in sample['input'] for c in ['∧', '∨', '→']):
                complexity = 3
            if any(c in sample['input'] for c in ['∫', '∂', '∑']):
                complexity = 4
            
            result = {
                'input': sample['input'][:100],
                'expected': sample['expected'][:100],
                'output': translation[:100] if translation else '',
                'direction': sample['direction'],
                'complexity': complexity,
                'gen_time': avg_time  # Average time per sample
            }
            
            results['translations'].append(result)
            results['by_direction'][sample['direction']].append(result)
            results['by_complexity'][f'level_{complexity}'].append(result)
        
        # Calculate metrics
        results['metrics'] = self._calculate_metrics(results)
        results['metrics']['total_time'] = total_time
        results['metrics']['batch_size'] = batch_size
        results['metrics']['num_gpus'] = self.num_gpus
        
        # Log GPU memory usage after evaluation
        logger.info("GPU memory usage after evaluation:")
        for i in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"  GPU {i}: {allocated:.1f}GB allocated")
        
        return results
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        all_translations = results['translations']
        if all_translations:
            metrics['total_samples'] = len(all_translations)
            metrics['avg_gen_time'] = np.mean([r['gen_time'] for r in all_translations])
            
            # By direction
            for direction in ['formal_to_informal', 'informal_to_formal']:
                dir_results = results['by_direction'].get(direction, [])
                if dir_results:
                    metrics[direction] = {
                        'count': len(dir_results),
                        'avg_time': np.mean([r['gen_time'] for r in dir_results])
                    }
            
            # By complexity
            for level in range(1, 5):
                level_results = results['by_complexity'].get(f'level_{level}', [])
                if level_results:
                    metrics[f'complexity_{level}'] = {
                        'count': len(level_results),
                        'avg_time': np.mean([r['gen_time'] for r in level_results])
                    }
        
        return metrics
    
    def generate_report(self, results: Dict) -> str:
        """Generate evaluation report"""
        report = []
        report.append("="*60)
        report.append("FIRMA EVALUATION REPORT (Multi-GPU)")
        report.append("="*60)
        report.append(f"Model: {self.model_path}")
        report.append(f"GPUs used: {results['metrics'].get('num_gpus', 1)}")
        report.append(f"Batch size: {results['metrics'].get('batch_size', 1)}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        metrics = results.get('metrics', {})
        
        # Overall stats
        report.append("OVERALL STATISTICS:")
        report.append("-"*40)
        report.append(f"Total samples evaluated: {metrics.get('total_samples', 0)}")
        report.append(f"Total evaluation time: {metrics.get('total_time', 0):.1f}s")
        report.append(f"Average generation time: {metrics.get('avg_gen_time', 0):.3f}s per sample")
        report.append(f"Throughput: {metrics.get('total_samples', 0) / max(metrics.get('total_time', 1), 1):.1f} samples/sec")
        report.append("")
        
        # Direction analysis
        report.append("BY DIRECTION:")
        report.append("-"*40)
        for direction in ['formal_to_informal', 'informal_to_formal']:
            if direction in metrics:
                stats = metrics[direction]
                report.append(f"{direction}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        report.append("")
        
        # Complexity analysis
        report.append("BY COMPLEXITY:")
        report.append("-"*40)
        for level in range(1, 5):
            key = f'complexity_{level}'
            if key in metrics:
                stats = metrics[key]
                report.append(f"Level {level}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        
        # Sample translations
        report.append("")
        report.append("SAMPLE TRANSLATIONS:")
        report.append("-"*40)
        
        for i, trans in enumerate(results['translations'][:5]):
            report.append(f"\nExample {i+1} ({trans['direction']}):")
            report.append(f"Input: {trans['input'][:100]}...")
            report.append(f"Output: {trans['output'][:100]}...")
            report.append(f"Time: {trans['gen_time']:.3f}s")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    """Main evaluation function"""
    
    # Model path
    model_path = Path("firma_model_t4/final")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize evaluator
    evaluator = FIRMAEvaluator(str(model_path))
    
    # Load test data - can handle more with multi-GPU
    test_samples = evaluator.load_test_data(limit=200)  # Increased from 200
    
    if not test_samples:
        logger.error("No test data found")
        return
    
    # Run evaluation
    results = evaluator.evaluate(test_samples)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    results_path = evaluator.results_dir / "evaluation_results.json"
    
    # Convert for JSON serialization
    save_results = {
        'metrics': results['metrics'],
        'sample_translations': results['translations'][:20],  # Save more samples
        'by_direction_counts': {k: len(v) for k, v in results['by_direction'].items()},
        'by_complexity_counts': {k: len(v) for k, v in results['by_complexity'].items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

class FIRMAEvaluator:
    """Real evaluation for FIRMA model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model and adapter
        logger.info("Loading FIRMA model...")
        base_model_name = "Qwen/Qwen3-8B"  # From your config
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            str(self.model_path),
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def load_test_data(self, data_dir: str = "./math_alignment_dataset") -> List[Dict]:
        """Load actual test data"""
        test_samples = []
        data_path = Path(data_dir)
        
        # Try different test file names
        test_files = ["test_clean.json", "test.json", "test.jsonl"]
        
        for filename in test_files:
            filepath = data_path / filename
            if filepath.exists():
                logger.info(f"Loading test data from {filename}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if filename.endswith('.jsonl'):
                            for line in f:
                                item = json.loads(line)
                                test_samples.append(item)
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                test_samples.extend(data)
                    break
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Process samples
        processed = []
        informal_key = "informal_stmt" if "test_clean" in str(filepath) else "informal_statement"
        formal_key = "formal_statement"
        
        for item in test_samples[:180]:  # Limit to 200 for evaluation
            if informal_key in item and formal_key in item:
                # Add both directions
                processed.append({
                    'input': item[formal_key],
                    'expected': item[informal_key],
                    'direction': 'formal_to_informal'
                })
                processed.append({
                    'input': item[informal_key],
                    'expected': item[formal_key],
                    'direction': 'informal_to_formal'
                })
        
        logger.info(f"Loaded {len(processed)} test samples")
        return processed
    
    def translate(self, text: str, direction: str) -> str:
        """Translate a single text"""
        # Create prompt based on direction
        if direction == 'formal_to_informal':
            prompt = f"Translate this formal mathematical statement to informal language:\n{text}\n\nInformal translation:"
        else:
            prompt = f"Translate this informal statement to formal mathematical notation:\n{text}\n\nFormal translation:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation
        if "translation:" in full_output.lower():
            translation = full_output.split("translation:")[-1].strip()
        else:
            translation = full_output[len(prompt):].strip()
        
        return translation
    
    def evaluate(self, test_samples: List[Dict]) -> Dict:
        """Evaluate model on test samples"""
        results = {
            'translations': [],
            'by_direction': defaultdict(list),
            'by_complexity': defaultdict(list),
            'metrics': {}
        }
        
        logger.info(f"Evaluating {len(test_samples)} samples...")
        
        for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
            try:
                # Translate
                start_time = time.time()
                translation = self.translate(sample['input'], sample['direction'])
                gen_time = time.time() - start_time
                
                # Compute complexity (simple heuristic)
                complexity = 1
                if any(c in sample['input'] for c in ['∀', '∃', '∈']):
                    complexity = 2
                if any(c in sample['input'] for c in ['∧', '∨', '→']):
                    complexity = 3
                if any(c in sample['input'] for c in ['∫', '∂', '∑']):
                    complexity = 4
                
                result = {
                    'input': sample['input'][:100],
                    'expected': sample['expected'][:100],
                    'output': translation[:100],
                    'direction': sample['direction'],
                    'complexity': complexity,
                    'gen_time': gen_time
                }
                
                results['translations'].append(result)
                results['by_direction'][sample['direction']].append(result)
                results['by_complexity'][f'level_{complexity}'].append(result)
                
            except Exception as e:
                logger.debug(f"Translation failed: {e}")
                continue
            
            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate metrics
        results['metrics'] = self._calculate_metrics(results)
        
        return results
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        all_translations = results['translations']
        if all_translations:
            metrics['total_samples'] = len(all_translations)
            metrics['avg_gen_time'] = np.mean([r['gen_time'] for r in all_translations])
            
            # By direction
            for direction in ['formal_to_informal', 'informal_to_formal']:
                dir_results = results['by_direction'].get(direction, [])
                if dir_results:
                    metrics[direction] = {
                        'count': len(dir_results),
                        'avg_time': np.mean([r['gen_time'] for r in dir_results])
                    }
            
            # By complexity
            for level in range(1, 5):
                level_results = results['by_complexity'].get(f'level_{level}', [])
                if level_results:
                    metrics[f'complexity_{level}'] = {
                        'count': len(level_results),
                        'avg_time': np.mean([r['gen_time'] for r in level_results])
                    }
        
        return metrics
    
    def generate_report(self, results: Dict) -> str:
        """Generate evaluation report"""
        report = []
        report.append("="*60)
        report.append("FIRMA EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Model: {self.model_path}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        metrics = results.get('metrics', {})
        
        # Overall stats
        report.append("OVERALL STATISTICS:")
        report.append("-"*40)
        report.append(f"Total samples evaluated: {metrics.get('total_samples', 0)}")
        report.append(f"Average generation time: {metrics.get('avg_gen_time', 0):.3f}s")
        report.append("")
        
        # Direction analysis
        report.append("BY DIRECTION:")
        report.append("-"*40)
        for direction in ['formal_to_informal', 'informal_to_formal']:
            if direction in metrics:
                stats = metrics[direction]
                report.append(f"{direction}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        report.append("")
        
        # Complexity analysis
        report.append("BY COMPLEXITY:")
        report.append("-"*40)
        for level in range(1, 5):
            key = f'complexity_{level}'
            if key in metrics:
                stats = metrics[key]
                report.append(f"Level {level}:")
                report.append(f"  Samples: {stats['count']}")
                report.append(f"  Avg time: {stats['avg_time']:.3f}s")
        
        # Sample translations
        report.append("")
        report.append("SAMPLE TRANSLATIONS:")
        report.append("-"*40)
        
        for i, trans in enumerate(results['translations'][:5]):
            report.append(f"\nExample {i+1} ({trans['direction']}):")
            report.append(f"Input: {trans['input'][:100]}...")
            report.append(f"Output: {trans['output'][:100]}...")
            report.append(f"Time: {trans['gen_time']:.3f}s")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    """Main evaluation function"""
    
    # Model path
    model_path = Path("firma_model_t4/final")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Initialize evaluator
    evaluator = FIRMAEvaluator(str(model_path))
    
    # Load test data
    test_samples = evaluator.load_test_data()
    
    if not test_samples:
        logger.error("No test data found")
        return
    
    # Run evaluation
    results = evaluator.evaluate(test_samples)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    results_path = evaluator.results_dir / "evaluation_results.json"
    
    # Convert for JSON serialization
    save_results = {
        'metrics': results['metrics'],
        'sample_translations': results['translations'][:10],
        'by_direction_counts': {k: len(v) for k, v in results['by_direction'].items()},
        'by_complexity_counts': {k: len(v) for k, v in results['by_complexity'].items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()