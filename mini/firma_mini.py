#!/usr/bin/env python3
"""
firma_mini.py - Lightweight FIRMA for quick testing and visualization
Perfect for running 100 sample tests and generating paper figures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FIRMAMiniConfig:
    """Lightweight config for testing"""
    # Use smallest possible model for speed
    base_model: str = "Qwen/Qwen2.5-Math-7B-Instruct"  # 2.7B params, very fast
    # Alternative: "EleutherAI/pythia-410m" for even faster
    
    # Reduced dimensions for speed
    hidden_dim: int = 256
    num_complexity_levels: int = 4
    
    # Testing parameters
    max_length: int = 128  # Shorter for speed
    batch_size: int = 8  # Can handle larger batches
    
    # No training needed for testing
    use_pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ComplexityAnalyzer:
    """Analyze mathematical complexity"""
    
    def __init__(self):
        self.formal_indicators = {
            1: ['=', '+', '-', '×', '÷'],
            2: ['∀', '∃', '∈', '∉', '⊂', '⊆'],
            3: ['∧', '∨', '¬', '→', '↔', '⇒', '⇔'],
            4: ['∫', '∂', '∇', '∑', '∏', '⊢', '⊨', 'lim']
        }
        
        self.informal_indicators = {
            1: ['equals', 'plus', 'minus', 'times', 'divided'],
            2: ['for all', 'exists', 'element of', 'subset'],
            3: ['and', 'or', 'not', 'implies', 'if and only if'],
            4: ['integral', 'derivative', 'limit', 'sum', 'product']
        }
    
    def compute_complexity(self, text: str) -> int:
        """Compute complexity level 1-4"""
        text_lower = text.lower()
        max_level = 1
        
        # Check formal indicators
        for level, indicators in self.formal_indicators.items():
            if any(ind in text for ind in indicators):
                max_level = max(max_level, level)
        
        # Check informal indicators
        for level, indicators in self.informal_indicators.items():
            if any(ind in text_lower for ind in indicators):
                max_level = max(max_level, level)
        
        return min(max_level, 4)
    
    def detect_formality(self, text: str) -> str:
        """Detect if text is formal or informal"""
        formal_chars = ['∀', '∃', '∈', '→', '∧', '∨', '∫', '∂', '⊢']
        formal_count = sum(1 for char in formal_chars if char in text)
        
        # Also check for LaTeX-style notation
        latex_indicators = ['\\forall', '\\exists', '\\int', '\\sum', '\\frac']
        latex_count = sum(1 for ind in latex_indicators if ind in text)
        
        if formal_count > 0 or latex_count > 0:
            return "formal"
        else:
            return "informal"

class FIRMAMini(nn.Module):
    """Lightweight FIRMA for testing"""
    
    def __init__(self, config: FIRMAMiniConfig):
        super().__init__()
        self.config = config
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Load small model
        logger.info(f"Loading {config.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map="auto" if config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded on {config.device}")
    
    def translate(self, text: str, target_style: Optional[str] = None) -> Dict:
        """Translate between formal and informal"""
        
        # Analyze input
        complexity = self.complexity_analyzer.compute_complexity(text)
        detected_style = self.complexity_analyzer.detect_formality(text)
        
        # Determine target style
        if target_style is None:
            target_style = "informal" if detected_style == "formal" else "formal"
        
        # Create prompt
        if target_style == "informal":
            prompt = f"Translate this formal mathematical statement to informal language:\n{text}\n\nInformal translation:"
        else:
            prompt = f"Translate this informal statement to formal mathematical notation:\n{text}\n\nFormal translation:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation
        if "translation:" in full_output.lower():
            translation = full_output.split("translation:")[-1].strip()
        else:
            # Fallback: take everything after the input
            translation = full_output[len(prompt):].strip()
        
        return {
            'input': text,
            'output': translation,
            'source_style': detected_style,
            'target_style': target_style,
            'complexity': complexity,
            'generation_time': generation_time,
            'prompt_length': len(inputs['input_ids'][0]),
            'output_length': len(translation.split())
        }
    
    def batch_translate(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """Translate multiple texts"""
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Translating")
        else:
            iterator = texts
        
        for text in iterator:
            try:
                result = self.translate(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Translation failed for: {text[:50]}... Error: {e}")
                results.append({
                    'input': text,
                    'output': '',
                    'error': str(e),
                    'complexity': self.complexity_analyzer.compute_complexity(text)
                })
        
        return results

def generate_test_samples(n_samples: int = 400) -> List[Dict]:
    """Load test samples from actual dataset files"""
    
    samples = []
    data_dir = Path("./FIRMA/mini/math_alignment_dataset")
    
    # File configurations with column names
    file_configs = [
        ("train_clean.json", "informal_statement", "formal_statement"),
        ("valid_clean.json", "informal_stmt", "formal_statement"),
        ("test_clean.json", "informal_stmt", "formal_statement")
    ]
    
    analyzer = ComplexityAnalyzer()
    
    for filename, informal_key, formal_key in file_configs:
        filepath = data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both list and single object formats
            if not isinstance(data, list):
                data = [data]
            
            # Process each item
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Get formal and informal statements
                formal = item.get(formal_key, "")
                informal = item.get(informal_key, "")
                
                if not formal or not informal:
                    continue
                
                # Compute complexity
                complexity = analyzer.compute_complexity(formal + " " + informal)
                
                # Add both directions
                samples.append({
                    'text': formal[:500],  # Truncate for speed
                    'direction': 'formal_to_informal',
                    'expected': informal[:500],
                    'complexity': complexity
                })
                
                samples.append({
                    'text': informal[:500],
                    'direction': 'informal_to_formal', 
                    'expected': formal[:500],
                    'complexity': complexity
                })
                
                # Stop if we have enough samples
                if len(samples) >= n_samples * 2:
                    break
                    
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
        
        if len(samples) >= n_samples * 2:
            break
    
    # Shuffle and limit to n_samples
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(samples)
    
    # If we have fewer samples than requested, use what we have
    final_samples = samples[:n_samples] if samples else []
    
    logger.info(f"Loaded {len(final_samples)} samples from dataset files")
    
    return final_samples

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    
    metrics = {
        'total_samples': len(results),
        'successful_translations': sum(1 for r in results if 'error' not in r),
        'avg_generation_time': np.mean([r.get('generation_time', 0) for r in results]),
        'avg_output_length': np.mean([r.get('output_length', 0) for r in results]),
        'by_complexity': {},
        'by_direction': {}
    }
    
    # Metrics by complexity
    for level in range(1, 5):
        level_results = [r for r in results if r.get('complexity') == level]
        if level_results:
            metrics['by_complexity'][f'level_{level}'] = {
                'count': len(level_results),
                'avg_time': np.mean([r.get('generation_time', 0) for r in level_results]),
                'avg_length': np.mean([r.get('output_length', 0) for r in level_results])
            }
    
    # Metrics by direction
    for direction in ['formal_to_informal', 'informal_to_formal']:
        dir_results = [r for r in results if r.get('source_style') == direction.split('_')[0]]
        if dir_results:
            metrics['by_direction'][direction] = {
                'count': len(dir_results),
                'avg_time': np.mean([r.get('generation_time', 0) for r in dir_results]),
                'avg_length': np.mean([r.get('output_length', 0) for r in dir_results])
            }
    
    return metrics

if __name__ == "__main__":
    # Quick test
    print("Initializing FIRMA Mini...")
    
    config = FIRMAMiniConfig()
    model = FIRMAMini(config)
    
    # Load samples from actual dataset
    print("\nLoading samples from dataset files...")
    test_samples = generate_test_samples(n_samples=450)
    
    if test_samples:
        # Test a few samples
        print(f"\nTesting on {min(3, len(test_samples))} samples...")
        for sample in test_samples[:3]:
            result = model.translate(sample['text'])
            print(f"\nInput: {result['input'][:100]}...")
            print(f"Output: {result['output'][:100]}...")
            print(f"Complexity: Level {result['complexity']}")
            print(f"Time: {result['generation_time']:.2f}s")
    else:
        print("No samples loaded. Please check dataset files.")
    
    print("\n✓ FIRMA Mini ready for testing!")
