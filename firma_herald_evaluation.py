#!/usr/bin/env python3
"""
evaluate_firma_herald.py - Comprehensive evaluation of FIRMA on Herald_proofs dataset
Includes BLEU, ROUGE-L, and Lean4 proof validation metrics
Addresses reviewer concerns about quantitative evaluation
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import logging
from datetime import datetime
import time
import subprocess
import tempfile
import re

# HuggingFace imports
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Metrics imports
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for Herald_proofs evaluation"""
    # Model paths
    model_path: str = "./firma_model/final"
    base_model: str = "Qwen/Qwen3-8B"  # Will be overridden if model has different base
    
    # Dataset settings
    dataset_name: str = "FrenzyMath/Herald_proofs"
    dataset_split: str = "test"  # Use test split for evaluation
    max_samples: Optional[int] = 300  # Limit samples for faster evaluation
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temperature for more deterministic outputs
    top_p: float = 0.95
    do_sample: bool = True
    num_beams: int = 1
    
    # Evaluation settings
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Lean4 settings
    check_lean4: bool = True
    lean4_timeout: int = 10  # seconds per proof
    lean4_executable: str = "lean4"  # Path to Lean4 executable
    
    # Output settings
    output_dir: str = "./evaluation_results"
    save_detailed_results: bool = True
    
    # Metrics to compute
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_exact_match: bool = True
    compute_semantic_similarity: bool = True
    compute_proof_validity: bool = True

@dataclass
class EvaluationResult:
    """Store evaluation results for a single sample"""
    sample_id: str
    formal_theorem: str
    informal_theorem: str
    formal_proof_reference: str
    informal_proof_reference: str
    
    # Generated outputs
    formal_to_informal_output: str
    informal_to_formal_output: str
    
    # Metrics
    formal_to_informal_bleu: float
    informal_to_formal_bleu: float
    formal_to_informal_rouge_l: float
    informal_to_formal_rouge_l: float
    
    # Proof alignment metrics
    proof_alignment_bleu: float
    proof_alignment_rouge_l: float
    
    # Lean4 validation
    lean4_valid: Optional[bool] = None
    lean4_error: Optional[str] = None
    
    # Timing
    generation_time: float = 0.0
    validation_time: float = 0.0
    
    # Additional metrics
    exact_match_formal: bool = False
    exact_match_informal: bool = False
    complexity_level: int = 1

class Lean4Validator:
    """Validate formal proofs using Lean4"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.lean4_available = self._check_lean4_available()
        
    def _check_lean4_available(self) -> bool:
        """Check if Lean4 is available"""
        try:
            result = subprocess.run(
                [self.config.lean4_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Lean4 available: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        logger.warning("Lean4 not available - proof validation will be skipped")
        return False
    
    def validate_proof(self, theorem: str, proof: str) -> Tuple[bool, Optional[str]]:
        """Validate a formal proof using Lean4"""
        if not self.lean4_available:
            return None, "Lean4 not available"
        
        # Create temporary Lean4 file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            # Write basic Lean4 structure
            lean_content = f"""
-- Auto-generated Lean4 proof validation
import Mathlib.Tactic

-- Theorem statement
{theorem}

-- Proof attempt
{proof}
"""
            f.write(lean_content)
            temp_file = f.name
        
        try:
            # Run Lean4 validation
            result = subprocess.run(
                [self.config.lean4_executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.lean4_timeout
            )
            
            # Check for errors
            if result.returncode == 0:
                return True, None
            else:
                # Extract error message
                error_msg = result.stderr if result.stderr else result.stdout
                error_lines = [line for line in error_msg.split('\n') if 'error' in line.lower()]
                return False, '; '.join(error_lines[:3])  # Return first 3 error lines
                
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

class FIRMAEvaluator:
    """Main evaluator for FIRMA on Herald_proofs dataset"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self._load_model()
        self._load_dataset()
        self.lean4_validator = Lean4Validator(config)
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load FIRMA model with error handling"""
        logger.info(f"Loading FIRMA model from {self.config.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if it's a PEFT model
            peft_config_path = Path(self.config.model_path) / "adapter_config.json"
            
            if peft_config_path.exists():
                # Load as PEFT model
                logger.info("Loading as PEFT model")
                
                # Determine base model from adapter config
                with open(peft_config_path, 'r') as f:
                    peft_config = json.load(f)
                    base_model_name = peft_config.get('base_model_name_or_path', self.config.base_model)
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Load PEFT adapters
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                self.model = self.model.merge_and_unload()  # Merge for faster inference
            else:
                # Load as regular model
                logger.info("Loading as regular model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True
                )
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_dataset(self):
        """Load Herald_proofs dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
            
            # Limit samples if specified
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            
            self.dataset = dataset
            logger.info(f"Loaded {len(self.dataset)} samples from Herald_proofs")
            
            # Log column names for verification
            logger.info(f"Dataset columns: {self.dataset.column_names}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def generate_translation(self, prompt: str) -> str:
        """Generate translation using FIRMA model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in generated:
                generated = generated[len(prompt):].strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def compute_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BLEU and ROUGE-L scores"""
        metrics = {}
        
        if not reference or not hypothesis:
            return {
                'bleu': 0.0,
                'rouge_l': 0.0
            }
        
        # Compute BLEU
        if self.config.compute_bleu:
            try:
                bleu = corpus_bleu([hypothesis], [[reference]]).score / 100.0
                metrics['bleu'] = bleu
            except:
                metrics['bleu'] = 0.0
        
        # Compute ROUGE-L
        if self.config.compute_rouge:
            try:
                rouge_scores = self.rouge_scorer.score(reference, hypothesis)
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics['rouge_l'] = 0.0
        
        return metrics
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single sample"""
        start_time = time.time()
        
        # Extract fields
        sample_id = str(sample.get('id', 'unknown'))
        formal_theorem = sample.get('formal_theorem', '')
        informal_theorem = sample.get('informal_theorem', '')
        formal_proof = sample.get('formal_proof', '')
        informal_proof = sample.get('informal_proof', '')
        
        # Generate translations
        # Formal to Informal
        f2i_prompt = f"Translate formal to informal:\n{formal_theorem}\nInformal:"
        formal_to_informal = self.generate_translation(f2i_prompt)
        
        # Informal to Formal  
        i2f_prompt = f"Translate informal to formal:\n{informal_theorem}\nFormal:"
        informal_to_formal = self.generate_translation(i2f_prompt)
        
        generation_time = time.time() - start_time
        
        # Compute translation metrics
        f2i_metrics = self.compute_metrics(informal_theorem, formal_to_informal)
        i2f_metrics = self.compute_metrics(formal_theorem, informal_to_formal)
        
        # Compute proof alignment metrics
        # Compare generated informal with reference informal proof
        proof_align_metrics = self.compute_metrics(informal_proof, formal_to_informal)
        
        # Validate formal proof with Lean4
        validation_start = time.time()
        lean4_valid = None
        lean4_error = None
        
        if self.config.check_lean4 and informal_to_formal:
            lean4_valid, lean4_error = self.lean4_validator.validate_proof(
                formal_theorem, informal_to_formal
            )
        
        validation_time = time.time() - validation_start
        
        # Check exact matches
        exact_match_formal = informal_to_formal.strip() == formal_theorem.strip()
        exact_match_informal = formal_to_informal.strip() == informal_theorem.strip()
        
        # Estimate complexity
        complexity = self._estimate_complexity(formal_theorem + informal_theorem)
        
        return EvaluationResult(
            sample_id=sample_id,
            formal_theorem=formal_theorem,
            informal_theorem=informal_theorem,
            formal_proof_reference=formal_proof,
            informal_proof_reference=informal_proof,
            formal_to_informal_output=formal_to_informal,
            informal_to_formal_output=informal_to_formal,
            formal_to_informal_bleu=f2i_metrics.get('bleu', 0.0),
            informal_to_formal_bleu=i2f_metrics.get('bleu', 0.0),
            formal_to_informal_rouge_l=f2i_metrics.get('rouge_l', 0.0),
            informal_to_formal_rouge_l=i2f_metrics.get('rouge_l', 0.0),
            proof_alignment_bleu=proof_align_metrics.get('bleu', 0.0),
            proof_alignment_rouge_l=proof_align_metrics.get('rouge_l', 0.0),
            lean4_valid=lean4_valid,
            lean4_error=lean4_error,
            generation_time=generation_time,
            validation_time=validation_time,
            exact_match_formal=exact_match_formal,
            exact_match_informal=exact_match_informal,
            complexity_level=complexity
        )
    
    def _estimate_complexity(self, text: str) -> int:
        """Estimate complexity level of mathematical statement"""
        complexity_indicators = {
            1: ['=', '+', '-', 'equals', 'sum'],
            2: ['∀', '∃', '∈', 'for all', 'exists'],
            3: ['∧', '∨', '→', '⇒', 'implies', 'if and only if'],
            4: ['∫', '∂', 'lim', 'topology', 'manifold', 'category']
        }
        
        max_level = 1
        for level, indicators in complexity_indicators.items():
            if any(ind in text for ind in indicators):
                max_level = max(max_level, level)
        
        return min(max_level, 4)
    
    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation"""
        logger.info("Starting evaluation on Herald_proofs dataset")
        
        results = []
        
        # Process samples with progress bar
        for sample in tqdm(self.dataset, desc="Evaluating samples"):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate sample {sample.get('id', 'unknown')}: {e}")
                continue
        
        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(results)
        
        # Save results
        self._save_results(results, aggregate_metrics)
        
        return aggregate_metrics
    
    def _compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute aggregate metrics across all samples"""
        if not results:
            return {}
        
        metrics = {
            'num_samples': len(results),
            'timestamp': datetime.now().isoformat(),
            
            # Translation metrics
            'formal_to_informal': {
                'bleu': np.mean([r.formal_to_informal_bleu for r in results]),
                'rouge_l': np.mean([r.formal_to_informal_rouge_l for r in results]),
                'exact_match': sum(r.exact_match_informal for r in results) / len(results)
            },
            'informal_to_formal': {
                'bleu': np.mean([r.informal_to_formal_bleu for r in results]),
                'rouge_l': np.mean([r.informal_to_formal_rouge_l for r in results]),
                'exact_match': sum(r.exact_match_formal for r in results) / len(results)
            },
            
            # Proof alignment
            'proof_alignment': {
                'bleu': np.mean([r.proof_alignment_bleu for r in results]),
                'rouge_l': np.mean([r.proof_alignment_rouge_l for r in results])
            },
            
            # Timing
            'avg_generation_time': np.mean([r.generation_time for r in results]),
            'avg_validation_time': np.mean([r.validation_time for r in results]),
            
            # Complexity breakdown
            'complexity_distribution': {}
        }
        
        # Lean4 validation results
        lean4_results = [r for r in results if r.lean4_valid is not None]
        if lean4_results:
            valid_count = sum(r.lean4_valid for r in lean4_results)
            metrics['lean4_validation'] = {
                'checked_samples': len(lean4_results),
                'valid_proofs': valid_count,
                'pass_rate': valid_count / len(lean4_results),
                'common_errors': self._analyze_lean4_errors(lean4_results)
            }
        
        # Complexity-wise performance
        for level in range(1, 5):
            level_results = [r for r in results if r.complexity_level == level]
            if level_results:
                metrics['complexity_distribution'][f'level_{level}'] = {
                    'count': len(level_results),
                    'avg_bleu': np.mean([
                        (r.formal_to_informal_bleu + r.informal_to_formal_bleu) / 2 
                        for r in level_results
                    ]),
                    'avg_rouge_l': np.mean([
                        (r.formal_to_informal_rouge_l + r.informal_to_formal_rouge_l) / 2 
                        for r in level_results
                    ])
                }
        
        return metrics
    
    def _analyze_lean4_errors(self, results: List[EvaluationResult]) -> Dict[str, int]:
        """Analyze common Lean4 validation errors"""
        error_counts = {}
        
        for result in results:
            if result.lean4_error:
                # Categorize error
                error_type = "unknown"
                if "timeout" in result.lean4_error.lower():
                    error_type = "timeout"
                elif "syntax" in result.lean4_error.lower():
                    error_type = "syntax_error"
                elif "type" in result.lean4_error.lower():
                    error_type = "type_error"
                elif "tactic" in result.lean4_error.lower():
                    error_type = "tactic_error"
                
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
    def _save_results(self, results: List[EvaluationResult], aggregate_metrics: Dict[str, Any]):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        metrics_file = Path(self.config.output_dir) / f"herald_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        logger.info(f"Saved aggregate metrics to {metrics_file}")
        
        # Save detailed results if requested
        if self.config.save_detailed_results:
            detailed_file = Path(self.config.output_dir) / f"herald_detailed_{timestamp}.json"
            detailed_data = [asdict(r) for r in results]
            with open(detailed_file, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            logger.info(f"Saved detailed results to {detailed_file}")
        
        # Print summary
        self._print_summary(aggregate_metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("FIRMA EVALUATION RESULTS ON HERALD_PROOFS")
        print("="*60)
        
        print(f"\nSamples evaluated: {metrics['num_samples']}")
        
        print("\nTranslation Performance:")
        print(f"  Formal → Informal:")
        print(f"    BLEU:        {metrics['formal_to_informal']['bleu']:.3f}")
        print(f"    ROUGE-L:     {metrics['formal_to_informal']['rouge_l']:.3f}")
        print(f"    Exact Match: {metrics['formal_to_informal']['exact_match']:.1%}")
        
        print(f"  Informal → Formal:")
        print(f"    BLEU:        {metrics['informal_to_formal']['bleu']:.3f}")
        print(f"    ROUGE-L:     {metrics['informal_to_formal']['rouge_l']:.3f}")
        print(f"    Exact Match: {metrics['informal_to_formal']['exact_match']:.1%}")
        
        print(f"\nProof Alignment:")
        print(f"  BLEU:    {metrics['proof_alignment']['bleu']:.3f}")
        print(f"  ROUGE-L: {metrics['proof_alignment']['rouge_l']:.3f}")
        
        if 'lean4_validation' in metrics:
            print(f"\nLean4 Validation:")
            print(f"  Samples checked: {metrics['lean4_validation']['checked_samples']}")
            print(f"  Valid proofs:    {metrics['lean4_validation']['valid_proofs']}")
            print(f"  Pass rate:       {metrics['lean4_validation']['pass_rate']:.1%}")
            
            if metrics['lean4_validation']['common_errors']:
                print(f"  Common errors:")
                for error_type, count in metrics['lean4_validation']['common_errors'].items():
                    print(f"    - {error_type}: {count}")
        
        print(f"\nAverage generation time: {metrics['avg_generation_time']:.2f}s")
        
        if metrics['complexity_distribution']:
            print(f"\nPerformance by Complexity:")
            for level, stats in sorted(metrics['complexity_distribution'].items()):
                print(f"  {level}: {stats['count']} samples, "
                      f"BLEU: {stats['avg_bleu']:.3f}, "
                      f"ROUGE-L: {stats['avg_rouge_l']:.3f}")
        
        print("="*60 + "\n")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FIRMA on Herald_proofs dataset")
    parser.add_argument("--model-path", type=str, default="./firma_model/final",
                        help="Path to FIRMA model")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--no-lean4", action="store_true",
                        help="Skip Lean4 validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_path=args.model_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        check_lean4=not args.no_lean4,
        device=args.device
    )
    
    # Run evaluation
    evaluator = FIRMAEvaluator(config)
    metrics = evaluator.evaluate()
    
    logger.info("Evaluation complete!")
    

if __name__ == "__main__":
    main()
