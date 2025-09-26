#!/usr/bin/env python3
"""
baseline_comparison.py - Compare FIRMA with REAL-Prover baseline model
Addresses reviewer request for performance comparison with traditional methods
REAL-Prover is based on Qwen2-7B-Math and fine-tuned on Herald_proofs dataset
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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace imports
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# Metrics imports
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for baseline comparison"""
    # Model paths
    firma_model_path: str = "./firma_model/final"
    baseline_model_name: str = "FrenzyMath/REAL-Prover"  # HuggingFace model
    
    # Base models (for loading if needed)
    firma_base_model: str = "Qwen/Qwen3-8B"
    baseline_base_model: str = "Qwen/Qwen2-7B-Math"  # REAL-Prover base
    
    # Dataset settings
    dataset_name: str = "FrenzyMath/Herald_proofs"
    dataset_split: str = "test"
    max_samples: Optional[int] = 230
    stratified_sampling: bool = True  # Sample across complexity levels
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True
    num_beams: int = 1
    
    # Evaluation settings
    batch_size: int = 2  # Lower for memory efficiency with two models
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    
    # Output settings
    output_dir: str = "./baseline_comparison_results"
    save_detailed_results: bool = True
    generate_plots: bool = True
    
    # Statistical tests
    run_statistical_tests: bool = True
    confidence_level: float = 0.95

@dataclass
class ModelResult:
    """Results for a single model on a single sample"""
    model_name: str
    sample_id: str
    
    # Translations
    formal_to_informal: str
    informal_to_formal: str
    
    # Metrics
    f2i_bleu: float
    i2f_bleu: float
    f2i_rouge_l: float
    i2f_rouge_l: float
    
    # Timing
    generation_time: float
    
    # Additional
    complexity_level: int
    avg_score: float  # Average of all metrics

class BaselineComparator:
    """Compare FIRMA with baseline models"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self._load_models()
        self._load_dataset()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.firma_results = []
        self.baseline_results = []
    
    def _load_models(self):
        """Load both FIRMA and baseline models"""
        logger.info("Loading models for comparison...")
        
        # Quantization config for memory efficiency
        bnb_config = None
        if self.config.use_4bit and self.config.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load FIRMA model
        self._load_firma_model(bnb_config)
        
        # Clear cache before loading second model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load baseline model
        self._load_baseline_model(bnb_config)
        
        logger.info("Both models loaded successfully")
    
    def _load_firma_model(self, bnb_config):
        """Load FIRMA model"""
        logger.info(f"Loading FIRMA from {self.config.firma_model_path}")
        
        try:
            # Load FIRMA tokenizer
            self.firma_tokenizer = AutoTokenizer.from_pretrained(
                self.config.firma_model_path,
                trust_remote_code=True
            )
            
            if self.firma_tokenizer.pad_token is None:
                self.firma_tokenizer.pad_token = self.firma_tokenizer.eos_token
            
            # Check if PEFT model
            peft_config_path = Path(self.config.firma_model_path) / "adapter_config.json"
            
            if peft_config_path.exists():
                # Load as PEFT model
                with open(peft_config_path, 'r') as f:
                    peft_config = json.load(f)
                    base_model_name = peft_config.get('base_model_name_or_path', self.config.firma_base_model)
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.firma_model = PeftModel.from_pretrained(base_model, self.config.firma_model_path)
                self.firma_model = self.firma_model.merge_and_unload()
            else:
                # Regular model
                self.firma_model = AutoModelForCausalLM.from_pretrained(
                    self.config.firma_model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.firma_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load FIRMA model: {e}")
            raise
    
    def _load_baseline_model(self, bnb_config):
        """Load REAL-Prover baseline model"""
        logger.info(f"Loading baseline model: {self.config.baseline_model_name}")
        
        try:
            # Load baseline tokenizer
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(
                self.config.baseline_model_name,
                trust_remote_code=True
            )
            
            if self.baseline_tokenizer.pad_token is None:
                self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
            
            # Load baseline model
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                self.config.baseline_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.baseline_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            # Try fallback to base model
            logger.info("Trying to load base model as fallback...")
            
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(
                self.config.baseline_base_model,
                trust_remote_code=True
            )
            
            if self.baseline_tokenizer.pad_token is None:
                self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
            
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                self.config.baseline_base_model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.baseline_model.eval()
            logger.info("Loaded base model as baseline")
    
    def _load_dataset(self):
        """Load Herald_proofs dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
            
            # Stratified sampling if requested
            if self.config.stratified_sampling and self.config.max_samples:
                dataset = self._stratified_sample(dataset, self.config.max_samples)
            elif self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            
            self.dataset = dataset
            logger.info(f"Loaded {len(self.dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _stratified_sample(self, dataset, n_samples):
        """Perform stratified sampling across complexity levels"""
        # Estimate complexity for all samples
        complexities = []
        for sample in dataset:
            text = sample.get('formal_theorem', '') + sample.get('informal_theorem', '')
            complexity = self._estimate_complexity(text)
            complexities.append(complexity)
        
        # Stratified sampling
        samples_per_level = n_samples // 4
        selected_indices = []
        
        for level in range(1, 5):
            level_indices = [i for i, c in enumerate(complexities) if c == level]
            if level_indices:
                n_select = min(samples_per_level, len(level_indices))
                selected = np.random.choice(level_indices, n_select, replace=False)
                selected_indices.extend(selected)
        
        # Fill remaining with random samples if needed
        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(dataset)))
            available = list(all_indices - set(selected_indices))
            if available:
                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                selected_indices.extend(additional)
        
        return dataset.select(sorted(selected_indices))
    
    def _estimate_complexity(self, text: str) -> int:
        """Estimate complexity level"""
        complexity_indicators = {
            1: ['=', '+', '-', 'equals', 'sum', 'difference'],
            2: ['∀', '∃', '∈', 'for all', 'exists', 'element'],
            3: ['∧', '∨', '→', '⇒', 'implies', 'iff', 'homomorphism'],
            4: ['∫', '∂', 'lim', 'topology', 'manifold', 'functor', 'sheaf']
        }
        
        max_level = 1
        for level, indicators in complexity_indicators.items():
            if any(ind in text for ind in indicators):
                max_level = max(max_level, level)
        
        return min(max_level, 4)
    
    def generate_translation(self, model, tokenizer, prompt: str) -> str:
        """Generate translation using specified model"""
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if prompt in generated:
                generated = generated[len(prompt):].strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def compute_metrics(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """Compute BLEU and ROUGE-L scores"""
        if not reference or not hypothesis:
            return 0.0, 0.0
        
        # BLEU
        try:
            bleu = corpus_bleu([hypothesis], [[reference]]).score / 100.0
        except:
            bleu = 0.0
        
        # ROUGE-L
        try:
            rouge_scores = self.rouge_scorer.score(reference, hypothesis)
            rouge_l = rouge_scores['rougeL'].fmeasure
        except:
            rouge_l = 0.0
        
        return bleu, rouge_l
    
    def evaluate_sample(self, model, tokenizer, model_name: str, sample: Dict[str, Any]) -> ModelResult:
        """Evaluate a single sample with a model"""
        start_time = time.time()
        
        # Extract fields
        sample_id = str(sample.get('id', 'unknown'))
        formal_theorem = sample.get('formal_theorem', '')
        informal_theorem = sample.get('informal_theorem', '')
        
        # Generate translations
        f2i_prompt = f"Translate formal to informal:\n{formal_theorem}\nInformal:"
        formal_to_informal = self.generate_translation(model, tokenizer, f2i_prompt)
        
        i2f_prompt = f"Translate informal to formal:\n{informal_theorem}\nFormal:"
        informal_to_formal = self.generate_translation(model, tokenizer, i2f_prompt)
        
        generation_time = time.time() - start_time
        
        # Compute metrics
        f2i_bleu, f2i_rouge = self.compute_metrics(informal_theorem, formal_to_informal)
        i2f_bleu, i2f_rouge = self.compute_metrics(formal_theorem, informal_to_formal)
        
        # Average score
        avg_score = (f2i_bleu + i2f_bleu + f2i_rouge + i2f_rouge) / 4
        
        # Complexity
        complexity = self._estimate_complexity(formal_theorem + informal_theorem)
        
        return ModelResult(
            model_name=model_name,
            sample_id=sample_id,
            formal_to_informal=formal_to_informal,
            informal_to_formal=informal_to_formal,
            f2i_bleu=f2i_bleu,
            i2f_bleu=i2f_bleu,
            f2i_rouge_l=f2i_rouge,
            i2f_rouge_l=i2f_rouge,
            generation_time=generation_time,
            complexity_level=complexity,
            avg_score=avg_score
        )
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run full comparison between models"""
        logger.info("Starting model comparison...")
        
        # Process samples
        for sample in tqdm(self.dataset, desc="Evaluating samples"):
            try:
                # Evaluate with FIRMA
                firma_result = self.evaluate_sample(
                    self.firma_model, 
                    self.firma_tokenizer,
                    "FIRMA",
                    sample
                )
                self.firma_results.append(firma_result)
                
                # Clear cache between models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Evaluate with baseline
                baseline_result = self.evaluate_sample(
                    self.baseline_model,
                    self.baseline_tokenizer,
                    "REAL-Prover",
                    sample
                )
                self.baseline_results.append(baseline_result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate sample {sample.get('id', 'unknown')}: {e}")
                continue
        
        # Compute comparative metrics
        comparison_metrics = self._compute_comparison_metrics()
        
        # Run statistical tests
        if self.config.run_statistical_tests:
            statistical_results = self._run_statistical_tests()
            comparison_metrics['statistical_tests'] = statistical_results
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_comparison_plots()
        
        # Save results
        self._save_comparison_results(comparison_metrics)
        
        return comparison_metrics
    
    def _compute_comparison_metrics(self) -> Dict[str, Any]:
        """Compute comparative metrics between models"""
        
        def compute_model_metrics(results: List[ModelResult]) -> Dict[str, Any]:
            if not results:
                return {}
            
            return {
                'formal_to_informal': {
                    'bleu': np.mean([r.f2i_bleu for r in results]),
                    'rouge_l': np.mean([r.f2i_rouge_l for r in results]),
                    'std_bleu': np.std([r.f2i_bleu for r in results]),
                    'std_rouge': np.std([r.f2i_rouge_l for r in results])
                },
                'informal_to_formal': {
                    'bleu': np.mean([r.i2f_bleu for r in results]),
                    'rouge_l': np.mean([r.i2f_rouge_l for r in results]),
                    'std_bleu': np.std([r.i2f_bleu for r in results]),
                    'std_rouge': np.std([r.i2f_rouge_l for r in results])
                },
                'overall': {
                    'avg_score': np.mean([r.avg_score for r in results]),
                    'std_score': np.std([r.avg_score for r in results])
                },
                'timing': {
                    'avg_generation_time': np.mean([r.generation_time for r in results]),
                    'std_generation_time': np.std([r.generation_time for r in results])
                }
            }
        
        # Compute metrics for each model
        firma_metrics = compute_model_metrics(self.firma_results)
        baseline_metrics = compute_model_metrics(self.baseline_results)
        
        # Compute improvements
        improvements = {}
        if firma_metrics and baseline_metrics:
            improvements = {
                'f2i_bleu_improvement': (
                    firma_metrics['formal_to_informal']['bleu'] - 
                    baseline_metrics['formal_to_informal']['bleu']
                ) / baseline_metrics['formal_to_informal']['bleu'] * 100,
                'i2f_bleu_improvement': (
                    firma_metrics['informal_to_formal']['bleu'] - 
                    baseline_metrics['informal_to_formal']['bleu']
                ) / baseline_metrics['informal_to_formal']['bleu'] * 100,
                'overall_improvement': (
                    firma_metrics['overall']['avg_score'] - 
                    baseline_metrics['overall']['avg_score']
                ) / baseline_metrics['overall']['avg_score'] * 100,
                'speedup': baseline_metrics['timing']['avg_generation_time'] / 
                          firma_metrics['timing']['avg_generation_time']
            }
        
        # Complexity-wise comparison
        complexity_comparison = {}
        for level in range(1, 5):
            firma_level = [r for r in self.firma_results if r.complexity_level == level]
            baseline_level = [r for r in self.baseline_results if r.complexity_level == level]
            
            if firma_level and baseline_level:
                complexity_comparison[f'level_{level}'] = {
                    'count': len(firma_level),
                    'firma_avg': np.mean([r.avg_score for r in firma_level]),
                    'baseline_avg': np.mean([r.avg_score for r in baseline_level]),
                    'improvement': (
                        np.mean([r.avg_score for r in firma_level]) - 
                        np.mean([r.avg_score for r in baseline_level])
                    ) / np.mean([r.avg_score for r in baseline_level]) * 100
                }
        
        return {
            'num_samples': len(self.firma_results),
            'timestamp': datetime.now().isoformat(),
            'models': {
                'FIRMA': firma_metrics,
                'REAL-Prover': baseline_metrics
            },
            'improvements': improvements,
            'complexity_comparison': complexity_comparison
        }
    
    def _run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical significance tests"""
        results = {}
        
        # Paired t-tests for main metrics
        if len(self.firma_results) == len(self.baseline_results):
            # BLEU scores
            firma_bleu = [(r.f2i_bleu + r.i2f_bleu) / 2 for r in self.firma_results]
            baseline_bleu = [(r.f2i_bleu + r.i2f_bleu) / 2 for r in self.baseline_results]
            
            t_stat_bleu, p_value_bleu = stats.ttest_rel(firma_bleu, baseline_bleu)
            
            # ROUGE scores
            firma_rouge = [(r.f2i_rouge_l + r.i2f_rouge_l) / 2 for r in self.firma_results]
            baseline_rouge = [(r.f2i_rouge_l + r.i2f_rouge_l) / 2 for r in self.baseline_results]
            
            t_stat_rouge, p_value_rouge = stats.ttest_rel(firma_rouge, baseline_rouge)
            
            # Overall scores
            firma_overall = [r.avg_score for r in self.firma_results]
            baseline_overall = [r.avg_score for r in self.baseline_results]
            
            t_stat_overall, p_value_overall = stats.ttest_rel(firma_overall, baseline_overall)
            
            # Effect sizes (Cohen's d)
            def cohens_d(x, y):
                return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)
            
            results = {
                'bleu': {
                    't_statistic': float(t_stat_bleu),
                    'p_value': float(p_value_bleu),
                    'significant': p_value_bleu < (1 - self.config.confidence_level),
                    'effect_size': cohens_d(firma_bleu, baseline_bleu)
                },
                'rouge': {
                    't_statistic': float(t_stat_rouge),
                    'p_value': float(p_value_rouge),
                    'significant': p_value_rouge < (1 - self.config.confidence_level),
                    'effect_size': cohens_d(firma_rouge, baseline_rouge)
                },
                'overall': {
                    't_statistic': float(t_stat_overall),
                    'p_value': float(p_value_overall),
                    'significant': p_value_overall < (1 - self.config.confidence_level),
                    'effect_size': cohens_d(firma_overall, baseline_overall)
                }
            }
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_p_value = stats.wilcoxon(firma_overall, baseline_overall)
            results['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_p_value),
                'significant': w_p_value < (1 - self.config.confidence_level)
            }
        
        return results
    
    def _generate_comparison_plots(self):
        """Generate visualization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Prepare data
        firma_f2i_bleu = [r.f2i_bleu for r in self.firma_results]
        baseline_f2i_bleu = [r.f2i_bleu for r in self.baseline_results]
        firma_i2f_bleu = [r.i2f_bleu for r in self.firma_results]
        baseline_i2f_bleu = [r.i2f_bleu for r in self.baseline_results]
        
        # Plot 1: F2I BLEU comparison
        axes[0, 0].boxplot([baseline_f2i_bleu, firma_f2i_bleu], labels=['REAL-Prover', 'FIRMA'])
        axes[0, 0].set_title('Formal→Informal BLEU')
        axes[0, 0].set_ylabel('BLEU Score')
        
        # Plot 2: I2F BLEU comparison
        axes[0, 1].boxplot([baseline_i2f_bleu, firma_i2f_bleu], labels=['REAL-Prover', 'FIRMA'])
        axes[0, 1].set_title('Informal→Formal BLEU')
        axes[0, 1].set_ylabel('BLEU Score')
        
        # Plot 3: Overall score distribution
        firma_overall = [r.avg_score for r in self.firma_results]
        baseline_overall = [r.avg_score for r in self.baseline_results]
        
        axes[0, 2].hist([baseline_overall, firma_overall], label=['REAL-Prover', 'FIRMA'], 
                        alpha=0.7, bins=20)
        axes[0, 2].set_title('Overall Score Distribution')
        axes[0, 2].set_xlabel('Average Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # Plot 4: Complexity-wise performance
        complexity_levels = [1, 2, 3, 4]
        firma_by_complexity = []
        baseline_by_complexity = []
        
        for level in complexity_levels:
            firma_level = [r.avg_score for r in self.firma_results if r.complexity_level == level]
            baseline_level = [r.avg_score for r in self.baseline_results if r.complexity_level == level]
            firma_by_complexity.append(np.mean(firma_level) if firma_level else 0)
            baseline_by_complexity.append(np.mean(baseline_level) if baseline_level else 0)
        
        x = np.arange(len(complexity_levels))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, baseline_by_complexity, width, label='REAL-Prover')
        axes[1, 0].bar(x + width/2, firma_by_complexity, width, label='FIRMA')
        axes[1, 0].set_title('Performance by Complexity Level')
        axes[1, 0].set_xlabel('Complexity Level')
        axes[1, 0].set_ylabel('Average Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(complexity_levels)
        axes[1, 0].legend()
        
        # Plot 5: Generation time comparison
        firma_times = [r.generation_time for r in self.firma_results]
        baseline_times = [r.generation_time for r in self.baseline_results]
        
        axes[1, 1].boxplot([baseline_times, firma_times], labels=['REAL-Prover', 'FIRMA'])
        axes[1, 1].set_title('Generation Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        
        # Plot 6: Improvement percentages
        metrics = ['F2I BLEU', 'I2F BLEU', 'F2I ROUGE', 'I2F ROUGE', 'Overall']
        improvements = []
        
        if self.firma_results and self.baseline_results:
            # Calculate improvements
            firma_metrics_avg = [
                np.mean(firma_f2i_bleu),
                np.mean(firma_i2f_bleu),
                np.mean([r.f2i_rouge_l for r in self.firma_results]),
                np.mean([r.i2f_rouge_l for r in self.firma_results]),
                np.mean(firma_overall)
            ]
            
            baseline_metrics_avg = [
                np.mean(baseline_f2i_bleu),
                np.mean(baseline_i2f_bleu),
                np.mean([r.f2i_rouge_l for r in self.baseline_results]),
                np.mean([r.i2f_rouge_l for r in self.baseline_results]),
                np.mean(baseline_overall)
            ]
            
            improvements = [(f - b) / b * 100 if b > 0 else 0 
                          for f, b in zip(firma_metrics_avg, baseline_metrics_avg)]
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        axes[1, 2].bar(metrics, improvements, color=colors)
        axes[1, 2].set_title('FIRMA Improvement over REAL-Prover (%)')
        axes[1, 2].set_ylabel('Improvement (%)')
        axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config.output_dir) / f"comparison_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plots to {plot_path}")
        
        plt.close()
    
    def _save_comparison_results(self, metrics: Dict[str, Any]):
        """Save comparison results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary metrics
        summary_file = Path(self.config.output_dir) / f"comparison_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_file}")
        
        # Save detailed results if requested
        if self.config.save_detailed_results:
            detailed_file = Path(self.config.output_dir) / f"comparison_detailed_{timestamp}.json"
            detailed_data = {
                'firma_results': [asdict(r) for r in self.firma_results],
                'baseline_results': [asdict(r) for r in self.baseline_results]
            }
            with open(detailed_file, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            logger.info(f"Saved detailed results to {detailed_file}")
        
        # Generate report
        self._generate_report(metrics, timestamp)
    
    def _generate_report(self, metrics: Dict[str, Any], timestamp: str):
        """Generate human-readable comparison report"""
        report_file = Path(self.config.output_dir) / f"comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT: FIRMA vs REAL-Prover\n")
            f.write(f"Dataset: Herald_proofs | Samples: {metrics['num_samples']}\n")
            f.write(f"Generated: {metrics['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            firma = metrics['models']['FIRMA']
            baseline = metrics['models']['REAL-Prover']
            
            f.write(f"{'Metric':<25} {'FIRMA':>12} {'REAL-Prover':>12} {'Improvement':>12}\n")
            f.write("-"*62 + "\n")
            
            # F2I metrics
            f.write(f"{'Formal→Informal BLEU':<25} "
                   f"{firma['formal_to_informal']['bleu']:>12.3f} "
                   f"{baseline['formal_to_informal']['bleu']:>12.3f} "
                   f"{metrics['improvements'].get('f2i_bleu_improvement', 0):>11.1f}%\n")
            
            f.write(f"{'Formal→Informal ROUGE-L':<25} "
                   f"{firma['formal_to_informal']['rouge_l']:>12.3f} "
                   f"{baseline['formal_to_informal']['rouge_l']:>12.3f}\n")
            
            # I2F metrics
            f.write(f"{'Informal→Formal BLEU':<25} "
                   f"{firma['informal_to_formal']['bleu']:>12.3f} "
                   f"{baseline['informal_to_formal']['bleu']:>12.3f} "
                   f"{metrics['improvements'].get('i2f_bleu_improvement', 0):>11.1f}%\n")
            
            f.write(f"{'Informal→Formal ROUGE-L':<25} "
                   f"{firma['informal_to_formal']['rouge_l']:>12.3f} "
                   f"{baseline['informal_to_formal']['rouge_l']:>12.3f}\n")
            
            # Overall
            f.write(f"{'Overall Score':<25} "
                   f"{firma['overall']['avg_score']:>12.3f} "
                   f"{baseline['overall']['avg_score']:>12.3f} "
                   f"{metrics['improvements'].get('overall_improvement', 0):>11.1f}%\n")
            
            # Timing
            f.write(f"{'Avg Generation Time (s)':<25} "
                   f"{firma['timing']['avg_generation_time']:>12.2f} "
                   f"{baseline['timing']['avg_generation_time']:>12.2f} "
                   f"{metrics['improvements'].get('speedup', 1):>11.2f}x\n")
            
            # Statistical significance
            if 'statistical_tests' in metrics:
                f.write("\n\nSTATISTICAL SIGNIFICANCE\n")
                f.write("-"*40 + "\n")
                
                tests = metrics['statistical_tests']
                for test_name, test_results in tests.items():
                    if isinstance(test_results, dict) and 'p_value' in test_results:
                        f.write(f"{test_name.upper():}\n")
                        f.write(f"  p-value: {test_results['p_value']:.4f}\n")
                        f.write(f"  Significant: {'Yes' if test_results.get('significant', False) else 'No'}\n")
                        if 'effect_size' in test_results:
                            f.write(f"  Effect size (Cohen's d): {test_results['effect_size']:.3f}\n")
                        f.write("\n")
            
            # Complexity analysis
            if metrics.get('complexity_comparison'):
                f.write("\nPERFORMANCE BY COMPLEXITY LEVEL\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Level':<10} {'Samples':>10} {'FIRMA':>12} {'REAL-Prover':>12} {'Improvement':>12}\n")
                f.write("-"*56 + "\n")
                
                for level, data in sorted(metrics['complexity_comparison'].items()):
                    f.write(f"{level:<10} {data['count']:>10} "
                           f"{data['firma_avg']:>12.3f} "
                           f"{data['baseline_avg']:>12.3f} "
                           f"{data['improvement']:>11.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Generated report: {report_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("COMPARISON SUMMARY: FIRMA vs REAL-Prover")
        print("="*60)
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"\nFIRMA Performance:")
        print(f"  F→I BLEU:  {firma['formal_to_informal']['bleu']:.3f}")
        print(f"  I→F BLEU:  {firma['informal_to_formal']['bleu']:.3f}")
        print(f"  Overall:   {firma['overall']['avg_score']:.3f}")
        print(f"\nREAL-Prover Performance:")
        print(f"  F→I BLEU:  {baseline['formal_to_informal']['bleu']:.3f}")
        print(f"  I→F BLEU:  {baseline['informal_to_formal']['bleu']:.3f}")
        print(f"  Overall:   {baseline['overall']['avg_score']:.3f}")
        
        if 'improvements' in metrics:
            print(f"\nFIRMA Improvements:")
            print(f"  Overall: {metrics['improvements'].get('overall_improvement', 0):+.1f}%")
            print(f"  Speed:   {metrics['improvements'].get('speedup', 1):.2f}x faster")
        
        if 'statistical_tests' in metrics and 'overall' in metrics['statistical_tests']:
            sig = metrics['statistical_tests']['overall'].get('significant', False)
            print(f"\nStatistically significant: {'Yes' if sig else 'No'}")
            if sig:
                print(f"  p-value: {metrics['statistical_tests']['overall']['p_value']:.4f}")
        
        print("="*60 + "\n")


def main():
    """Main comparison function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare FIRMA with baseline models")
    parser.add_argument("--firma-path", type=str, default="./firma_model/final",
                        help="Path to FIRMA model")
    parser.add_argument("--baseline", type=str, default="FrenzyMath/REAL-Prover",
                        help="Baseline model name or path")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="./baseline_comparison_results",
                        help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ComparisonConfig(
        firma_model_path=args.firma_path,
        baseline_model_name=args.baseline,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        device=args.device
    )
    
    # Run comparison
    comparator = BaselineComparator(config)
    metrics = comparator.run_comparison()
    
    logger.info("Comparison complete!")
    

if __name__ == "__main__":
    main()
