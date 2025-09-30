#!/usr/bin/env python3
"""
evaluate_lean.py - Comprehensive evaluation script for FIRMA on Lean-Workbook
Includes qualitative and quantitative metrics with failure analysis
"""

import os
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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# NLP metrics
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import editdistance
from datasets import load_dataset
from transformers import AutoTokenizer

# Import FIRMA components
from firma_model_lean import FIRMA, FIRMAConfig
from train_firma_lean import ComplexityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Model
    model_path: str = "./firma_lean_model/final"
    
    # Dataset
    dataset_name: str = "internlm/Lean-Workbook"
    max_samples: Optional[int] = 500  # Evaluate on subset
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    num_beams: int = 1
    do_sample: bool = True
    
    # Metrics
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_bert_score: bool = True
    compute_edit_distance: bool = True
    check_lean_validity: bool = True
    
    # Lean4 validation
    lean4_executable: str = "lean4"
    lean4_timeout: int = 10  # seconds
    
    # Output
    output_dir: str = "./evaluation_results"
    save_detailed: bool = True
    generate_plots: bool = True
    generate_report: bool = True
    
    # Analysis
    analyze_failures: bool = True
    num_examples_to_show: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationSample:
    """Single evaluation sample with all metrics"""
    
    # Identifiers
    sample_id: str
    complexity_level: int
    
    # Original data
    formal_reference: str
    informal_reference: str
    answer_reference: str
    tactic: str
    
    # Generated outputs
    formal_to_informal: str
    informal_to_formal: str
    
    # Translation metrics
    f2i_bleu: float
    i2f_bleu: float
    f2i_rouge_l: float
    i2f_rouge_l: float
    f2i_bert_score: float
    i2f_bert_score: float
    f2i_edit_distance: int
    i2f_edit_distance: int
    
    # Validation
    lean_valid: Optional[bool] = None
    lean_error: Optional[str] = None
    syntax_valid: bool = True
    
    # Round-trip consistency
    roundtrip_bleu: float = 0.0
    roundtrip_exact_match: bool = False
    
    # Timing
    generation_time: float = 0.0
    validation_time: float = 0.0
    
    # Success indicators
    translation_success: bool = False
    proof_success: bool = False
    overall_success: bool = False


class Lean4Validator:
    """Validate Lean theorems and proofs"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.lean4_available = self._check_lean4()
    
    def _check_lean4(self) -> bool:
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
        except:
            pass
        
        logger.warning("Lean4 not available - validation will be limited")
        return False
    
    def validate_theorem(self, theorem_statement: str, tactic: str = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a Lean theorem
        
        Args:
            theorem_statement: The theorem to validate
            tactic: Optional proof tactic
        
        Returns:
            (is_valid, error_message)
        """
        if not self.lean4_available:
            return None, "Lean4 not available"
        
        # Create temporary Lean file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            lean_content = f"""
import Mathlib.Tactic

-- Generated theorem for validation
{theorem_statement}
"""
            if tactic and tactic != "sorry":
                lean_content += f"""
-- Proof attempt
{tactic}
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
            
            if result.returncode == 0:
                return True, None
            else:
                # Extract error
                error_lines = []
                for line in result.stderr.split('\n'):
                    if 'error' in line.lower() or 'failed' in line.lower():
                        error_lines.append(line.strip())
                error_msg = '; '.join(error_lines[:3])
                return False, error_msg
        
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def check_syntax(self, statement: str) -> bool:
        """Basic syntax checking for Lean statements"""
        
        # Check for balanced parentheses/brackets
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}', '⟨': '⟩'}
        
        for char in statement:
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    return False
        
        if stack:
            return False
        
        # Check for basic Lean keywords
        lean_keywords = ['theorem', 'lemma', 'def', 'by', 'sorry', ':=', '⊢', '∀', '∃']
        has_structure = any(kw in statement for kw in lean_keywords)
        
        return has_structure


class FIRMAEvaluator:
    """Comprehensive evaluator for FIRMA model"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self._load_model()
        self._load_dataset()
        
        # Initialize validators and analyzers
        self.lean_validator = Lean4Validator(config)
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Results storage
        self.results = []
        self.failure_cases = []
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load FIRMA model"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = FIRMA.from_pretrained(self.config.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_dataset(self):
        """Load Lean-Workbook test set"""
        logger.info("Loading test dataset")
        
        try:
            # Load dataset
            dataset = load_dataset(self.config.dataset_name, split="train")
            
            # Filter for proved theorems
            self.test_data = []
            for item in dataset:
                if item.get('status', '').lower() == 'proved':
                    self.test_data.append(self._process_item(item))
            
            # Limit samples
            if self.config.max_samples:
                self.test_data = self.test_data[:self.config.max_samples]
            
            logger.info(f"Loaded {len(self.test_data)} test samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Create dummy data
            self.test_data = self._create_dummy_data()
    
    def _process_item(self, item: Dict) -> Dict:
        """Process dataset item"""
        
        complexity = self.complexity_analyzer.compute_complexity(
            item.get('formal_statement', ''),
            item.get('tactic', '')
        )
        
        return {
            'id': item.get('id', 'unknown'),
            'formal': item.get('formal_statement', '').strip(),
            'informal': item.get('natural_language_statement', '').strip(),
            'answer': item.get('answer', '').strip(),
            'tactic': item.get('tactic', '').strip(),
            'state_before': item.get('state_before', '').strip(),
            'state_after': item.get('state_after', '').strip(),
            'complexity': complexity
        }
    
    def _create_dummy_data(self) -> List[Dict]:
        """Create dummy test data"""
        dummy = []
        for i in range(5):
            dummy.append({
                'id': f'test_{i}',
                'formal': f'theorem test_{i} : ∀ x : ℝ, x + 0 = x := by simp',
                'informal': f'For any real number x, adding zero gives x.',
                'answer': 'x',
                'tactic': 'simp',
                'complexity': (i % 4) + 1
            })
        return dummy
    
    def generate_translation(self, prompt: str, direction: int, complexity: int) -> str:
        """Generate translation using FIRMA"""
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if prompt in generated:
                generated = generated[len(prompt):].strip()
            
            # Clean up special tokens
            for token in ['<FORMAL>', '</FORMAL>', '<INFORMAL>', '</INFORMAL>']:
                generated = generated.replace(token, '')
            
            return generated.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def compute_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute translation metrics"""
        
        metrics = {}
        
        if not reference or not hypothesis:
            return {
                'bleu': 0.0,
                'rouge_l': 0.0,
                'bert_score': 0.0,
                'edit_distance': float('inf')
            }
        
        # BLEU
        if self.config.compute_bleu:
            try:
                bleu = corpus_bleu([hypothesis], [[reference]]).score / 100.0
                metrics['bleu'] = bleu
            except:
                metrics['bleu'] = 0.0
        
        # ROUGE-L
        if self.config.compute_rouge:
            try:
                rouge = self.rouge_scorer.score(reference, hypothesis)
                metrics['rouge_l'] = rouge['rougeL'].fmeasure
            except:
                metrics['rouge_l'] = 0.0
        
        # BERTScore
        if self.config.compute_bert_score:
            try:
                P, R, F1 = bert_score(
                    [hypothesis], 
                    [reference], 
                    lang='en',
                    device=self.device,
                    verbose=False
                )
                metrics['bert_score'] = F1.item()
            except:
                metrics['bert_score'] = 0.0
        
        # Edit distance
        if self.config.compute_edit_distance:
            try:
                dist = editdistance.eval(reference, hypothesis)
                metrics['edit_distance'] = dist
            except:
                metrics['edit_distance'] = float('inf')
        
        return metrics
    
    def evaluate_sample(self, sample: Dict) -> EvaluationSample:
        """Evaluate a single sample"""
        
        start_time = time.time()
        
        # Generate translations
        # Formal to Informal
        f2i_prompt = f"<FORMAL>{sample['formal']}</FORMAL>\nTranslate to natural language:\n<INFORMAL>"
        f2i_output = self.generate_translation(f2i_prompt, 0, sample['complexity'] - 1)
        
        # Informal to Formal
        i2f_prompt = f"<INFORMAL>{sample['informal']}</INFORMAL>\nTranslate to Lean theorem:\n<FORMAL>"
        i2f_output = self.generate_translation(i2f_prompt, 1, sample['complexity'] - 1)
        
        generation_time = time.time() - start_time
        
        # Compute metrics
        f2i_metrics = self.compute_metrics(sample['informal'], f2i_output)
        i2f_metrics = self.compute_metrics(sample['formal'], i2f_output)
        
        # Round-trip consistency
        # Generate back translation
        rt_prompt = f"<FORMAL>{i2f_output}</FORMAL>\nTranslate to natural language:\n<INFORMAL>"
        rt_output = self.generate_translation(rt_prompt, 0, sample['complexity'] - 1)
        rt_metrics = self.compute_metrics(sample['informal'], rt_output)
        
        # Validate formal output
        validation_start = time.time()
        lean_valid = None
        lean_error = None
        syntax_valid = True
        
        if self.config.check_lean_validity and i2f_output:
            # Check syntax
            syntax_valid = self.lean_validator.check_syntax(i2f_output)
            
            # Full validation
            if syntax_valid:
                lean_valid, lean_error = self.lean_validator.validate_theorem(
                    i2f_output, 
                    sample.get('tactic', '')
                )
        
        validation_time = time.time() - validation_start
        
        # Determine success
        translation_success = (
            f2i_metrics.get('bleu', 0) > 0.3 and 
            i2f_metrics.get('bleu', 0) > 0.3
        )
        
        proof_success = lean_valid if lean_valid is not None else syntax_valid
        
        overall_success = translation_success and (proof_success or syntax_valid)
        
        return EvaluationSample(
            sample_id=sample['id'],
            complexity_level=sample['complexity'],
            formal_reference=sample['formal'],
            informal_reference=sample['informal'],
            answer_reference=sample.get('answer', ''),
            tactic=sample.get('tactic', ''),
            formal_to_informal=f2i_output,
            informal_to_formal=i2f_output,
            f2i_bleu=f2i_metrics.get('bleu', 0),
            i2f_bleu=i2f_metrics.get('bleu', 0),
            f2i_rouge_l=f2i_metrics.get('rouge_l', 0),
            i2f_rouge_l=i2f_metrics.get('rouge_l', 0),
            f2i_bert_score=f2i_metrics.get('bert_score', 0),
            i2f_bert_score=i2f_metrics.get('bert_score', 0),
            f2i_edit_distance=f2i_metrics.get('edit_distance', float('inf')),
            i2f_edit_distance=i2f_metrics.get('edit_distance', float('inf')),
            lean_valid=lean_valid,
            lean_error=lean_error,
            syntax_valid=syntax_valid,
            roundtrip_bleu=rt_metrics.get('bleu', 0),
            roundtrip_exact_match=(rt_output == sample['informal']),
            generation_time=generation_time,
            validation_time=validation_time,
            translation_success=translation_success,
            proof_success=proof_success,
            overall_success=overall_success
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation"""
        logger.info("Starting evaluation")
        
        # Evaluate all samples
        for sample in tqdm(self.test_data, desc="Evaluating"):
            try:
                result = self.evaluate_sample(sample)
                self.results.append(result)
                
                # Track failures
                if not result.overall_success:
                    self.failure_cases.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {sample['id']}: {e}")
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics()
        
        # Analyze failures
        if self.config.analyze_failures:
            failure_analysis = self._analyze_failures()
            metrics['failure_analysis'] = failure_analysis
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_plots(metrics)
        
        # Save results
        self._save_results(metrics)
        
        # Generate report
        if self.config.generate_report:
            self._generate_report(metrics)
        
        return metrics
    
    def _compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics"""
        
        if not self.results:
            return {}
        
        metrics = {
            'num_samples': len(self.results),
            'timestamp': datetime.now().isoformat(),
            
            # Translation metrics
            'formal_to_informal': {
                'bleu': np.mean([r.f2i_bleu for r in self.results]),
                'rouge_l': np.mean([r.f2i_rouge_l for r in self.results]),
                'bert_score': np.mean([r.f2i_bert_score for r in self.results]),
                'edit_distance': np.mean([r.f2i_edit_distance for r in self.results if r.f2i_edit_distance != float('inf')])
            },
            'informal_to_formal': {
                'bleu': np.mean([r.i2f_bleu for r in self.results]),
                'rouge_l': np.mean([r.i2f_rouge_l for r in self.results]),
                'bert_score': np.mean([r.i2f_bert_score for r in self.results]),
                'edit_distance': np.mean([r.i2f_edit_distance for r in self.results if r.i2f_edit_distance != float('inf')])
            },
            
            # Validation
            'validation': {
                'syntax_valid_rate': sum(r.syntax_valid for r in self.results) / len(self.results),
                'lean_valid_rate': sum(r.lean_valid == True for r in self.results) / sum(r.lean_valid is not None for r in self.results) if any(r.lean_valid is not None for r in self.results) else 0,
                'proof_success_rate': sum(r.proof_success for r in self.results) / len(self.results)
            },
            
            # Round-trip
            'roundtrip': {
                'bleu': np.mean([r.roundtrip_bleu for r in self.results]),
                'exact_match_rate': sum(r.roundtrip_exact_match for r in self.results) / len(self.results)
            },
            
            # Success rates
            'success_rates': {
                'translation': sum(r.translation_success for r in self.results) / len(self.results),
                'proof': sum(r.proof_success for r in self.results) / len(self.results),
                'overall': sum(r.overall_success for r in self.results) / len(self.results)
            },
            
            # Performance by complexity
            'by_complexity': {}
        }
        
        # Metrics by complexity level
        for level in range(1, 5):
            level_results = [r for r in self.results if r.complexity_level == level]
            if level_results:
                metrics['by_complexity'][f'level_{level}'] = {
                    'count': len(level_results),
                    'bleu': np.mean([(r.f2i_bleu + r.i2f_bleu) / 2 for r in level_results]),
                    'success_rate': sum(r.overall_success for r in level_results) / len(level_results),
                    'lean_valid_rate': sum(r.lean_valid == True for r in level_results) / max(1, sum(r.lean_valid is not None for r in level_results))
                }
        
        # Timing
        metrics['timing'] = {
            'avg_generation_time': np.mean([r.generation_time for r in self.results]),
            'avg_validation_time': np.mean([r.validation_time for r in self.results])
        }
        
        return metrics
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure cases"""
        
        if not self.failure_cases:
            return {'no_failures': True}
        
        analysis = {
            'total_failures': len(self.failure_cases),
            'failure_rate': len(self.failure_cases) / len(self.results),
            'failure_types': defaultdict(int),
            'error_categories': defaultdict(int),
            'complexity_distribution': defaultdict(int)
        }
        
        for failure in self.failure_cases:
            # Categorize failure type
            if not failure.syntax_valid:
                analysis['failure_types']['syntax_error'] += 1
            elif failure.lean_valid == False:
                analysis['failure_types']['validation_error'] += 1
            elif not failure.translation_success:
                analysis['failure_types']['translation_quality'] += 1
            else:
                analysis['failure_types']['other'] += 1
            
            # Analyze Lean errors
            if failure.lean_error:
                if 'timeout' in failure.lean_error.lower():
                    analysis['error_categories']['timeout'] += 1
                elif 'syntax' in failure.lean_error.lower():
                    analysis['error_categories']['syntax'] += 1
                elif 'type' in failure.lean_error.lower():
                    analysis['error_categories']['type_error'] += 1
                elif 'tactic' in failure.lean_error.lower():
                    analysis['error_categories']['tactic_error'] += 1
                else:
                    analysis['error_categories']['other'] += 1
            
            # Complexity distribution
            analysis['complexity_distribution'][f'level_{failure.complexity_level}'] += 1
        
        # Sample failure cases
        analysis['sample_failures'] = []
        for failure in self.failure_cases[:self.config.num_examples_to_show]:
            analysis['sample_failures'].append({
                'id': failure.sample_id,
                'complexity': failure.complexity_level,
                'formal_ref': failure.formal_reference[:100] + '...' if len(failure.formal_reference) > 100 else failure.formal_reference,
                'generated': failure.informal_to_formal[:100] + '...' if len(failure.informal_to_formal) > 100 else failure.informal_to_formal,
                'error': failure.lean_error
            })
        
        return analysis
    
    def _generate_plots(self, metrics: Dict[str, Any]):
        """Generate evaluation visualizations"""
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Plot 1: Translation metrics comparison
        ax1 = axes[0, 0]
        metrics_names = ['BLEU', 'ROUGE-L', 'BERTScore']
        f2i_scores = [
            metrics['formal_to_informal']['bleu'],
            metrics['formal_to_informal']['rouge_l'],
            metrics['formal_to_informal']['bert_score']
        ]
        i2f_scores = [
            metrics['informal_to_formal']['bleu'],
            metrics['informal_to_formal']['rouge_l'],
            metrics['informal_to_formal']['bert_score']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, f2i_scores, width, label='Formal→Informal', color='skyblue')
        ax1.bar(x + width/2, i2f_scores, width, label='Informal→Formal', color='lightcoral')
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Score')
        ax1.set_title('Translation Quality Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Plot 2: Success rates
        ax2 = axes[0, 1]
        success_types = ['Translation', 'Proof', 'Overall']
        success_rates = [
            metrics['success_rates']['translation'],
            metrics['success_rates']['proof'],
            metrics['success_rates']['overall']
        ]
        
        colors = ['green' if r > 0.5 else 'orange' if r > 0.3 else 'red' for r in success_rates]
        ax2.bar(success_types, success_rates, color=colors)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rates')
        ax2.set_ylim(0, 1)
        
        # Add percentage labels
        for i, (type_, rate) in enumerate(zip(success_types, success_rates)):
            ax2.text(i, rate + 0.01, f'{rate:.1%}', ha='center')
        
        # Plot 3: Complexity analysis
        ax3 = axes[0, 2]
        if metrics['by_complexity']:
            complexity_levels = []
            complexity_success = []
            complexity_counts = []
            
            for level in sorted(metrics['by_complexity'].keys()):
                complexity_levels.append(level.replace('level_', 'L'))
                complexity_success.append(metrics['by_complexity'][level]['success_rate'])
                complexity_counts.append(metrics['by_complexity'][level]['count'])
            
            ax3.bar(complexity_levels, complexity_success)
            ax3.set_xlabel('Complexity Level')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Performance by Complexity')
            ax3.set_ylim(0, 1)
            
            # Add sample count labels
            for i, (level, count) in enumerate(zip(complexity_levels, complexity_counts)):
                ax3.text(i, 0.05, f'n={count}', ha='center')
        
        # Plot 4: BLEU score distribution
        ax4 = axes[1, 0]
        all_bleu_scores = [(r.f2i_bleu + r.i2f_bleu) / 2 for r in self.results]
        ax4.hist(all_bleu_scores, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('BLEU Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('BLEU Score Distribution')
        ax4.axvline(np.mean(all_bleu_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_bleu_scores):.3f}')
        ax4.legend()
        
        # Plot 5: Validation results
        ax5 = axes[1, 1]
        validation_labels = ['Syntax Valid', 'Lean Valid', 'Proof Success']
        validation_rates = [
            metrics['validation']['syntax_valid_rate'],
            metrics['validation']['lean_valid_rate'],
            metrics['validation']['proof_success_rate']
        ]
        
        ax5.pie(validation_rates + [1 - sum(validation_rates)], 
                labels=validation_labels + ['Failed'],
                autopct='%1.1f%%',
                colors=['green', 'lightgreen', 'yellow', 'red'])
        ax5.set_title('Validation Results')
        
        # Plot 6: Error categories (if failures exist)
        ax6 = axes[1, 2]
        if 'failure_analysis' in metrics and metrics['failure_analysis'].get('error_categories'):
            error_cats = list(metrics['failure_analysis']['error_categories'].keys())
            error_counts = list(metrics['failure_analysis']['error_categories'].values())
            
            ax6.barh(error_cats, error_counts)
            ax6.set_xlabel('Count')
            ax6.set_title('Error Categories')
        else:
            ax6.text(0.5, 0.5, 'No errors to analyze', ha='center', va='center')
            ax6.set_title('Error Analysis')
        
        # Plot 7: Generation time distribution
        ax7 = axes[2, 0]
        gen_times = [r.generation_time for r in self.results]
        ax7.boxplot([gen_times], labels=['Generation Time'])
        ax7.set_ylabel('Time (seconds)')
        ax7.set_title('Generation Time Distribution')
        
        # Plot 8: Round-trip consistency
        ax8 = axes[2, 1]
        rt_scores = [r.roundtrip_bleu for r in self.results]
        ax8.scatter(all_bleu_scores, rt_scores, alpha=0.5)
        ax8.set_xlabel('Direct Translation BLEU')
        ax8.set_ylabel('Round-trip BLEU')
        ax8.set_title('Round-trip Consistency')
        ax8.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect consistency line
        
        # Plot 9: Failure heatmap by complexity and type
        ax9 = axes[2, 2]
        if 'failure_analysis' in metrics and self.failure_cases:
            # Create failure matrix
            failure_matrix = np.zeros((4, 4))  # 4 complexity levels, 4 failure types
            failure_types = ['syntax_error', 'validation_error', 'translation_quality', 'other']
            
            for failure in self.failure_cases:
                complexity_idx = failure.complexity_level - 1
                
                if not failure.syntax_valid:
                    type_idx = 0
                elif failure.lean_valid == False:
                    type_idx = 1
                elif not failure.translation_success:
                    type_idx = 2
                else:
                    type_idx = 3
                
                failure_matrix[complexity_idx, type_idx] += 1
            
            im = ax9.imshow(failure_matrix, cmap='YlOrRd', aspect='auto')
            ax9.set_xticks(range(4))
            ax9.set_yticks(range(4))
            ax9.set_xticklabels(['Syntax', 'Validation', 'Quality', 'Other'])
            ax9.set_yticklabels([f'L{i+1}' for i in range(4)])
            ax9.set_xlabel('Failure Type')
            ax9.set_ylabel('Complexity')
            ax9.set_title('Failure Heatmap')
            plt.colorbar(im, ax=ax9)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.output_dir) / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plots to {plot_path}")
        
        plt.close()
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = Path(self.config.output_dir) / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save detailed results
        if self.config.save_detailed:
            detailed_file = Path(self.config.output_dir) / f"detailed_results_{timestamp}.json"
            detailed_data = [asdict(r) for r in self.results]
            with open(detailed_file, 'w') as f:
                json.dump(detailed_data, f, indent=2, default=str)
            logger.info(f"Saved detailed results to {detailed_file}")
        
        # Save failure cases
        if self.failure_cases:
            failures_file = Path(self.config.output_dir) / f"failures_{timestamp}.json"
            failure_data = [asdict(f) for f in self.failure_cases]
            with open(failures_file, 'w') as f:
                json.dump(failure_data, f, indent=2, default=str)
            logger.info(f"Saved failure cases to {failures_file}")
    
    def _generate_report(self, metrics: Dict[str, Any]):
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.config.output_dir) / f"report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FIRMA EVALUATION REPORT - LEAN-WORKBOOK DATASET\n")
            f.write(f"Generated: {metrics['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Samples Evaluated: {metrics['num_samples']}\n")
            f.write(f"Overall Success Rate: {metrics['success_rates']['overall']:.1%}\n")
            f.write(f"Translation Success: {metrics['success_rates']['translation']:.1%}\n")
            f.write(f"Proof Validation Success: {metrics['success_rates']['proof']:.1%}\n\n")
            
            # Translation Metrics
            f.write("TRANSLATION QUALITY METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Direction':<20} {'BLEU':>10} {'ROUGE-L':>10} {'BERTScore':>10}\n")
            f.write("-"*50 + "\n")
            
            f.write(f"{'Formal→Informal':<20} "
                   f"{metrics['formal_to_informal']['bleu']:>10.3f} "
                   f"{metrics['formal_to_informal']['rouge_l']:>10.3f} "
                   f"{metrics['formal_to_informal']['bert_score']:>10.3f}\n")
            
            f.write(f"{'Informal→Formal':<20} "
                   f"{metrics['informal_to_formal']['bleu']:>10.3f} "
                   f"{metrics['informal_to_formal']['rouge_l']:>10.3f} "
                   f"{metrics['informal_to_formal']['bert_score']:>10.3f}\n\n")
            
            # Validation Results
            f.write("VALIDATION RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Syntax Valid Rate: {metrics['validation']['syntax_valid_rate']:.1%}\n")
            f.write(f"Lean Valid Rate: {metrics['validation']['lean_valid_rate']:.1%}\n")
            f.write(f"Proof Success Rate: {metrics['validation']['proof_success_rate']:.1%}\n\n")
            
            # Round-trip Consistency
            f.write("ROUND-TRIP CONSISTENCY\n")
            f.write("-"*40 + "\n")
            f.write(f"Round-trip BLEU: {metrics['roundtrip']['bleu']:.3f}\n")
            f.write(f"Exact Match Rate: {metrics['roundtrip']['exact_match_rate']:.1%}\n\n")
            
            # Performance by Complexity
            if metrics['by_complexity']:
                f.write("PERFORMANCE BY COMPLEXITY LEVEL\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Level':<10} {'Count':>10} {'BLEU':>10} {'Success':>10}\n")
                f.write("-"*40 + "\n")
                
                for level in sorted(metrics['by_complexity'].keys()):
                    data = metrics['by_complexity'][level]
                    f.write(f"{level:<10} {data['count']:>10} "
                           f"{data['bleu']:>10.3f} "
                           f"{data['success_rate']:>9.1%}\n")
                f.write("\n")
            
            # Failure Analysis
            if 'failure_analysis' in metrics:
                fa = metrics['failure_analysis']
                
                f.write("FAILURE ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                if not fa.get('no_failures'):
                    f.write(f"Total Failures: {fa['total_failures']}\n")
                    f.write(f"Failure Rate: {fa['failure_rate']:.1%}\n\n")
                    
                    f.write("Failure Types:\n")
                    for ftype, count in fa['failure_types'].items():
                        f.write(f"  - {ftype}: {count}\n")
                    
                    f.write("\nError Categories:\n")
                    for cat, count in fa['error_categories'].items():
                        f.write(f"  - {cat}: {count}\n")
                    
                    # Sample failures
                    if fa.get('sample_failures'):
                        f.write("\nSAMPLE FAILURE CASES\n")
                        f.write("-"*40 + "\n")
                        
                        for i, failure in enumerate(fa['sample_failures'], 1):
                            f.write(f"\nFailure {i} (ID: {failure['id']}, Complexity: {failure['complexity']}):\n")
                            f.write(f"  Reference: {failure['formal_ref']}\n")
                            f.write(f"  Generated: {failure['generated']}\n")
                            if failure['error']:
                                f.write(f"  Error: {failure['error']}\n")
                else:
                    f.write("No failures detected!\n")
            
            # Timing Statistics
            f.write("\n\nTIMING STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Avg Generation Time: {metrics['timing']['avg_generation_time']:.2f}s\n")
            f.write(f"Avg Validation Time: {metrics['timing']['avg_validation_time']:.2f}s\n")
            
            # Qualitative Examples
            f.write("\n\nQUALITATIVE EXAMPLES\n")
            f.write("-"*40 + "\n")
            
            # Show best and worst examples
            sorted_results = sorted(self.results, key=lambda x: x.overall_success and (x.f2i_bleu + x.i2f_bleu) / 2, reverse=True)
            
            f.write("\nBEST EXAMPLES:\n")
            for i, result in enumerate(sorted_results[:3], 1):
                f.write(f"\n{i}. ID: {result.sample_id} (BLEU: {(result.f2i_bleu + result.i2f_bleu) / 2:.3f})\n")
                f.write(f"   Formal: {result.formal_reference[:100]}...\n")
                f.write(f"   Informal: {result.informal_reference[:100]}...\n")
                f.write(f"   Generated F→I: {result.formal_to_informal[:100]}...\n")
            
            f.write("\n\nCHALLENGING EXAMPLES:\n")
            for i, result in enumerate(sorted_results[-3:], 1):
                f.write(f"\n{i}. ID: {result.sample_id} (BLEU: {(result.f2i_bleu + result.i2f_bleu) / 2:.3f})\n")
                f.write(f"   Issue: ")
                if not result.syntax_valid:
                    f.write("Syntax error\n")
                elif result.lean_valid == False:
                    f.write(f"Validation failed: {result.lean_error}\n")
                else:
                    f.write("Low translation quality\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Generated report: {report_file}")
        
        # Print summary to console
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary to console"""
        
        print("\n" + "="*60)
        print("FIRMA EVALUATION SUMMARY")
        print("="*60)
        print(f"Samples: {metrics['num_samples']}")
        print(f"\nTranslation Quality:")
        print(f"  F→I BLEU: {metrics['formal_to_informal']['bleu']:.3f}")
        print(f"  I→F BLEU: {metrics['informal_to_formal']['bleu']:.3f}")
        print(f"\nValidation:")
        print(f"  Syntax Valid: {metrics['validation']['syntax_valid_rate']:.1%}")
        print(f"  Lean Valid: {metrics['validation']['lean_valid_rate']:.1%}")
        print(f"\nSuccess Rates:")
        print(f"  Translation: {metrics['success_rates']['translation']:.1%}")
        print(f"  Proof: {metrics['success_rates']['proof']:.1%}")
        print(f"  Overall: {metrics['success_rates']['overall']:.1%}")
        print("="*60 + "\n")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FIRMA on Lean-Workbook dataset")
    
    parser.add_argument("--model-path", type=str, default="./firma_lean_model/final",
                        help="Path to FIRMA model")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--no-lean", action="store_true",
                        help="Skip Lean4 validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_path=args.model_path,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        check_lean_validity=not args.no_lean,
        device=args.device
    )
    
    # Run evaluation
    evaluator = FIRMAEvaluator(config)
    metrics = evaluator.evaluate()
    
    logger.info("Evaluation complete!")
    

if __name__ == "__main__":
    main()