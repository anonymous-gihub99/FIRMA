#!/usr/bin/env python3
"""
train_firma_lean.py - Training script for FIRMA on Lean-Workbook dataset
OPTIMIZED VERSION for faster training and reduced memory usage
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import gc

from firma_model_lean import FIRMA, FIRMAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration for Lean-Workbook dataset - OPTIMIZED"""
    
    # Dataset - REDUCED for faster training
    dataset_name: str = "Qwen/Qwen3-0.6B"
    max_samples: Optional[int] = 5000
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    
    # Training - OPTIMIZED FOR MEMORY
    output_dir: str = "./firma_lean_model"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Optimization
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduling
    lr_scheduler_type: str = "cosine"
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging
    logging_steps: int = 25
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Hardware - OPTIMIZED
    fp16: bool = False
    bf16: bool = True if torch.cuda.is_bf16_supported() else False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    
    # Seeds
    seed: int = 42
    
    # Paths
    cache_dir: str = "./cache"
    
    # Progressive training
    progressive_training: bool = False
    complexity_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])


class ComplexityAnalyzer:
    """Analyze theorem complexity for progressive training"""
    
    def __init__(self):
        self.complexity_indicators = {
            1: {
                'keywords': ['=', '≠', '+', '-', '*', '/', '<', '>', '≤', '≥'],
                'tactics': ['rfl', 'simp', 'norm_num', 'ring'],
                'max_depth': 2
            },
            2: {
                'keywords': ['∀', '∃', '∈', '∉', '⊂', '⊃', '∩', '∪', 'iff', '↔'],
                'tactics': ['intro', 'apply', 'exact', 'constructor', 'cases'],
                'max_depth': 5
            },
            3: {
                'keywords': ['∧', '∨', '¬', '→', '⇒', '⇔', 'lim', '∫', '∂'],
                'tactics': ['induction', 'rw', 'have', 'suffices', 'obtain'],
                'max_depth': 10
            },
            4: {
                'keywords': ['⊢', '⊨', '∏', '∑', 'Hom', 'Ext', '⊗', '⊕'],
                'tactics': ['tactic', 'meta', 'conv', 'calc', 'gcongr'],
                'max_depth': float('inf')
            }
        }
    
    def compute_complexity(self, formal_statement: str, tactic: str = None) -> int:
        if not formal_statement:
            return 1
        
        max_level = 1
        
        for level, indicators in self.complexity_indicators.items():
            if any(keyword in formal_statement for keyword in indicators['keywords']):
                max_level = max(max_level, level)
        
        if tactic:
            tactic_lower = tactic.lower()
            for level, indicators in self.complexity_indicators.items():
                if any(t in tactic_lower for t in indicators['tactics']):
                    max_level = max(max_level, level)
        
        if len(formal_statement) > 500:
            max_level = max(max_level, 3)
        if formal_statement.count('∀') + formal_statement.count('∃') > 3:
            max_level = max(max_level, 3)
        
        return min(max_level, 4)


class LeanWorkbookDataset(Dataset):
    """Dataset for Lean-Workbook - OPTIMIZED"""
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: int = 512,
        complexity_filter: Optional[int] = None,
        cache_encodings: bool = True
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.complexity_analyzer = ComplexityAnalyzer()
        self.cache_encodings = cache_encodings
        self.encoding_cache = {}
        
        logger.info(f"Loading Lean-Workbook dataset (split: {split})")
        
        try:
            dataset = load_dataset("internlm/Lean-Workbook", split="train")
            
            self.data = []
            proved_items = [item for item in dataset if item.get('status', '').lower() == 'proved']
            
            if max_samples and len(proved_items) > max_samples:
                np.random.seed(42)
                indices = np.random.choice(len(proved_items), max_samples, replace=False)
                proved_items = [proved_items[i] for i in indices]
            
            for item in tqdm(proved_items, desc="Processing dataset"):
                processed_item = self._process_item(item)
                
                if complexity_filter is None or processed_item['complexity'] <= complexity_filter:
                    self.data.append(processed_item)
            
            self.data = self._split_data(split)
            
            logger.info(f"Loaded {len(self.data)} samples for {split} split")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            self.data = self._create_dummy_data()
    
    def _process_item(self, item: Dict) -> Dict:
        theorem_id = item.get('id', 'unknown')
        formal_statement = item.get('formal_statement', '')
        natural_language = item.get('natural_language_statement', '')
        answer = item.get('answer', '')
        tactic = item.get('tactic', '')
        state_before = item.get('state_before', '')
        state_after = item.get('state_after', '')
        
        complexity = self.complexity_analyzer.compute_complexity(formal_statement, tactic)
        
        return {
            'id': theorem_id,
            'formal': formal_statement.strip(),
            'informal': natural_language.strip(),
            'answer': answer.strip(),
            'tactic': tactic.strip(),
            'state_before': state_before.strip(),
            'state_after': state_after.strip(),
            'complexity': complexity - 1,
            'proof_valid': 1
        }
    
    def _split_data(self, split: str) -> List[Dict]:
        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        
        n_train = int(0.9 * len(indices))
        n_val = int(0.05 * len(indices))
        
        if split == "train":
            split_indices = indices[:n_train]
        elif split == "val":
            split_indices = indices[n_train:n_train + n_val]
        else:
            split_indices = indices[n_train + n_val:]
        
        return [self.data[i] for i in split_indices]
    
    def _create_dummy_data(self) -> List[Dict]:
        logger.warning("Creating dummy data for testing")
        
        dummy_data = []
        for i in range(10):
            dummy_data.append({
                'id': f'dummy_{i}',
                'formal': f'theorem test_{i} (x : ℝ) : x + 0 = x := by simp',
                'informal': f'Adding zero to any real number x gives x itself.',
                'answer': 'x',
                'tactic': 'simp',
                'state_before': 'x : ℝ ⊢ x + 0 = x',
                'state_after': 'no goals',
                'complexity': i % 4,
                'proof_valid': 1
            })
        
        return dummy_data
    
    def __len__(self):
        return len(self.data) * 2
    
    def __getitem__(self, idx):
        direction = idx % 2
        data_idx = (idx // 2) % len(self.data)
        
        cache_key = f"{data_idx}_{direction}"
        if self.cache_encodings and cache_key in self.encoding_cache:
            return self.encoding_cache[cache_key]
        
        item = self.data[data_idx]
        
        if direction == 0:
            prompt = f"<FORMAL>{item['formal']}</FORMAL>\nTranslate to natural language:\n<INFORMAL>"
            target = f"{item['informal']}</INFORMAL>"
            formality = 0.2
        else:
            prompt = f"<INFORMAL>{item['informal']}</INFORMAL>\nTranslate to Lean theorem:\n<FORMAL>"
            target = f"{item['formal']}</FORMAL>"
            formality = 0.8
        
        full_text = prompt + target
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length
        )
        
        labels = encoding['input_ids'].clone()
        labels[0, :len(prompt_tokens['input_ids'])] = -100
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'direction': torch.tensor(direction, dtype=torch.long),
            'complexity': torch.tensor(item['complexity'], dtype=torch.long),
            'formality': torch.tensor(formality, dtype=torch.float),
            'proof_valid': torch.tensor(item['proof_valid'], dtype=torch.long)
        }
        
        if self.cache_encodings:
            self.encoding_cache[cache_key] = result
        
        return result


class FIRMATrainer:
    """Trainer for FIRMA model - OPTIMIZED"""
    
    def __init__(self, model_config: FIRMAConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(training_config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_tokenizer()
        self._setup_model()
        self._setup_datasets()
    
    def _setup_tokenizer(self):
        logger.info(f"Setting up tokenizer for {self.model_config.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=True,
            cache_dir=self.training_config.cache_dir
        )
        
        special_tokens = {
            'additional_special_tokens': self.model_config.lean_special_tokens
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        logger.info("Initializing FIRMA model")
        
        self.model = FIRMA(self.model_config)
        self.model.base_model.resize_token_embeddings(len(self.tokenizer))
        
        #
        # THE FIX: This block is removed. device_map="auto" handles GPU placement.
        #
        # if self.model_config.device == "cuda":
        #     self.model = self.model.cuda()
        
        logger.info("Model initialized successfully")
    
    def _setup_datasets(self):
        logger.info("Setting up datasets")
        
        self.train_dataset = LeanWorkbookDataset(
            self.tokenizer,
            split="train",
            max_samples=self.training_config.max_samples,
            max_length=self.model_config.max_length,
            cache_encodings=True
        )
        
        self.val_dataset = LeanWorkbookDataset(
            self.tokenizer,
            split="val",
            max_samples=500,
            max_length=self.model_config.max_length,
            cache_encodings=True
        )
        
        self.test_dataset = LeanWorkbookDataset(
            self.tokenizer,
            split="test",
            max_samples=500,
            max_length=self.model_config.max_length,
            cache_encodings=True
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train(self):
        """Train FIRMA model"""
        
        self._train_standard()
        self._save_model("final")
        
        logger.info("Training complete!")
    
    def _train_standard(self):
        """Standard training - OPTIMIZED"""
        logger.info("Starting optimized training")
        
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            logging_steps=self.training_config.logging_steps,
            logging_first_step=self.training_config.logging_first_step,
            report_to=self.training_config.report_to,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            seed=self.training_config.seed,
            remove_unused_columns=False,
            label_names=['labels']
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        self._save_model("best")
    
    def _save_model(self, checkpoint_name: str):
        output_path = Path(self.training_config.output_dir) / checkpoint_name
        
        logger.info(f"Saving model to {output_path}")
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info(f"Model saved successfully")


def main():
    """Main training function - OPTIMIZED"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FIRMA on Lean-Workbook (OPTIMIZED)")
    
    parser.add_argument("--base-model", type=str, default="qwen/qwen3-0.6B")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./firma_lean_model")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=5000)
    
    # THE FIX: Add a proper boolean flag for quantization
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    
    model_config = FIRMAConfig(
        base_model=args.base_model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
        gradient_checkpointing=True,
        use_4bit=args.use_4bit  # THE FIX: Pass the command-line argument
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        progressive_training=False,
        gradient_checkpointing=True
    )
    
    trainer = FIRMATrainer(model_config, training_config)
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()