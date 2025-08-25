#!/usr/bin/env python3
"""
firma_model_t4_fixed.py - FIRMA model with DDP fixes
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import gc
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('firma_t4_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FIRMAConfig:
    """Configuration optimized for 2x T4 GPUs (30GB total)"""
    
    # Model - smaller for T4
    base_model: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    hidden_dim: int = 512
    num_complexity_levels: int = 4
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Training parameters - optimized for T4 memory
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 8
    max_length: int = 256
    eval_batch_size: int = 4
    
    # Loss weights
    lambda_translation: float = 0.5
    lambda_roundtrip: float = 0.2
    lambda_complexity: float = 0.15
    lambda_validity: float = 0.15
    
    # QLoRA settings - optimized for T4
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Paths
    output_dir: str = "./firma_model_t4"
    data_dir: str = "./FIRMA/math_alignment_dataset"
    
    # Training settings
    progressive_training: bool = True
    gradient_checkpointing: bool = True
    complexity_schedule: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    # Hardware
    fp16: bool = True
    save_total_limit: int = 2
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 100
    
    # DDP fixes
    use_static_graph: bool = True  # NEW: Enable static graph
    ddp_find_unused_parameters: bool = False  # NEW: Disable unused parameter detection

class ComplexityAnalyzer:
    """Analyze mathematical statement complexity"""
    
    def __init__(self):
        self.complexity_indicators = {
            1: ['=', '+', '-', 'equals', 'plus', 'minus', 'sum'],
            2: ['∀', '∃', '∈', 'for all', 'exists', 'in', 'element'],
            3: ['∧', '∨', '→', '⇒', 'and', 'or', 'implies', 'if'],
            4: ['⊢', '⊨', '∫', '∂', 'proves', 'models', 'integral', 'derivative']
        }
    
    def compute_complexity(self, text: str) -> int:
        """Compute complexity level (1-4)"""
        if not text:
            return 1
        max_level = 1
        for level, indicators in self.complexity_indicators.items():
            if any(ind in text.lower() for ind in indicators):
                max_level = max(max_level, level)
        return min(max_level, 4)

class FIRMADataset(Dataset):
    """Optimized dataset for T4 GPUs"""
    
    def __init__(self, data_path: str, tokenizer, config: FIRMAConfig, 
                 split: str = "train", max_complexity: Optional[int] = None):
        self.data = []
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.max_complexity = max_complexity
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Load data
        if Path(data_path).exists():
            logger.info(f"Loading {split} dataset from {data_path}")
            self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} samples for {split}")
    
    def _load_data(self, data_path: str):
        """Load and preprocess data"""
        try:
            if data_path.endswith('.jsonl'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                self._add_item(item)
                            except:
                                continue
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[:800]:
                            self._add_item(item)
        except Exception as e:
            logger.error(f"Error loading {data_path}: {e}")
    
    def _add_item(self, item):
        """Add preprocessed item"""
        if not isinstance(item, dict):
            return
        
        # Handle various key formats
        formal = None
        informal = None
        
        for key in ['formal_statement', 'formal']:
            if key in item and item[key]:
                formal = str(item[key])[:600]
                break
        
        for key in ['informal_stmt', 'informal_statement', 'informal']:
            if key in item and item[key]:
                informal = str(item[key])[:600]
                break
        
        if formal and informal:
            complexity = self.complexity_analyzer.compute_complexity(formal + ' ' + informal)
            
            if self.max_complexity and complexity > self.max_complexity:
                return
            
            self.data.append({
                'formal': formal,
                'informal': informal,
                'complexity': complexity
            })
    
    def __len__(self):
        return len(self.data) * 2 if self.data else 0
    
    def __getitem__(self, idx):
        if not self.data:
            return self._get_dummy_item()
        
        direction = idx % 2
        data_idx = (idx // 2) % len(self.data)
        item = self.data[data_idx]
        
        # Create prompts
        if direction == 0:  # formal -> informal
            prompt = f"Translate to informal: {item['formal']}\nInformal:"
            target = item['informal']
        else:  # informal -> formal
            prompt = f"Translate to formal: {item['informal']}\nFormal:"
            target = item['formal']
        
        # Tokenize
        full_text = f"{prompt} {target}"
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels
        prompt_ids = self.tokenizer(prompt, truncation=True)['input_ids']
        labels = encoding['input_ids'].clone()
        labels[0, :len(prompt_ids)] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'direction': torch.tensor(direction, dtype=torch.long),
            'complexity': torch.tensor(item['complexity'], dtype=torch.long)
        }
    
    def _get_dummy_item(self):
        """Return dummy item for empty dataset"""
        return {
            'input_ids': torch.zeros(self.config.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.config.max_length, dtype=torch.long),
            'labels': torch.full((self.config.max_length,), -100, dtype=torch.long),
            'direction': torch.tensor(0, dtype=torch.long),
            'complexity': torch.tensor(1, dtype=torch.long)
        }

class FIRMA(nn.Module):
    """FIRMA model with DDP fixes"""
    
    def __init__(self, config: FIRMAConfig):
        super().__init__()
        self.config = config
        self.setup_base_model()
        
        # Get model hidden size
        self.model_hidden_size = getattr(self.base_model.config, 'hidden_size', 1536)
        
        # FIXED: Separate auxiliary components to avoid parameter conflicts
        self.use_auxiliary = True  # Flag to control auxiliary losses
        
        if self.use_auxiliary:
            self.direction_embedding = nn.Embedding(2, 64)
            self.complexity_embedding = nn.Embedding(config.num_complexity_levels, 64)
            self.input_projection = nn.Linear(self.model_hidden_size, config.hidden_dim)
            self.feature_projection = nn.Linear(config.hidden_dim + 128, config.hidden_dim)
            self.complexity_classifier = nn.Linear(config.hidden_dim, config.num_complexity_levels)
            self.validity_scorer = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def setup_base_model(self):
        """Setup base model with DDP fixes"""
        
        # Check hardware
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        # Log GPU info
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        
        # QLoRA config for T4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # FIXED: Better device mapping for DDP
        if world_size > 1 and local_rank >= 0:
            # For DDP, don't use device_map="auto"
            device_map = None
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
            logger.info(f"DDP mode: rank {local_rank}/{world_size}")
        else:
            device_map = "auto"
            logger.info("Single GPU mode")
        
        # Load model
        logger.info(f"Loading {self.config.base_model} for T4...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False
        )
        
        # Prepare for training
        self.base_model = prepare_model_for_kbit_training(
            self.base_model,
            use_gradient_checkpointing=self.config.gradient_checkpointing
        )
        
        # FIXED: Apply LoRA with better settings for DDP
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # FIXED: Add these for better DDP compatibility
            modules_to_save=None,  # Don't save additional modules
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.print_trainable_parameters()
        
        # FIXED: Enable static graph for DDP
        if world_size > 1 and self.config.use_static_graph:
            try:
                self.base_model._set_static_graph()
                logger.info("Enabled static graph for DDP")
            except AttributeError:
                logger.warning("Static graph not available for this model")
        
        # Move to device if DDP
        if world_size > 1 and local_rank >= 0:
            device = f'cuda:{local_rank}' if local_rank < torch.cuda.device_count() else 'cuda'
            self.base_model = self.base_model.to(device)
    
    def forward(self, input_ids, attention_mask, labels=None, 
                direction=None, complexity=None, **kwargs):
        """Forward pass with DDP fixes"""
        
        # FIXED: Always compute base model output first
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=self.use_auxiliary and self.training
        )
        
        # FIXED: Conditional auxiliary loss computation with consistent graph
        if self.use_auxiliary and self.training and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            try:
                # Get last hidden state and pool
                hidden = outputs.hidden_states[-1]
                pooled = hidden.mean(dim=1)
                
                # Ensure tensors are on the same device
                device = pooled.device
                
                # Project to smaller dimension
                pooled = self.input_projection(pooled)
                
                # FIXED: Always compute embeddings to maintain static graph
                if direction is not None:
                    if not isinstance(direction, torch.Tensor):
                        direction = torch.tensor(direction, device=device, dtype=torch.long)
                    if direction.dim() == 0:
                        direction = direction.unsqueeze(0)
                    elif direction.dim() > 1:
                        direction = direction.view(-1)
                else:
                    direction = torch.zeros(pooled.size(0), device=device, dtype=torch.long)
                
                if complexity is not None:
                    if not isinstance(complexity, torch.Tensor):
                        complexity = torch.tensor(complexity, device=device, dtype=torch.long)
                    if complexity.dim() == 0:
                        complexity = complexity.unsqueeze(0)
                    elif complexity.dim() > 1:
                        complexity = complexity.view(-1)
                else:
                    complexity = torch.ones(pooled.size(0), device=device, dtype=torch.long)
                
                # Ensure same batch size
                batch_size = pooled.size(0)
                if direction.size(0) != batch_size:
                    direction = direction.expand(batch_size)
                if complexity.size(0) != batch_size:
                    complexity = complexity.expand(batch_size)
                
                # Compute embeddings
                dir_emb = self.direction_embedding(direction)
                comp_emb = self.complexity_embedding(complexity)
                
                # Concatenate and project
                features = torch.cat([pooled, dir_emb, comp_emb], dim=-1)
                features = self.feature_projection(features)
                
                # Compute auxiliary losses
                comp_logits = self.complexity_classifier(features)
                comp_loss = F.cross_entropy(comp_logits, complexity)
                
                validity = self.validity_scorer(features)
                val_loss = -validity.mean() * 0.1
                
                # Update total loss
                if outputs.loss is not None:
                    outputs.loss = (outputs.loss + 
                                  self.config.lambda_complexity * comp_loss + 
                                  val_loss)
                        
            except Exception as e:
                logger.debug(f"Auxiliary loss computation failed: {e}")
                # Don't fail the forward pass
                pass
        
        return outputs
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            try:
                self.base_model.gradient_checkpointing_enable(**kwargs)
            except:
                self.base_model.gradient_checkpointing_enable()
    
    def generate(self, **kwargs):
        """Generate text"""
        return self.base_model.generate(**kwargs)
    
    def save_pretrained(self, path):
        """Save model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(path)
        
        # Save FIRMA components
        if self.use_auxiliary:
            torch.save({
                'direction_embedding': self.direction_embedding.state_dict(),
                'complexity_embedding': self.complexity_embedding.state_dict(),
                'input_projection': self.input_projection.state_dict(),
                'feature_projection': self.feature_projection.state_dict(),
                'complexity_classifier': self.complexity_classifier.state_dict(),
                'validity_scorer': self.validity_scorer.state_dict(),
                'config': self.config
            }, Path(path) / 'firma_components.pt')

class FIRMATrainer:
    """Trainer with DDP fixes"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def train(self):
        """Train FIRMA with DDP fixes"""
        logger.info("="*60)
        logger.info("Starting FIRMA training with DDP fixes")
        logger.info("="*60)
        
        # Get datasets
        train_dataset = self._get_dataset("train")
        val_dataset = self._get_dataset("val")
        
        if len(train_dataset) == 0:
            logger.error("No training data!")
            return
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # FIXED: Training arguments with DDP fixes
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            eval_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            optim="paged_adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
            # FIXED: DDP specific settings
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=25,  # Smaller bucket for T4
            ddp_broadcast_buffers=False,  # Save memory
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer
        )
        
        # Train with memory monitoring
        try:
            logger.info("Starting training...")
            start_time = time.time()
            
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            trainer.train()
            
            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed/3600:.2f} hours")
            
            # Save final model
            final_path = Path(self.config.output_dir) / "final"
            logger.info(f"Saving final model to {final_path}")
            self.model.save_pretrained(str(final_path))
            self.tokenizer.save_pretrained(str(final_path))
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _get_dataset(self, split):
        """Get dataset for split"""
        file_map = {
            'train': ['train.json', 'train.jsonl', 'train_clean.json'],
            'val': ['val.json', 'valid_clean.json'],
            'test': ['test.json', 'test_clean.json']
        }
        
        for filename in file_map.get(split, []):
            filepath = Path(self.config.data_dir) / filename
            if filepath.exists():
                logger.info(f"Using {filename} for {split}")
                return FIRMADataset(str(filepath), self.tokenizer, self.config, split)
        
        logger.warning(f"No {split} data found")
        return FIRMADataset("", self.tokenizer, self.config, split)

def main():
    """Main training function with DDP support"""
    
    # FIXED: Initialize distributed training if needed
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        logger.info(f"Initialized DDP on rank {local_rank}")
    
    # Initialize config for T4
    config = FIRMAConfig()
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer for {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    logger.info("Initializing FIRMA model with DDP fixes...")
    model = FIRMA(config)
    
    # Train
    trainer = FIRMATrainer(model, tokenizer, config)
    trainer.train()
    
    return model, tokenizer

if __name__ == "__main__":
    # Verify environment
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    else:
        logger.error("No CUDA devices available!")
        sys.exit(1)
    
    # Check for distributed setup
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        if local_rank >= torch.cuda.device_count():
            logger.error(f"LOCAL_RANK {local_rank} >= available GPUs")
            sys.exit(1)
    
    # Run training
    try:
        model, tokenizer = main()
        logger.info("="*60)
        logger.info("FIRMA TRAINING COMPLETE!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
