#!/usr/bin/env python3
"""
firma_model_updated.py - FIRMA model updated for AI4M dataset structure
Aligned with actual dataset format and Qwen models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FIRMAConfig:
    """Configuration for FIRMA model"""
    # Model architecture - Using Qwen for better performance
    base_model: str = "Qwen/Qwen2.5-Math-7B-Instruct"  # Better than Llemma based on results
    hidden_dim: int = 768
    num_complexity_levels: int = 4
    num_attention_heads: int = 12
    
    # Training parameters
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    num_epochs: int = 3
    batch_size: int = 2  # Small for T4 GPU
    gradient_accumulation: int = 16  # Effective batch = 32
    max_length: int = 512
    
    # Loss weights
    lambda_roundtrip: float = 0.2
    lambda_complexity: float = 0.1
    lambda_validity: float = 0.3
    
    # QLoRA settings
    use_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Paths - Updated for actual dataset
    output_dir: str = "./firma_model"
    data_dir: str = "./math_alignment_dataset"
    
    # Progressive training
    progressive_training: bool = False  # Disable for simpler training
    
class FIRMADataset(Dataset):
    """Dataset for FIRMA training - aligned with actual dataset structure"""
    
    def __init__(self, data_path: str, tokenizer, config: FIRMAConfig, split: str = "train"):
        self.data = []
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load data based on split
        if Path(data_path).exists():
            logger.info(f"Loading {split} dataset from {data_path}")
            
            # Check for different file formats
            if data_path.endswith('.jsonl'):
                # JSONL format
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            item = json.loads(line)
                            # Debug: print first item structure
                            if line_num == 0:
                                logger.info(f"First item keys: {list(item.keys())}")
                            
                            # Check for the ACTUAL key names in your dataset
                            # Your files use 'informal_stmt' and 'formal_stmt'
                            if 'informal_stmt' in item and 'formal_stmt' in item:
                                self.data.append({
                                    'formal': item['formal_stmt'],
                                    'informal': item['informal_stmt']
                                })
                            elif 'informal_statement' in item and 'formal_statement' in item:
                                self.data.append({
                                    'formal': item['formal_statement'],
                                    'informal': item['informal_statement']
                                })
                            elif 'formal' in item and 'informal' in item:
                                self.data.append(item)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse line {line_num}")
                            continue
                            
            elif data_path.endswith('.json'):
                # JSON array format - This is what your files use
                with open(data_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # Debug: Check structure
                        if data:
                            if isinstance(data, list):
                                logger.info(f"JSON array with {len(data)} items")
                                if len(data) > 0:
                                    logger.info(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                            elif isinstance(data, dict):
                                logger.info(f"JSON object with keys: {list(data.keys())[:10]}")
                        
                        # Handle JSON array (your format)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    # YOUR FILES USE 'informal_stmt' and 'formal_stmt'
                                    if 'informal_statement' in item and 'formal_statement' in item:
                                        self.data.append({
                                            'formal': item['formal_statement'],
                                            'informal': item['informal_statement']
                                        })
                                    elif 'informal_stmt' in item and 'formal_statement' in item:
                                        self.data.append({
                                            'formal': item['formal_statement'],
                                            'informal': item['informal_stmt']
                                        })
                                    elif 'formal' in item and 'informal' in item:
                                        self.data.append(item)
                                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse {data_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading {data_path}: {e}")
        
        # Log what we loaded
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        if self.data and len(self.data) > 0:
            sample = self.data[0]
            logger.info(f"Sample item: formal='{sample['formal'][:50]}...', informal='{sample['informal'][:50]}...'")
    
    def __len__(self):
        return len(self.data) * 2 if self.data else 1  # Return at least 1 to avoid empty dataset errors
    
    def __getitem__(self, idx):
        if not self.data:
            # Return dummy data
            return self._get_dummy_item()
            
        # Determine direction and actual data index
        direction = idx % 2  # 0: formal->informal, 1: informal->formal
        data_idx = (idx // 2) % len(self.data)
        
        item = self.data[data_idx]
        
        # Create input based on direction
        if direction == 0:  # formal -> informal
            source = item.get('formal', '')
            target = item.get('informal', '')
            prompt = f"Translate formal to informal:\n{source}\nInformal:"
        else:  # informal -> formal
            source = item.get('informal', '')
            target = item.get('formal', '')
            prompt = f"Translate informal to formal:\n{source}\nFormal:"
        
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
        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.config.max_length)
        prompt_length = len(prompt_tokens['input_ids'])
        
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100  # Mask prompt
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'direction': direction,
            'complexity': item.get('complexity_level', 2) - 1 if 'complexity_level' in item else 1,
            'formality': 0.2 if direction == 0 else 0.8
        }
    
    def _get_dummy_item(self):
        """Return dummy item when no data is available"""
        return {
            'input_ids': torch.zeros(self.config.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.config.max_length, dtype=torch.long),
            'labels': torch.zeros(self.config.max_length, dtype=torch.long),
            'direction': 0,
            'complexity': 1,
            'formality': 0.5
        }

class FIRMA(nn.Module):
    """Simplified FIRMA model focusing on core functionality"""
    
    def __init__(self, config: FIRMAConfig):
        super().__init__()
        self.config = config
        
        # Load base model with QLoRA
        self.setup_base_model()
        
        # Additional components for FIRMA
        self.direction_embeddings = nn.Embedding(2, 128)
        self.complexity_embeddings = nn.Embedding(config.num_complexity_levels, 128)
        
        # Projection layer
        self.output_projection = nn.Linear(128, 128)
        
    def setup_base_model(self):
        """Initialize base model with QLoRA"""
        
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        # Load Qwen model
        logger.info(f"Loading {self.config.base_model}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_4bit else torch.float16
        )
        
        # Prepare for training
        if self.config.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Simplified forward pass"""
        # Use base model directly
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, **kwargs):
        """Generate method for inference"""
        return self.base_model.generate(**kwargs)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        elif hasattr(self.base_model, 'enable_input_require_grads'):
            self.base_model.enable_input_require_grads()
    
    def save_pretrained(self, output_dir):
        """Save model"""
        self.base_model.save_pretrained(output_dir)
        
        # Save additional FIRMA components
        firma_state = {
            'direction_embeddings': self.direction_embeddings.state_dict(),
            'complexity_embeddings': self.complexity_embeddings.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'config': self.config
        }
        torch.save(firma_state, Path(output_dir) / 'firma_components.pt')
    
    def load_pretrained(self, input_dir):
        """Load model - simplified to avoid checkpoint mismatch"""
        try:
            # Try to load PEFT model
            from peft import PeftModel
            self.base_model = PeftModel.from_pretrained(
                self.base_model.base_model if hasattr(self.base_model, 'base_model') else self.base_model,
                input_dir
            )
            logger.info(f"Successfully loaded PEFT model from {input_dir}")
        except Exception as e:
            logger.warning(f"Could not load PEFT model: {e}")
            logger.info("Model will use fresh initialization")
        
        # Try to load FIRMA components
        firma_path = Path(input_dir) / 'firma_components.pt'
        if firma_path.exists():
            try:
                firma_state = torch.load(firma_path, map_location='cpu')
                self.direction_embeddings.load_state_dict(firma_state['direction_embeddings'])
                self.complexity_embeddings.load_state_dict(firma_state['complexity_embeddings'])
                self.output_projection.load_state_dict(firma_state['output_projection'])
                logger.info("Loaded FIRMA components")
            except Exception as e:
                logger.warning(f"Could not load FIRMA components: {e}")

class FIRMATrainer:
    """Trainer for FIRMA model"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def train(self):
        """Train FIRMA model"""
        logger.info("Starting FIRMA training...")
        
        # Prepare datasets
        train_dataset = self._get_dataset("train")
        val_dataset = self._get_dataset("val")
        
        # Check if we have data
        if len(train_dataset) <= 1:  # Using <= 1 because we return at least 1 for dummy
            logger.error("No training data found!")
            logger.info("Please check your dataset files. Expected format:")
            logger.info("  - JSON files with 'formal' and 'informal' keys")
            logger.info("  - Or 'text' field with pattern: 'informal statement [...].formal statement [...]'")
            return
        
        logger.info(f"Training with {len(train_dataset)} samples")
        logger.info(f"Validation with {len(val_dataset)} samples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(Path(self.config.output_dir) / "final")
        self.tokenizer.save_pretrained(Path(self.config.output_dir) / "final")
        
        logger.info("Training complete!")
    
    def _get_dataset(self, split):
        """Get dataset for specific split"""
        # Try different file naming conventions
        possible_files = [
            Path(self.config.data_dir) / f"{split}.jsonl",
            Path(self.config.data_dir) / f"{split}.json",
            Path(self.config.data_dir) / f"{split}_clean.json",
        ]
        
        # Special handling for train split
        if split == "train":
            possible_files.append(Path(self.config.data_dir) / "statements_part1.json")
            possible_files.append(Path(self.config.data_dir) / "train_clean.json")
        elif split == "val":
            possible_files.append(Path(self.config.data_dir) / "valid_clean.json")
            possible_files.append(Path(self.config.data_dir) / "validation.json")
        elif split == "test":
            possible_files.append(Path(self.config.data_dir) / "test_clean.json")
        
        for file_path in possible_files:
            if file_path and file_path.exists():
                logger.info(f"Loading {split} data from {file_path}")
                dataset = FIRMADataset(str(file_path), self.tokenizer, self.config, split)
                if len(dataset.data) > 0:
                    return dataset
        
        logger.warning(f"No {split} data found, creating empty dataset")
        return FIRMADataset("", self.tokenizer, self.config, split)

def create_and_train_firma():
    """Main function to create and train FIRMA"""
    
    # Initialize configuration
    config = FIRMAConfig()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    logger.info("Initializing FIRMA model...")
    model = FIRMA(config)
    
    # Enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
    except:
        pass
    
    # Create trainer
    trainer = FIRMATrainer(model, tokenizer, config)
    
    # Train
    trainer.train()
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = create_and_train_firma()