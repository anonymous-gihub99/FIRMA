#!/usr/bin/env python3
"""
baseline_implementation_qwen.py - Baseline models with Qwen for formal-informal translation
Using Qwen2.5-Math-7B for fine-tuning and Qwen3-4B for API baseline
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import faiss
from sentence_transformers import SentenceTransformer
import evaluate
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaselineConfig:
    """Configuration for baseline experiments with Qwen models"""
    # Updated to use Qwen models
    model_name: str = "Qwen/Qwen2.5-Math-7B"  # Math-specialized Qwen model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # For retrieval (Qwen3-Embedding-4B too large for T4)
    api_model: str = "Qwen/Qwen2.5-3B"  # Smaller Qwen for API simulation (fits on T4)
    
    output_dir: str = "./baseline_models"
    data_dir: str = "./math_alignment_dataset"
    max_length: int = 512
    batch_size: int = 2  # Small batch for T4 GPUs
    gradient_accumulation: int = 16  # Effective batch = 32
    learning_rate: float = 2e-4
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_4bit: bool = True  # QLoRA for memory efficiency
    device_map: str = "auto"
    
class MathTranslationDataset(Dataset):
    """Dataset for formal-informal translation"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, direction: str = "both"):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    def __len__(self):
        return len(self.data) * (2 if self.direction == "both" else 1)
    
    def __getitem__(self, idx):
        if not self.data:
            # Return dummy data if no data loaded
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }
            
        if self.direction == "both":
            pair_idx = idx // 2
            is_forward = idx % 2 == 0
        else:
            pair_idx = idx % len(self.data)
            is_forward = self.direction == "formal_to_informal"
        
        item = self.data[pair_idx]
        
        if is_forward:
            prompt = f"Translate formal mathematical statement to informal:\n[FORMAL]: {item['formal']}\n[INFORMAL]:"
            target = item['informal']
        else:
            prompt = f"Translate informal mathematical statement to formal:\n[INFORMAL]: {item['informal']}\n[FORMAL]:"
            target = item['formal']
        
        # Combine prompt and target
        full_text = f"{prompt} {target}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask prompt tokens)
        labels = encoding["input_ids"].clone()
        prompt_length = len(self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length))
        labels[0, :prompt_length] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

class DirectFineTuningBaseline:
    """Baseline 1: Direct fine-tuning with QLoRA using Qwen2.5-Math-7B"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / "direct_finetuning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup model with QLoRA
        self.setup_model()
        
    def setup_model(self):
        """Initialize Qwen model with QLoRA configuration"""
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load Qwen model
        logger.info(f"Loading {self.config.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config if self.config.use_4bit else None,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_4bit else torch.float16
        )
        
        # Prepare for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration for Qwen
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
    def train(self):
        """Train the Qwen model"""
        logger.info("Starting training with Qwen2.5-Math-7B...")
        
        # Load datasets
        train_dataset = MathTranslationDataset(
            f"{self.config.data_dir}/train.jsonl",
            self.tokenizer,
            self.config.max_length,
            direction="both"
        )
        
        val_dataset = MathTranslationDataset(
            f"{self.config.data_dir}/val.jsonl",
            self.tokenizer,
            self.config.max_length,
            direction="both"
        )
        
        if len(train_dataset.data) == 0:
            logger.warning("No training data found!")
            return
        
        # Training arguments optimized for T4/L4 GPUs
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(str(self.output_dir / "final_model"))
        logger.info(f"Model saved to {self.output_dir / 'final_model'}")
        
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate translation"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class RetrievalAugmentedBaseline:
    """Baseline 2: Retrieval-augmented translation using FAISS"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "retrieval_augmented"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedder = SentenceTransformer(config.embedding_model)
        
        # Storage for pairs and embeddings
        self.formal_texts = []
        self.informal_texts = []
        self.formal_embeddings = None
        self.informal_embeddings = None
        self.formal_index = None
        self.informal_index = None
        
    def build_index(self):
        """Build FAISS indices for retrieval"""
        logger.info("Building retrieval indices...")
        
        # Load training data
        train_path = f"{self.config.data_dir}/train.jsonl"
        if not Path(train_path).exists():
            logger.error(f"Training data not found at {train_path}")
            return
            
        with open(train_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    self.formal_texts.append(item['formal'])
                    self.informal_texts.append(item['informal'])
                except:
                    continue
        
        if not self.formal_texts:
            logger.error("No data loaded for index building")
            return
        
        # Compute embeddings
        logger.info("Computing embeddings...")
        batch_size = 32
        
        # Formal embeddings
        formal_embs = []
        for i in tqdm(range(0, len(self.formal_texts), batch_size), desc="Formal embeddings"):
            batch = self.formal_texts[i:i+batch_size]
            embs = self.embedder.encode(batch, convert_to_numpy=True)
            formal_embs.append(embs)
        self.formal_embeddings = np.vstack(formal_embs).astype('float32')
        
        # Informal embeddings
        informal_embs = []
        for i in tqdm(range(0, len(self.informal_texts), batch_size), desc="Informal embeddings"):
            batch = self.informal_texts[i:i+batch_size]
            embs = self.embedder.encode(batch, convert_to_numpy=True)
            informal_embs.append(embs)
        self.informal_embeddings = np.vstack(informal_embs).astype('float32')
        
        # Build FAISS indices
        dimension = self.formal_embeddings.shape[1]
        
        self.formal_index = faiss.IndexFlatL2(dimension)
        self.formal_index.add(self.formal_embeddings)
        
        self.informal_index = faiss.IndexFlatL2(dimension)
        self.informal_index.add(self.informal_embeddings)
        
        # Save indices
        faiss.write_index(self.formal_index, str(self.output_dir / "formal_index.faiss"))
        faiss.write_index(self.informal_index, str(self.output_dir / "informal_index.faiss"))
        
        logger.info(f"Indices built with {len(self.formal_texts)} pairs")
        
    def translate(self, text: str, direction: str = "formal_to_informal", k: int = 5) -> str:
        """Translate using retrieval"""
        if not self.formal_index or not self.informal_index:
            return text  # Return input if indices not built
            
        # Encode query
        query_embedding = self.embedder.encode([text], convert_to_numpy=True).astype('float32')
        
        if direction == "formal_to_informal":
            distances, indices = self.formal_index.search(query_embedding, k)
            candidates = [self.informal_texts[idx] for idx in indices[0] if idx < len(self.informal_texts)]
        else:
            distances, indices = self.informal_index.search(query_embedding, k)
            candidates = [self.formal_texts[idx] for idx in indices[0] if idx < len(self.formal_texts)]
        
        if candidates:
            # Return the best match or combine multiple
            if distances[0][0] < 0.1:
                return candidates[0]
            else:
                # Simple voting - return most common pattern
                return candidates[0] if candidates else text
        
        return text

class CommercialAPIBaseline:
    """Baseline 3: API baseline using Qwen3-4B"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "api_baseline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_model()
        
    def setup_model(self):
        """Setup Qwen3-4B for API simulation"""
        logger.info(f"Loading {self.config.api_model} for API baseline...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.api_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use smaller precision for T4 compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.api_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
    def translate_with_cot(self, text: str, direction: str = "formal_to_informal") -> str:
        """Translate using chain-of-thought prompting with Qwen"""
        
        if direction == "formal_to_informal":
            prompt = f"""You are an expert mathematician. Translate the following formal mathematical statement into clear, informal language.

Formal statement: {text}

Step-by-step translation:
1. Identify mathematical symbols and their meanings
2. Express each component in plain English
3. Combine into a natural explanation

Informal explanation:"""
        else:
            prompt = f"""You are an expert mathematician. Translate the following informal mathematical statement into precise formal notation.

Informal statement: {text}

Step-by-step translation:
1. Identify mathematical objects and relationships
2. Choose appropriate formal symbols
3. Write the complete formal statement

Formal notation:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation
        if direction == "formal_to_informal":
            marker = "Informal explanation:"
        else:
            marker = "Formal notation:"
        
        if marker in response:
            translation = response.split(marker)[-1].strip()
        else:
            translation = response.strip()
        
        return translation

# Keep the BaselineEvaluator class the same as before
class BaselineEvaluator:
    """Evaluate all baseline models"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.results_dir = Path(config.output_dir) / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
    def evaluate_model(self, model, test_data_path: str, model_name: str) -> Dict:
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            'model': model_name,
            'formal_to_informal': {},
            'informal_to_formal': {}
        }
        
        # Load test data
        test_data = []
        if Path(test_data_path).exists():
            with open(test_data_path, 'r') as f:
                for line in f:
                    try:
                        test_data.append(json.loads(line))
                    except:
                        continue
        
        # Sample for evaluation
        test_data = test_data[:100] if test_data else []
        
        for direction in ['formal_to_informal', 'informal_to_formal']:
            predictions = []
            references = []
            
            for item in tqdm(test_data, desc=f"{model_name} - {direction}"):
                if direction == 'formal_to_informal':
                    source = item['formal']
                    target = item['informal']
                else:
                    source = item['informal']
                    target = item['formal']
                
                # Generate translation
                try:
                    if hasattr(model, 'translate'):
                        pred = model.translate(source, direction)
                    elif hasattr(model, 'translate_with_cot'):
                        pred = model.translate_with_cot(source, direction)
                    else:
                        prompt = f"Translate {'formal to informal' if direction == 'formal_to_informal' else 'informal to formal'}:\n{source}\nTranslation:"
                        pred = model.generate(prompt)
                        if "Translation:" in pred:
                            pred = pred.split("Translation:")[-1].strip()
                except Exception as e:
                    logger.error(f"Error during translation: {e}")
                    pred = ""
                
                predictions.append(pred)
                references.append(target)
            
            # Calculate metrics
            if predictions and references:
                try:
                    bleu_score = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
                    rouge_score = self.rouge.compute(predictions=predictions, references=references)
                    
                    sample_size = min(20, len(predictions))
                    bert_score = self.bertscore.compute(
                        predictions=predictions[:sample_size],
                        references=references[:sample_size],
                        lang="en"
                    )
                    
                    results[direction] = {
                        'bleu': bleu_score['bleu'],
                        'rouge1': rouge_score['rouge1'],
                        'rouge2': rouge_score['rouge2'],
                        'rougeL': rouge_score['rougeL'],
                        'bertscore_f1': np.mean(bert_score['f1'])
                    }
                except Exception as e:
                    logger.error(f"Error computing metrics: {e}")
                    results[direction] = {
                        'bleu': 0.0,
                        'rouge1': 0.0,
                        'rouge2': 0.0,
                        'rougeL': 0.0,
                        'bertscore_f1': 0.0
                    }
        
        # Save results
        with open(self.results_dir / f"{model_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results