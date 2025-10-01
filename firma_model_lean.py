#!/usr/bin/env python3
"""
firma_model_lean.py - FIRMA model adapted for Lean-Workbook dataset
Bidirectional translation between formal theorems and natural language
FIXED: Dynamic attention heads + cache compatibility + shared tensor save error
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FIRMAConfig:
    """Configuration for FIRMA model on Lean-Workbook dataset"""
    
    base_model: str = "Qwen/Qwen3-0.6B"
    hidden_dim: int = 768
    num_complexity_levels: int = 4
    num_attention_heads: int = 12
    dropout_rate: float = 0.1
    
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_length: int = 512
    max_grad_norm: float = 1.0
    
    lambda_translation: float = 1.0
    lambda_roundtrip: float = 0.2
    lambda_complexity: float = 0.1
    lambda_validity: float = 0.1
    lambda_proof: float = 0.15
    
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    use_4bit: bool = True
    
    output_dir: str = "./firma_lean_model"
    cache_dir: str = "./cache"
    
    gradient_checkpointing: bool = True
    progressive_training: bool = True
    complexity_schedule: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    lean_special_tokens: List[str] = field(default_factory=lambda: [
        "<THEOREM>", "</THEOREM>",
        "<PROOF>", "</PROOF>",
        "<FORMAL>", "</FORMAL>",
        "<INFORMAL>", "</INFORMAL>"
    ])
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    eps: float = 1e-8
    init_std: float = 0.02


class ProofStructureEncoder(nn.Module):
    """Encode proof structure from Lean tactics"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.self_attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        self.tactic_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.state_transition = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        attn_out, _ = self.self_attention(
            hidden_states, 
            hidden_states, 
            hidden_states,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        tactic_features = self.tactic_encoder(attn_out)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                tactic_features, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            gru_out, _ = self.state_transition(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, 
                batch_first=True, 
                total_length=seq_len
            )
        else:
            gru_out, _ = self.state_transition(tactic_features)
        
        output = self.layer_norm(gru_out + hidden_states)
        
        return output


class ComplexityRouter(nn.Module):
    """Route processing based on theorem complexity"""
    
    def __init__(self, hidden_size: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_levels)
        )
        
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.05)
            ) for _ in range(num_levels)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states, pooled_hidden=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if pooled_hidden is None:
            pooled_hidden = hidden_states.mean(dim=1)
        
        complexity_logits = self.complexity_classifier(pooled_hidden)
        complexity_probs = F.softmax(complexity_logits, dim=-1)
        
        transformed = torch.zeros_like(hidden_states)
        for level in range(self.num_levels):
            level_weight = complexity_probs[:, level:level+1].unsqueeze(1)
            level_output = self.level_transforms[level](hidden_states)
            transformed += level_weight * level_output
        
        gate_input = torch.cat([hidden_states, transformed], dim=-1)
        gate_weights = self.gate(gate_input)
        
        output = gate_weights * transformed + (1 - gate_weights) * hidden_states
        
        return output, complexity_logits


class FIRMA(nn.Module):
    """FIRMA model for Lean-Workbook dataset"""
    
    def __init__(self, config: FIRMAConfig):
        super().__init__()
        self.firma_config = config
        
        self._init_base_model()
        
        self.model_hidden_size = self.base_model.config.hidden_size
        logger.info(f"Model hidden size: {self.model_hidden_size}")
        
        self._init_firma_components()
        self._init_weights()
    
    @property
    def config(self):
        """Return base model config for Trainer compatibility"""
        return self.base_model.config
    
    def _get_valid_num_heads(self, hidden_size: int) -> int:
        """Calculate valid number of attention heads based on hidden size"""
        preferred_heads = [16, 12, 8, 6, 4, 2, 1]
        
        for num_heads in preferred_heads:
            if hidden_size % num_heads == 0:
                head_dim = hidden_size // num_heads
                if head_dim >= 32:
                    return num_heads
        
        return 1
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.base_model.gradient_checkpointing_disable()
    
    def _init_base_model(self):
        """Initialize base language model with quantization"""
        
        bnb_config = None
        if self.firma_config.use_4bit and self.firma_config.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        
        logger.info(f"Loading base model: {self.firma_config.base_model}")
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.firma_config.base_model,
                quantization_config=bnb_config,
                device_map="auto" if self.firma_config.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.firma_config.device == "cuda" else torch.float32,
                use_cache=False,
                attn_implementation="eager",
                cache_dir=self.firma_config.cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.firma_config.base_model}: {e}")
            logger.info("Falling back to GPT-2")
            self.firma_config.base_model = "gpt2"
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                cache_dir=self.firma_config.cache_dir
            )
        
        if self.firma_config.use_4bit and bnb_config:
            self.base_model = prepare_model_for_kbit_training(
                self.base_model,
                use_gradient_checkpointing=self.firma_config.gradient_checkpointing
            )
        
        if self.firma_config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            self.base_model.config.use_cache = False
        
        if self.firma_config.use_lora:
            self._apply_lora()
    
    def _apply_lora(self):
        """Apply LoRA adapters to base model"""
        logger.info("Applying LoRA adapters")
        
        peft_config = LoraConfig(
            r=self.firma_config.lora_r,
            lora_alpha=self.firma_config.lora_alpha,
            target_modules=self.firma_config.lora_target_modules,
            lora_dropout=self.firma_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.print_trainable_parameters()
    
    def _init_firma_components(self):
        """Initialize FIRMA-specific components"""
        
        self.direction_embeddings = nn.Embedding(2, self.model_hidden_size)
        self.complexity_embeddings = nn.Embedding(self.firma_config.num_complexity_levels, self.model_hidden_size)
        self.formality_embeddings = nn.Linear(1, self.model_hidden_size)
        
        num_heads = self._get_valid_num_heads(self.model_hidden_size)
        logger.info(f"Using {num_heads} attention heads for hidden size {self.model_hidden_size}")
        
        self.proof_encoder = ProofStructureEncoder(
            self.model_hidden_size,
            num_heads
        )
        
        self.complexity_router = ComplexityRouter(
            self.model_hidden_size,
            self.firma_config.num_complexity_levels
        )
        
        self.projection = None
        if self.model_hidden_size != self.firma_config.hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.model_hidden_size, self.firma_config.hidden_dim),
                nn.LayerNorm(self.firma_config.hidden_dim),
                nn.GELU()
            )
        
        final_dim = self.firma_config.hidden_dim if self.projection else self.model_hidden_size
        
        self.complexity_classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.firma_config.dropout_rate),
            nn.Linear(256, self.firma_config.num_complexity_levels)
        )
        
        self.validity_scorer = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.firma_config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.proof_validity = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.firma_config.dropout_rate),
            nn.Linear(256, 2)
        )
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        
        nn.init.normal_(self.direction_embeddings.weight, mean=0.0, std=self.firma_config.init_std)
        nn.init.normal_(self.complexity_embeddings.weight, mean=0.0, std=self.firma_config.init_std)
        nn.init.normal_(self.formality_embeddings.weight, mean=0.0, std=self.firma_config.init_std)
        nn.init.zeros_(self.formality_embeddings.bias)
        
        for module in [self.complexity_classifier, self.validity_scorer, self.proof_validity]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=self.firma_config.init_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def pool_hidden_states(self, hidden_states, attention_mask=None):
        """Pool hidden states with attention mask"""
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=self.firma_config.eps)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        direction=None,
        complexity=None,
        formality=None,
        proof_valid=None,
        return_dict=True,
        **kwargs
    ):
        """Forward pass with FIRMA enhancements"""
        
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['num_items_in_batch']}
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **filtered_kwargs
        )
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
            
            proof_encoded = self.proof_encoder(hidden_states, attention_mask)
            pooled = self.pool_hidden_states(proof_encoded, attention_mask)
            
            if direction is not None:
                direction = torch.clamp(direction, 0, 1)
                pooled = pooled + self.direction_embeddings(direction) * 0.1
            
            if complexity is not None:
                complexity = torch.clamp(complexity, 0, self.firma_config.num_complexity_levels - 1)
                pooled = pooled + self.complexity_embeddings(complexity) * 0.1
            
            if formality is not None:
                formality = torch.clamp(formality, 0.0, 1.0)
                if formality.dim() == 1:
                    formality = formality.unsqueeze(-1)
                pooled = pooled + self.formality_embeddings(formality) * 0.1
            
            routed_hidden, complexity_logits = self.complexity_router(proof_encoded, pooled)
            
            if self.projection is not None:
                pooled = self.projection(pooled)
            
            total_loss = outputs.loss if outputs.loss is not None else 0.0
            
            if complexity is not None and self.firma_config.lambda_complexity > 0:
                complexity_pred = self.complexity_classifier(pooled)
                complexity_loss = F.cross_entropy(complexity_pred, complexity)
                total_loss = total_loss + self.firma_config.lambda_complexity * complexity_loss
            
            if self.firma_config.lambda_validity > 0:
                validity_score = self.validity_scorer(pooled)
                validity_loss = -torch.log(validity_score + self.firma_config.eps).mean()
                total_loss = total_loss + self.firma_config.lambda_validity * validity_loss
            
            if proof_valid is not None and self.firma_config.lambda_proof > 0:
                proof_pred = self.proof_validity(pooled)
                proof_loss = F.cross_entropy(proof_pred, proof_valid)
                total_loss = total_loss + self.firma_config.lambda_proof * proof_loss
            
            outputs.loss = total_loss
        
        return outputs
    
    def generate(self, **kwargs):
        """Generate text with cache handling"""
        
        cache_setting = self.base_model.config.use_cache if hasattr(self.base_model, 'config') else True
        
        if self.firma_config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_disable()
            if hasattr(self.base_model, 'config'):
                self.base_model.config.use_cache = True
        
        try:
            output = self.base_model.generate(**kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            output = kwargs.get('input_ids', torch.zeros(1, 1, dtype=torch.long))
        finally:
            if self.firma_config.gradient_checkpointing:
                self.base_model.gradient_checkpointing_enable()
                if hasattr(self.base_model, 'config'):
                    self.base_model.config.use_cache = cache_setting
        
        return output
    
    def save_pretrained(self, output_dir):
        """Save model and FIRMA components"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # THE FIX: Untie weights before saving to avoid shared memory error
        if hasattr(self.base_model, 'base_model'):
            # For PEFT models
            base = self.base_model.base_model.model
        else:
            # For non-PEFT models
            base = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        
        # Untie weights if they are tied
        if hasattr(base, 'lm_head') and hasattr(base, 'embed_tokens'):
            if base.lm_head.weight.data_ptr() == base.embed_tokens.weight.data_ptr():
                logger.info("Untying shared weights before saving")
                base.lm_head.weight = nn.Parameter(base.embed_tokens.weight.clone())
        
        self.base_model.save_pretrained(output_dir)
        
        firma_state = {
            'direction_embeddings': self.direction_embeddings.state_dict(),
            'complexity_embeddings': self.complexity_embeddings.state_dict(),
            'formality_embeddings': self.formality_embeddings.state_dict(),
            'proof_encoder': self.proof_encoder.state_dict(),
            'complexity_router': self.complexity_router.state_dict(),
            'complexity_classifier': self.complexity_classifier.state_dict(),
            'validity_scorer': self.validity_scorer.state_dict(),
            'proof_validity': self.proof_validity.state_dict(),
            'config': self.firma_config
        }
        
        if self.projection is not None:
            firma_state['projection'] = self.projection.state_dict()
        
        torch.save(firma_state, Path(output_dir) / 'firma_components.pt')
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        """Load pretrained FIRMA model"""
        
        if config is None:
            firma_components = torch.load(Path(model_path) / 'firma_components.pt', map_location='cpu')
            config = firma_components['config']
        
        model = cls(config)
        
        from transformers import AutoModelForCausalLM
        model.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map="auto" if config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        firma_components = torch.load(Path(model_path) / 'firma_components.pt', map_location=config.device)
        
        model.direction_embeddings.load_state_dict(firma_components['direction_embeddings'])
        model.complexity_embeddings.load_state_dict(firma_components['complexity_embeddings'])
        model.formality_embeddings.load_state_dict(firma_components['formality_embeddings'])
        model.proof_encoder.load_state_dict(firma_components['proof_encoder'])
        model.complexity_router.load_state_dict(firma_components['complexity_router'])
        model.complexity_classifier.load_state_dict(firma_components['complexity_classifier'])
        model.validity_scorer.load_state_dict(firma_components['validity_scorer'])
        model.proof_validity.load_state_dict(firma_components['proof_validity'])
        
        if 'projection' in firma_components and model.projection is not None:
            model.projection.load_state_dict(firma_components['projection'])
        
        logger.info(f"Model loaded from {model_path}")
        return model
