#!/usr/bin/env python3
"""
dataset_curation_ai4m.py - Dataset curation using AI4M/less-proofnet-lean4-ranked
Real Lean 4 formal-informal mathematical pairs from a non-profit community
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
from datasets import load_dataset

@dataclass
class MathPair:
    """Single formal-informal mathematical statement pair"""
    formal: str
    informal: str
    source: str
    theorem_name: str
    complexity_level: int  # 1-4
    proof_depth: int
    symbol_density: float
    domain: str
    metadata: Dict

class AI4MDatasetCurator:
    def __init__(self, output_dir: str = "./math_alignment_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize collection statistics
        self.stats = {
            'total_pairs': 0,
            'by_complexity': {1: 0, 2: 0, 3: 0, 4: 0},
            'parse_errors': 0,
            'filtered_out': 0
        }
        
    def parse_ai4m_text(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse AI4M dataset text to extract formal and informal statements
        Format: "informal statement [text].formal statement [text]"
        Note: The period before "formal statement" has no space after it
        """
        try:
            # The pattern is: "informal statement [informal_text].formal statement [formal_text]"
            # Split by the exact pattern ".formal statement"
            
            if 'informal statement' in text and '.formal statement' in text:
                # Find the exact position of ".formal statement"
                formal_marker = '.formal statement'
                formal_start_idx = text.find(formal_marker)
                
                if formal_start_idx > 0:
                    # Everything before ".formal statement" is the informal part
                    informal_full = text[:formal_start_idx]
                    
                    # Remove "informal statement" prefix from informal part
                    if informal_full.startswith('informal statement'):
                        informal = informal_full[len('informal statement'):].strip()
                    else:
                        informal = informal_full.strip()
                    
                    # Everything after ".formal statement" is the formal part
                    formal = text[formal_start_idx + len(formal_marker):].strip()
                    
                    # Clean up
                    informal = ' '.join(informal.split())
                    formal = ' '.join(formal.split())
                    
                    # Validate
                    if informal and formal and len(informal) > 5 and len(formal) > 5:
                        return informal, formal
            
            # Fallback: try without the period
            elif 'informal statement' in text and 'formal statement' in text:
                # Use regex to be more flexible
                import re
                
                # Try multiple patterns
                patterns = [
                    r'informal statement\s+(.*?)\.formal statement\s+(.*)',  # With period, no space
                    r'informal statement\s+(.*?)\.\s*formal statement\s+(.*)',  # With period and space
                    r'informal statement\s+(.*?)\s+formal statement\s+(.*)',  # Just spaces
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        informal = match.group(1).strip()
                        formal = match.group(2).strip()
                        
                        # Clean up
                        informal = ' '.join(informal.split())
                        formal = ' '.join(formal.split())
                        
                        if informal and formal and len(informal) > 5 and len(formal) > 5:
                            return informal, formal
            
            return None
            
        except Exception as e:
            # Silently fail to avoid spam in logs
            return None
    
    def extract_theorem_name(self, formal_statement: str) -> str:
        """Extract theorem name from formal statement if present"""
        # Look for theorem/lemma/exercise names
        patterns = [
            r'theorem\s+(\w+)',
            r'lemma\s+(\w+)',
            r'exercise_(\w+)',
            r'example_(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, formal_statement)
            if match:
                return match.group(1)
        
        # Generate a name from hash if no explicit name found
        return f"thm_{hash(formal_statement) % 10000:04d}"
    
    def calculate_complexity(self, formal_statement: str) -> Dict:
        """Calculate complexity metrics for a formal statement"""
        # Count logical symbols and quantifiers
        logic_symbols = ['∀', '∃', '→', '↔', '¬', '∧', '∨', '⊢', '⊨', '∣', '⊂', '⊆', '∈']
        symbol_count = sum(formal_statement.count(s) for s in logic_symbols)
        
        # Count quantifier depth
        quantifier_depth = self._get_quantifier_depth(formal_statement)
        
        # Count nested structures (parentheses, brackets)
        nesting_level = self._get_nesting_level(formal_statement)
        
        # Symbol density
        total_chars = len(formal_statement)
        symbol_density = symbol_count / max(total_chars, 1)
        
        # Determine complexity level (1-4)
        if quantifier_depth <= 1 and symbol_count <= 3 and nesting_level <= 2:
            level = 1  # Basic
        elif quantifier_depth <= 2 and symbol_count <= 6 and nesting_level <= 4:
            level = 2  # Intermediate
        elif quantifier_depth <= 3 and symbol_count <= 10:
            level = 3  # Advanced
        else:
            level = 4  # Expert
        
        return {
            'level': level,
            'depth': quantifier_depth,
            'symbol_density': symbol_density,
            'symbol_count': symbol_count,
            'nesting_level': nesting_level
        }
    
    def _get_quantifier_depth(self, statement: str) -> int:
        """Calculate maximum quantifier nesting depth"""
        depth = 0
        current_depth = 0
        
        for char in statement:
            if char in ['∀', '∃']:
                current_depth += 1
                depth = max(depth, current_depth)
            elif char in ['.', ',', ':'] and current_depth > 0:
                current_depth -= 1
        
        return depth
    
    def _get_nesting_level(self, statement: str) -> int:
        """Calculate maximum nesting level of parentheses/brackets"""
        max_depth = 0
        current_depth = 0
        
        for char in statement:
            if char in ['(', '[', '{']:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in [')', ']', '}']:
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def infer_domain(self, informal: str, formal: str) -> str:
        """Infer mathematical domain from statement content"""
        text = (informal + ' ' + formal).lower()
        
        domain_keywords = {
            'Algebra': ['group', 'ring', 'field', 'polynomial', 'matrix', 'vector'],
            'Analysis': ['limit', 'continuous', 'derivative', 'integral', 'convergent', 'sequence'],
            'NumberTheory': ['prime', 'divisible', 'gcd', 'modulo', 'integer', 'odd', 'even'],
            'Topology': ['open', 'closed', 'compact', 'hausdorff', 'neighborhood', 'closure'],
            'Logic': ['prove', 'implies', 'iff', 'exists', 'forall'],
            'Geometry': ['triangle', 'circle', 'angle', 'perpendicular', 'parallel'],
            'SetTheory': ['subset', 'union', 'intersection', 'cardinality', 'element']
        }
        
        # Count keyword matches for each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'General'
    
    def load_and_process_ai4m(self, max_samples: Optional[int] = None, debug: bool = False) -> List[MathPair]:
        """Load and process the AI4M dataset from HuggingFace"""
        print("Loading AI4M/less-proofnet-lean4-ranked dataset...")
        
        # Cache file for processed data
        cache_file = self.cache_dir / "ai4m_processed.pkl"
        
        if cache_file.exists():
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load dataset from HuggingFace
        try:
            dataset = load_dataset("AI4M/less-proofnet-lean4-ranked", split="train")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure you have internet connection and the dataset is accessible")
            return []
        
        pairs = []
        
        # Debug: Print first few examples to verify format
        if debug:
            print("\nDEBUG: First 3 examples from dataset:")
            for i, item in enumerate(dataset):
                if i >= 3:
                    break
                text = item.get('text', '')
                print(f"\nExample {i+1}:")
                print(f"Raw text: {text[:200]}...")
                parsed = self.parse_ai4m_text(text)
                if parsed:
                    informal, formal = parsed
                    print(f"✓ Successfully parsed:")
                    print(f"  Informal: {informal[:100]}...")
                    print(f"  Formal: {formal[:100]}...")
                else:
                    print(f"✗ Failed to parse")
        
        # Process the dataset
        for idx, item in enumerate(tqdm(dataset, desc="Processing AI4M dataset")):
            if max_samples and idx >= max_samples:
                break
            
            # Extract text field
            text = item.get('text', '')
            
            # Parse formal and informal statements
            parsed = self.parse_ai4m_text(text)
            
            if parsed is None:
                self.stats['parse_errors'] += 1
                # Debug: show first few parse errors
                if debug and self.stats['parse_errors'] <= 3:
                    print(f"\nDEBUG: Parse error on item {idx}:")
                    print(f"Text: {text[:200]}...")
                continue
            
            informal, formal = parsed
            
            # Skip if either is too short
            if len(informal) < 10 or len(formal) < 10:
                self.stats['filtered_out'] += 1
                continue
            
            # Calculate complexity
            complexity = self.calculate_complexity(formal)
            
            # Extract theorem name
            theorem_name = self.extract_theorem_name(formal)
            
            # Infer domain
            domain = self.infer_domain(informal, formal)
            
            # Create MathPair
            pair = MathPair(
                formal=formal,
                informal=informal,
                source="AI4M/less-proofnet-lean4",
                theorem_name=theorem_name,
                complexity_level=complexity['level'],
                proof_depth=complexity['depth'],
                symbol_density=complexity['symbol_density'],
                domain=domain,
                metadata={
                    'index': idx,
                    'symbol_count': complexity['symbol_count'],
                    'nesting_level': complexity['nesting_level']
                }
            )
            
            pairs.append(pair)
            self.stats['by_complexity'][complexity['level']] += 1
        
        self.stats['total_pairs'] = len(pairs)
        
        # Save to cache
        if pairs:  # Only cache if we got data
            with open(cache_file, 'wb') as f:
                pickle.dump(pairs, f)
        
        print(f"\nProcessed {len(pairs)} pairs from AI4M dataset")
        print(f"Parse errors: {self.stats['parse_errors']}")
        print(f"Filtered out: {self.stats['filtered_out']}")
        
        return pairs
    
    def create_splits(self, pairs: List[MathPair], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[MathPair]]:
        """Create train/val/test splits stratified by complexity"""
        np.random.seed(42)
        
        # Group by complexity level
        by_complexity = {i: [] for i in range(1, 5)}
        for pair in pairs:
            by_complexity[pair.complexity_level].append(pair)
        
        splits = {'train': [], 'val': [], 'test': []}
        
        # Stratified splitting
        for level, level_pairs in by_complexity.items():
            np.random.shuffle(level_pairs)
            n = len(level_pairs)
            
            train_end = int(train_ratio * n)
            val_end = int((train_ratio + val_ratio) * n)
            
            splits['train'].extend(level_pairs[:train_end])
            splits['val'].extend(level_pairs[train_end:val_end])
            splits['test'].extend(level_pairs[val_end:])
        
        # Shuffle final splits
        for split in splits:
            np.random.shuffle(splits[split])
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(splits['train'])} pairs")
        print(f"  Val: {len(splits['val'])} pairs")
        print(f"  Test: {len(splits['test'])} pairs")
        
        return splits
    
    def save_dataset(self, splits: Dict[str, List[MathPair]]):
        """Save dataset in JSONL format"""
        print("\nSaving dataset...")
        
        # Save each split as JSONL
        for split_name, split_data in splits.items():
            jsonl_path = self.output_dir / f"{split_name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for pair in split_data:
                    f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')
            print(f"  Saved {split_name}.jsonl ({len(split_data)} pairs)")
        
        # Save statistics
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create challenge set (hardest examples)
        challenge_set = []
        for pair in splits['test']:
            if pair.complexity_level >= 3:
                challenge_set.append(pair)
        
        challenge_set = challenge_set[:500]  # Limit to 500
        if challenge_set:
            challenge_path = self.output_dir / "challenge_set.jsonl"
            with open(challenge_path, 'w', encoding='utf-8') as f:
                for pair in challenge_set:
                    f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')
            print(f"  Saved challenge_set.jsonl ({len(challenge_set)} pairs)")
        
        print(f"\nDataset saved to {self.output_dir}")
        
        # Print complexity distribution
        print("\nComplexity distribution:")
        for level in [1, 2, 3, 4]:
            count = self.stats['by_complexity'][level]
            percentage = (count / self.stats['total_pairs'] * 100) if self.stats['total_pairs'] > 0 else 0
            print(f"  Level {level}: {count} pairs ({percentage:.1f}%)")
    
    def run_full_pipeline(self, max_samples: Optional[int] = None, debug: bool = False):
        """Execute the complete dataset curation pipeline"""
        print("="*60)
        print("AI4M Dataset Curation Pipeline")
        print("="*60)
        
        # Load and process AI4M dataset
        pairs = self.load_and_process_ai4m(max_samples=max_samples, debug=debug)
        
        if not pairs:
            print("No pairs extracted from dataset!")
            print("\nTroubleshooting: Try running with debug=True to see parsing details")
            return None
        
        # Create splits
        splits = self.create_splits(pairs)
        
        # Save dataset
        self.save_dataset(splits)
        
        print("\n" + "="*60)
        print("Dataset curation complete!")
        print("="*60)
        
        return splits

if __name__ == "__main__":
    # Create curator
    curator = AI4MDatasetCurator(output_dir="./math_alignment_dataset")
    
    # Run pipeline with debug mode for first run
    # Use debug=True to see parsing details, False for normal operation
    splits = curator.run_full_pipeline(max_samples=None, debug=True)
    
    if splits:
        # Display sample pairs
        print("\nSample pairs from training set:")
        for i, pair in enumerate(splits['train'][:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Informal: {pair.informal[:100]}...")
            print(f"Formal: {pair.formal[:100]}...")
            print(f"Complexity: Level {pair.complexity_level}")
            print(f"Domain: {pair.domain}")