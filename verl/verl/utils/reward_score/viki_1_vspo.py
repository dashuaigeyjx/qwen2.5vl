# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VIKI-L1 reward function with VSPO semantic validation integration.
This module integrates VSPO (Validating Semantic Pitfalls in Ontology) 
semantic validation as an additional reward signal for GRPO training.
"""

import re
import ast
from typing import Optional, Dict, Any
from mathruler.grader import extract_boxed_content, grade_answer

# VSPO imports - semantic similarity for ontology validation
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    VSPO_AVAILABLE = True
except ImportError:
    VSPO_AVAILABLE = False
    print("Warning: sentence_transformers not available. VSPO semantic validation will be disabled.")


class VSPOSemanticValidator:
    """VSPO semantic validator for ontology-based semantic consistency checking."""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        threshold: float = 0.7,
        use_cuda: bool = True,
        enabled: bool = True
    ):
        """
        Initialize VSPO semantic validator.
        
        Args:
            model_name: Sentence transformer model name for semantic similarity
            threshold: Cosine similarity threshold for semantic matching
            use_cuda: Whether to use CUDA if available
            enabled: Whether VSPO validation is enabled
        """
        self.enabled = enabled and VSPO_AVAILABLE
        if not self.enabled:
            return
            
        self.threshold = threshold
        device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
        except Exception as e:
            print(f"Warning: Failed to load VSPO model: {e}. VSPO validation disabled.")
            self.enabled = False
    
    def compute_semantic_score(
        self,
        generated_text: str,
        reference_text: Optional[str] = None,
        ontology_context: Optional[str] = None
    ) -> float:
        """
        Compute semantic similarity score using VSPO methodology.
        
        Args:
            generated_text: Generated response text
            reference_text: Reference/ground truth text (optional)
            ontology_context: Ontology context for semantic validation (optional)
            
        Returns:
            Semantic similarity score between 0.0 and 1.0
        """
        if not self.enabled:
            return 0.0
            
        try:
            # Extract reasoning and answer from generated text
            # Support both <think> and <think> tags for compatibility
            reasoning_match = re.search(r'<(?:think|redacted_reasoning)>(.*?)</(?:think|redacted_reasoning)>', generated_text, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
            
            if not reasoning_match or not answer_match:
                return 0.0
            
            reasoning_text = reasoning_match.group(1).strip()
            answer_text = answer_match.group(1).strip()
            
            # Combine reasoning and answer for semantic validation
            combined_text = f"{reasoning_text} {answer_text}"
            
            # If reference text is provided, compute similarity
            if reference_text:
                texts = [combined_text, reference_text]
                embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
                similarity = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
                # Normalize to [0, 1] and apply threshold
                normalized_score = max(0.0, (similarity + 1.0) / 2.0)  # cosine sim is [-1, 1]
                return normalized_score if normalized_score >= self.threshold else 0.0
            
            # If ontology context is provided, validate against ontology
            if ontology_context:
                texts = [combined_text, ontology_context]
                embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
                similarity = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
                normalized_score = max(0.0, (similarity + 1.0) / 2.0)
                return normalized_score if normalized_score >= self.threshold else 0.0
            
            # If no reference, return a baseline score based on structure
            return 0.5
            
        except Exception as e:
            print(f"Error in VSPO semantic validation: {e}")
            return 0.0


# Global VSPO validator instance (lazy initialization)
_vspo_validator: Optional[VSPOSemanticValidator] = None


def get_vspo_validator(
    model_name: str = 'all-MiniLM-L6-v2',
    threshold: float = 0.7,
    use_cuda: bool = True,
    enabled: bool = True
) -> VSPOSemanticValidator:
    """Get or create global VSPO validator instance."""
    global _vspo_validator
    if _vspo_validator is None:
        _vspo_validator = VSPOSemanticValidator(
            model_name=model_name,
            threshold=threshold,
            use_cuda=use_cuda,
            enabled=enabled
        )
    return _vspo_validator


def format_reward(predict_str: str) -> float:
    """
    Check overall structure with <think> and <answer> tags.
    Same as original VIKI-L1 format_reward.
    """
    structure_pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
    structure_match = re.fullmatch(structure_pattern, predict_str)
    
    if not structure_match:
        return 0.0
    
    # Check if answer is in Python list format
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = re.search(answer_pattern, predict_str)
    
    if not answer_match:
        return 0.0
    
    answer_content = answer_match.group(1).strip()
    list_pattern = re.compile(r'\[.*\]', re.DOTALL)
    list_match = re.fullmatch(list_pattern, answer_content)
    
    return 1.0 if list_match else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Accuracy reward based on answer correctness.
    Same as original VIKI-L1 acc_reward.
    """
    # Extract answer from <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = re.search(answer_pattern, predict_str)
    answer = match.group(1).strip() if match else ""
    
    try:
        # Parse both strings as Python lists
        pred_list = ast.literal_eval(answer)
        gt_list = ast.literal_eval(ground_truth)

        if isinstance(pred_list, list) and isinstance(gt_list, list):
            # Try hashing via tuple conversion (handle list of lists)
            pred_set = set(tuple(x) if isinstance(x, list) else x for x in pred_list)
            gt_set = set(tuple(x) if isinstance(x, list) else x for x in gt_list)
            return 1.0 if pred_set == gt_set else 0.0
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # Fall back to original comparison if parsing fails
    return 1.0 if grade_answer(answer, str(ground_truth)) else 0.0


def vspo_semantic_reward(
    predict_str: str,
    ground_truth: Optional[str] = None,
    ontology_context: Optional[str] = None,
    vspo_weight: float = 0.1,
    **vspo_kwargs
) -> float:
    """
    VSPO semantic validation reward.
    
    Args:
        predict_str: Generated prediction string
        ground_truth: Ground truth string (optional)
        ontology_context: Ontology context for validation (optional)
        vspo_weight: Weight for VSPO reward component
        **vspo_kwargs: Additional VSPO validator parameters
        
    Returns:
        VSPO semantic reward score between 0.0 and vspo_weight
    """
    validator = get_vspo_validator(**vspo_kwargs)
    semantic_score = validator.compute_semantic_score(
        generated_text=predict_str,
        reference_text=ground_truth,
        ontology_context=ontology_context
    )
    return semantic_score * vspo_weight


def compute_score(
    predict_str: str,
    ground_truth: str,
    vspo_enabled: bool = True,
    vspo_weight: float = 0.1,
    format_weight: float = 0.1,
    acc_weight: float = 0.8,
    ontology_context: Optional[str] = None,
    **vspo_kwargs
) -> float:
    """
    Compute combined reward score with VSPO semantic validation.
    
    Args:
        predict_str: Generated prediction string
        ground_truth: Ground truth string
        vspo_enabled: Whether to enable VSPO semantic validation
        vspo_weight: Weight for VSPO reward component (default: 0.1)
        format_weight: Weight for format reward (default: 0.1)
        acc_weight: Weight for accuracy reward (default: 0.8)
        ontology_context: Ontology context for VSPO validation (optional)
        **vspo_kwargs: Additional VSPO validator parameters
        
    Returns:
        Combined reward score
        
    Note:
        Total weights should sum to 1.0. If vspo_enabled=True, 
        the weights are normalized: format_weight + acc_weight + vspo_weight = 1.0
    """
    # Normalize weights
    if vspo_enabled:
        total_weight = format_weight + acc_weight + vspo_weight
        format_weight = format_weight / total_weight
        acc_weight = acc_weight / total_weight
        vspo_weight = vspo_weight / total_weight
    else:
        # If VSPO disabled, redistribute weights
        total_weight = format_weight + acc_weight
        format_weight = format_weight / total_weight
        acc_weight = acc_weight / total_weight
        vspo_weight = 0.0
    
    # Compute base rewards
    format_score = format_reward(predict_str)
    acc_score = acc_reward(predict_str, ground_truth)
    
    # Compute VSPO semantic reward if enabled
    vspo_score = 0.0
    if vspo_enabled:
        vspo_score = vspo_semantic_reward(
            predict_str=predict_str,
            ground_truth=ground_truth,
            ontology_context=ontology_context,
            vspo_weight=1.0,  # Will be weighted later
            enabled=vspo_enabled,
            **vspo_kwargs
        )
    
    # Combine rewards
    total_score = (
        format_weight * format_score +
        acc_weight * acc_score +
        vspo_weight * vspo_score
    )
    
    return total_score


# Backward compatibility: default compute_score without VSPO
def compute_score_original(predict_str: str, ground_truth: str) -> float:
    """Original VIKI-L1 compute_score without VSPO (for backward compatibility)."""
    return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)
