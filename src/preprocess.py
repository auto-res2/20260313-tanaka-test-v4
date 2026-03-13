"""
Data preprocessing for GSM8K dataset.
Loads and prepares grade-school math problems for inference.
"""

import re
from typing import List, Dict, Any
from datasets import load_dataset
import random


def load_gsm8k(
    split: str = "test",
    n_samples: int = None,
    shuffle_seed: int = 42,
    cache_dir: str = ".cache",
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split to load (train or test)
        n_samples: Number of samples to use (None = all)
        shuffle_seed: Random seed for shuffling
        cache_dir: Directory to cache downloaded dataset

    Returns:
        List of dictionaries with 'question', 'answer', and 'numeric_answer' fields
    """
    # Load dataset from HuggingFace
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    examples = []
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer from the answer text
        # GSM8K answers end with #### followed by the numeric answer
        numeric_answer = extract_numeric_answer(answer_text)

        examples.append(
            {
                "question": question,
                "answer": answer_text,
                "numeric_answer": numeric_answer,
            }
        )

    # Shuffle and subsample if requested
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(examples)

    if n_samples is not None and n_samples < len(examples):
        examples = examples[:n_samples]

    return examples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract the numeric answer from GSM8K answer text.
    GSM8K answers end with #### followed by the numeric answer.

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numeric answer as float
    """
    # Find the #### marker
    if "####" in answer_text:
        numeric_part = answer_text.split("####")[-1].strip()
        # Remove commas and convert to float
        numeric_part = numeric_part.replace(",", "")
        try:
            return float(numeric_part)
        except ValueError:
            # Try to extract any number from the text
            numbers = re.findall(r"-?\d+\.?\d*", numeric_part)
            if numbers:
                return float(numbers[0])
            return 0.0
    else:
        # Try to find any number in the answer
        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
        if numbers:
            return float(numbers[-1])
        return 0.0


def extract_number_from_text(text: str) -> float:
    """
    Extract a numeric answer from generated text.
    Used to parse model outputs.

    Args:
        text: Generated text that should contain a numeric answer

    Returns:
        Extracted number as float, or 0.0 if not found
    """
    # Remove common prefixes
    text = text.strip()

    # Look for patterns like "The answer is X", "#### X", "FINAL: X", etc.
    patterns = [
        r"(?:FINAL|answer|Answer|result|Result)[\s:=]+(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",  # number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue

    # Fallback: find any number in the text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        # Take the last number as it's likely the final answer
        num_str = numbers[-1].replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    return 0.0
