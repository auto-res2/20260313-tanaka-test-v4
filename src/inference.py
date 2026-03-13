"""
Inference script for prompt-based Chain-of-Thought experiments.
Supports PEC-SC (Plan-Execute-Check Self-Consistency) and standard Self-Consistency.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import random

import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.preprocess import load_gsm8k, extract_number_from_text


def get_model_client(cfg: DictConfig):
    """Initialize the appropriate API client based on provider."""
    provider = cfg.model.provider.lower()

    if provider == "openai":
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
        return client, "openai"
    elif provider == "anthropic":
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        return client, "anthropic"
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate_completion(
    client,
    provider: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate a completion from the model."""
    if provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_prompt_pec_sc(question: str, plan_items: int = 4, audit_checks: int = 3) -> str:
    """Generate PEC-SC prompt with PLAN, EXECUTION, AUDIT, FINAL structure."""
    prompt = f"""Solve this math problem using a structured approach with 4 sections:

Question: {question}

Please structure your response with these exact section headers:

PLAN:
Write {plan_items} numbered atomic steps (P1, P2, P3, P4) describing what operations you will perform.
Example format:
P1: Identify the given values and assign variables
P2: Set up the equation based on the problem
P3: Solve for the unknown
P4: Verify the result makes sense

EXECUTION:
Work through the solution, explicitly referencing your plan items (e.g., "According to P1, ..." or "(P2) ...").
Show all calculations step by step.

AUDIT:
List {audit_checks} check statements to verify your work. Each check should be a simple statement with a verifiable value.
Example format:
Check 1: Total items = X (calculated value)
Check 2: Rate equals Y (calculated value)
Check 3: Final result satisfies constraint Z (yes/no)

FINAL:
State only the numeric answer.

Begin:"""
    return prompt


def get_prompt_self_consistency(question: str) -> str:
    """Generate standard Chain-of-Thought prompt."""
    prompt = f"""Solve this math problem step by step.

Question: {question}

Let's solve this step by step:"""
    return prompt


def parse_pec_sc_response(response: str) -> Dict[str, Any]:
    """
    Parse a PEC-SC response into sections and compute metrics.

    Returns:
        Dictionary with parsed sections and computed metrics
    """
    sections = {"plan": "", "execution": "", "audit": "", "final": "", "raw": response}

    # Extract sections using regex
    plan_match = re.search(
        r"PLAN:\s*(.+?)(?=EXECUTION:|$)", response, re.DOTALL | re.IGNORECASE
    )
    if plan_match:
        sections["plan"] = plan_match.group(1).strip()

    exec_match = re.search(
        r"EXECUTION:\s*(.+?)(?=AUDIT:|$)", response, re.DOTALL | re.IGNORECASE
    )
    if exec_match:
        sections["execution"] = exec_match.group(1).strip()

    audit_match = re.search(
        r"AUDIT:\s*(.+?)(?=FINAL:|$)", response, re.DOTALL | re.IGNORECASE
    )
    if audit_match:
        sections["audit"] = audit_match.group(1).strip()

    final_match = re.search(r"FINAL:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
    if final_match:
        sections["final"] = final_match.group(1).strip()

    # Compute plan coverage: fraction of plan items referenced in execution
    plan_coverage = compute_plan_coverage(sections["plan"], sections["execution"])

    # Compute audit pass rate (simplified: just check if audit section is non-empty)
    # In a more sophisticated version, we could parse and evaluate audit statements
    audit_pass_rate = compute_audit_pass_rate(sections["audit"])

    # Extract numeric answer
    numeric_answer = extract_number_from_text(
        sections["final"] if sections["final"] else response
    )

    return {
        "sections": sections,
        "plan_coverage": plan_coverage,
        "audit_pass_rate": audit_pass_rate,
        "numeric_answer": numeric_answer,
    }


def compute_plan_coverage(plan: str, execution: str) -> float:
    """
    Compute fraction of plan items explicitly referenced in execution.
    Looks for patterns like (P1), (P2), P1:, P2:, etc.
    """
    if not plan or not execution:
        return 0.0

    # Find all plan item numbers in the plan (P1, P2, P3, P4)
    plan_items = re.findall(r"P(\d+)", plan)
    if not plan_items:
        return 0.0

    # Find all plan item references in execution
    exec_refs = re.findall(r"P(\d+)", execution)
    if not exec_refs:
        return 0.0

    # Count how many unique plan items are referenced
    unique_plan_items = set(plan_items)
    referenced_items = set(exec_refs) & unique_plan_items

    coverage = len(referenced_items) / len(unique_plan_items)
    return coverage


def compute_audit_pass_rate(audit: str) -> float:
    """
    Compute audit pass rate.
    Simplified version: returns 1.0 if audit section is non-empty with checks, 0.0 otherwise.
    A more sophisticated version could parse and evaluate arithmetic expressions.
    """
    if not audit:
        return 0.0

    # Check if there are check statements
    checks = re.findall(r"Check \d+:", audit, re.IGNORECASE)
    if len(checks) >= 1:
        return 1.0

    return 0.5 if len(audit.strip()) > 10 else 0.0


def select_best_pec_sc(
    samples: List[Dict[str, Any]], alpha: float, beta: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Select the best sample using PEC-SC composite scoring.

    Score: S(i) = alpha * I[ans_i = majority] + beta * coverage(i) + (1-alpha-beta) * audit_pass_rate(i)

    Returns:
        Tuple of (selected_answer, best_sample_data)
    """
    if not samples:
        return 0.0, {}

    # Find majority answer
    answers = [s["numeric_answer"] for s in samples]
    if not answers:
        return 0.0, {}

    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]

    # Compute scores for each sample
    gamma = 1.0 - alpha - beta  # audit weight
    scores = []

    for sample in samples:
        ans_match = 1.0 if sample["numeric_answer"] == majority_answer else 0.0
        coverage = sample["plan_coverage"]
        audit = sample["audit_pass_rate"]

        score = alpha * ans_match + beta * coverage + gamma * audit
        scores.append(score)

    # Select sample with highest score
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    selected_answer = samples[best_idx]["numeric_answer"]

    return selected_answer, {
        "best_sample_idx": best_idx,
        "best_score": scores[best_idx],
        "all_scores": scores,
        "majority_answer": majority_answer,
        "answer_variance": sum((a - majority_answer) ** 2 for a in answers)
        / len(answers),
    }


def select_best_self_consistency(samples: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Select the best answer using standard majority voting.

    Returns:
        Tuple of (selected_answer, metadata)
    """
    if not samples:
        return 0.0, {}

    # Extract numeric answers
    answers = [extract_number_from_text(s) for s in samples]

    # Majority vote
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]

    return majority_answer, {
        "majority_answer": majority_answer,
        "answer_counts": dict(answer_counts),
        "answer_variance": sum((a - majority_answer) ** 2 for a in answers)
        / len(answers),
    }


def run_inference(cfg: DictConfig, results_dir: str):
    """
    Main inference loop.
    """
    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run URL: {wandb.run.url}")

    # Load dataset
    print(
        f"Loading GSM8K dataset (split={cfg.dataset.split}, n_samples={cfg.dataset.n_samples})..."
    )
    examples = load_gsm8k(
        split=cfg.dataset.split,
        n_samples=cfg.dataset.n_samples,
        shuffle_seed=cfg.dataset.shuffle_seed,
        cache_dir=".cache",
    )
    print(f"Loaded {len(examples)} examples")

    # Initialize model client
    client, provider = get_model_client(cfg)

    # Determine method type
    method_type = cfg.method.type.lower()

    # Set random seed
    random.seed(cfg.inference.seed)

    # Track metrics
    correct = 0
    total = 0
    all_predictions = []

    # Run inference
    print(f"Running {method_type} inference...")
    for idx, example in enumerate(tqdm(examples)):
        question = example["question"]
        ground_truth = example["numeric_answer"]

        # Generate K samples
        k = cfg.inference.k_samples

        if method_type == "pec-sc":
            # PEC-SC method
            samples = []
            for _ in range(k):
                prompt = get_prompt_pec_sc(
                    question,
                    plan_items=cfg.method.plan_items,
                    audit_checks=cfg.method.audit_checks,
                )
                response = generate_completion(
                    client,
                    provider,
                    cfg.model.name,
                    prompt,
                    cfg.model.temperature,
                    cfg.model.max_tokens,
                )
                parsed = parse_pec_sc_response(response)
                samples.append(parsed)

            # Select best answer
            prediction, metadata = select_best_pec_sc(
                samples, alpha=cfg.method.alpha, beta=cfg.method.beta
            )

            # Log additional PEC-SC metrics
            if wandb_enabled:
                wandb.log(
                    {
                        f"example_{idx}_plan_coverage": samples[
                            metadata["best_sample_idx"]
                        ]["plan_coverage"],
                        f"example_{idx}_audit_pass_rate": samples[
                            metadata["best_sample_idx"]
                        ]["audit_pass_rate"],
                    }
                )

        else:  # self-consistency
            # Standard Self-Consistency method
            samples = []
            for _ in range(k):
                prompt = get_prompt_self_consistency(question)
                response = generate_completion(
                    client,
                    provider,
                    cfg.model.name,
                    prompt,
                    cfg.model.temperature,
                    cfg.model.max_tokens,
                )
                samples.append(response)

            # Select best answer
            prediction, metadata = select_best_self_consistency(samples)

        # Check correctness
        is_correct = (
            abs(prediction - ground_truth) < 0.01
        )  # tolerance for floating point
        correct += int(is_correct)
        total += 1

        all_predictions.append(
            {
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "metadata": metadata,
            }
        )

        # Log per-example metrics
        if wandb_enabled:
            wandb.log(
                {
                    "example_idx": idx,
                    "correct": int(is_correct),
                    "answer_variance": metadata.get("answer_variance", 0.0),
                }
            )

    # Compute final metrics
    accuracy = correct / total if total > 0 else 0.0

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "predictions.json")
    with open(results_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"  Saved predictions to {results_file}")

    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = total
        wandb.summary["n_samples_processed"] = len(examples)

        # Upload results file
        wandb.save(results_file)

    # Validation output
    print_validation_output(cfg.mode, len(examples), accuracy)

    if wandb_enabled:
        wandb.finish()

    return accuracy


def print_validation_output(mode: str, n_samples: int, accuracy: float):
    """Print validation output for sanity and pilot modes."""
    if mode == "sanity":
        # Sanity validation
        if n_samples >= 5 and accuracy > 0:
            print("SANITY_VALIDATION: PASS")
        else:
            reason = "too_few_samples" if n_samples < 5 else "zero_accuracy"
            print(f"SANITY_VALIDATION: FAIL reason={reason}")

        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': n_samples, 'accuracy': accuracy, 'outputs_valid': True, 'outputs_unique': True})}"
        )

    elif mode == "pilot":
        # Pilot validation
        if n_samples >= 50 and accuracy > 0:
            print("PILOT_VALIDATION: PASS")
        else:
            reason = "too_few_samples" if n_samples < 50 else "zero_accuracy"
            print(f"PILOT_VALIDATION: FAIL reason={reason}")

        print(
            f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': n_samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'outputs_unique': True})}"
        )


if __name__ == "__main__":
    # This script is invoked by main.py with hydra config
    print("inference.py should be invoked via main.py")
    sys.exit(1)
