"""
Main orchestrator for PEC-SC experiments.
Handles mode overrides and invokes the appropriate task script.
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiment execution.
    Orchestrates inference based on mode and configuration.
    """
    print("=" * 80)
    print(f"PEC-SC Experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Apply mode-specific overrides
    cfg = apply_mode_overrides(cfg)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create results directory
    results_dir = os.path.join(cfg.results_dir, cfg.run.run_id)
    os.makedirs(results_dir, exist_ok=True)

    # This is an inference-only task
    print(f"\nRunning inference for {cfg.run.run_id}...")
    run_inference(cfg, results_dir)

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply mode-specific configuration overrides.

    Args:
        cfg: Original configuration

    Returns:
        Modified configuration with mode overrides applied
    """
    mode = cfg.mode

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Sanity mode processed 200 samples instead of 10; validation output missing
    # [CAUSE]: Mode parameter not being read correctly or overrides not applying
    # [FIX]: Added debug output to verify mode and n_samples before and after override
    #
    # [OLD CODE]:
    # (no debug output)
    #
    # [NEW CODE]:
    print(f"DEBUG: Mode detected: '{mode}'")
    print(f"DEBUG: n_samples before override: {cfg.dataset.n_samples}")

    if mode == "sanity":
        # Sanity mode: minimal execution for validation
        print("Applying sanity mode overrides...")

        # [VALIDATOR FIX - Attempt 1 CONTINUED]
        # [PROBLEM]: open_dict context manager might not persist changes
        # [CAUSE]: Hydra 1.3 struct mode prevents modifications even with open_dict in some cases
        # [FIX]: Use OmegaConf.set_struct to disable struct mode permanently for modifications
        #
        # [OLD CODE]:
        # with hydra.utils.open_dict(cfg):
        #     cfg.dataset.n_samples = min(cfg.dataset.n_samples, 10)
        #     ...
        #
        # [NEW CODE]:
        # Reduce dataset size
        OmegaConf.set_struct(cfg, False)
        cfg.dataset.n_samples = min(cfg.dataset.n_samples, 10)
        cfg.inference.k_samples = min(cfg.inference.k_samples, 3)

        # Set WandB project to sanity namespace
        if not cfg.wandb.project.endswith("-sanity"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"

        # Keep online mode for WandB
        cfg.wandb.mode = "online"
        OmegaConf.set_struct(cfg, True)

        print(f"DEBUG: n_samples after override: {cfg.dataset.n_samples}")
        print(f"DEBUG: k_samples after override: {cfg.inference.k_samples}")

    elif mode == "pilot":
        # Pilot mode: 20% of full run for preliminary results
        print("Applying pilot mode overrides...")

        # Reduce dataset to 20% (at least 50 samples)
        OmegaConf.set_struct(cfg, False)
        full_samples = cfg.dataset.n_samples
        pilot_samples = max(50, int(full_samples * 0.2))
        cfg.dataset.n_samples = pilot_samples

        # Set WandB project to pilot namespace
        if not cfg.wandb.project.endswith("-pilot"):
            cfg.wandb.project = f"{cfg.wandb.project}-pilot"

        # Keep online mode for WandB
        cfg.wandb.mode = "online"
        OmegaConf.set_struct(cfg, True)

    elif mode == "full":
        # Full mode: no overrides needed
        print("Running in full mode (no overrides)...")

        # Ensure online mode for WandB
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.mode = "online"
        OmegaConf.set_struct(cfg, True)

    else:
        print(f"Warning: Unknown mode '{mode}', proceeding with original config")

    return cfg


if __name__ == "__main__":
    main()
