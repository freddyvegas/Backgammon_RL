#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Opponent pool with fixed checkpoint loading and MPS support.
- Proper device handling (CUDA/MPS/CPU)
- Error recovery
- Better logging
"""

from pathlib import Path
import random
import shutil
import importlib
import torch


def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS > CPU
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class OpponentPool:
    """
    Manages a pool of frozen agent checkpoints for opponent sampling.
    
    IMPROVEMENTS:
    - Fixed device handling for loaded opponents (CUDA/MPS/CPU)
    - Error recovery when loading fails
    - Better logging and diagnostics
    """

    def __init__(self, pool_dir: Path, agent_module_name: str = "ppo_agent",
                 max_size: int = 12, seed: int = 42, device: str = None):
        """
        Args:
            pool_dir: Directory to store opponent snapshots
            agent_module_name: Name of agent module to import (must expose PPOAgent)
            max_size: Maximum number of snapshots to keep
            seed: Random seed for opponent sampling
            device: Device for opponents ('cpu', 'cuda', 'mps', or None for auto-detect)
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.agent_module_name = agent_module_name
        self.max_size = max_size
        self.snapshots = []  # list[(path: Path, snapshot_id: int)]
        self._snapshot_counter = 0
        
        # Determine device for opponents
        if device is not None:
            self.device = device
            print(f"OpponentPool using explicit device: {device}")
        else:
            self.device = get_device()
            print(f"OpponentPool using auto-detected device: {self.device}")

        self._load_existing_snapshots()
        random.seed(seed)

        print(f"OpponentPool initialized:")
        print(f"  Directory: {self.pool_dir}")
        print(f"  Max size: {max_size}")
        print(f"  Device: {self.device}")
        print(f"  Existing snapshots: {len(self.snapshots)}")

    def _load_existing_snapshots(self):
        """Load existing opponent snapshots from directory."""
        if not self.pool_dir.exists():
            return

        for snapshot_file in sorted(self.pool_dir.glob("opp_*.pt")):
            try:
                snap_id = int(snapshot_file.stem.split('_')[1])
                self.snapshots.append((snapshot_file, snap_id))
                self._snapshot_counter = max(self._snapshot_counter, snap_id + 1)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse opponent snapshot {snapshot_file.name}")

        self.snapshots.sort(key=lambda x: x[1])  # sort by id

    def add_snapshot(self, src_path: Path, label: str = ""):
        """
        Add a new opponent snapshot to the pool. Evicts oldest if pool is full.
        """
        dst = self.pool_dir / f"opp_{self._snapshot_counter}.pt"
        if str(src_path) != str(dst):
            try:
                shutil.copy(src_path, dst)
            except Exception as e:
                print(f"[OpponentPool] Error copying snapshot: {e}")
                return

        self.snapshots.append((dst, self._snapshot_counter))
        print(f"[OpponentPool] Added snapshot {self._snapshot_counter} {label}")
        self._snapshot_counter += 1

        if len(self.snapshots) > self.max_size:
            old_path, old_id = self.snapshots.pop(0)
            try:
                old_path.unlink()
                print(f"[OpponentPool] Evicted snapshot {old_id} (pool full)")
            except Exception as e:
                print(f"[OpponentPool] Could not delete {old_path}: {e}")

        print(f"[OpponentPool] Pool size: {len(self.snapshots)}/{self.max_size}")

    def sample_opponent(self, bias_recent: bool = True, max_retries: int = 3):
        """
        Sample an opponent from the pool with error recovery.

        Args:
            bias_recent: If True, bias sampling toward recent snapshots
            max_retries: Maximum number of retries if loading fails

        Returns:
            A NEW ppo_agent.PPOAgent instance with loaded weights, or None if fails.
        """
        if not self.snapshots:
            return None

        for attempt in range(max_retries):
            try:
                # Choose snapshot index
                if bias_recent and len(self.snapshots) > 1:
                    # geometric bias toward recent snapshots
                    n = len(self.snapshots)
                    weights = [0.5 ** (n - 1 - i) for i in range(n)]
                    idx = random.choices(range(n), weights=weights, k=1)[0]
                else:
                    idx = random.randrange(len(self.snapshots))

                snapshot_path, snapshot_id = self.snapshots[idx]

                # Import agent module and create a fresh PPOAgent instance
                agent_mod = importlib.import_module(self.agent_module_name)
                
                # Create opponent with explicit device
                # Check if PPOAgent accepts device parameter
                try:
                    opponent = agent_mod.PPOAgent(device=self.device)
                except TypeError:
                    opponent = agent_mod.PPOAgent()

                # Load weights into this instance
                opponent.load(str(snapshot_path), map_location=self.device, load_optimizer=False)
                
                # Set eval mode
                opponent.set_eval_mode(True)
                
                # Verify it loaded correctly
                if hasattr(opponent, 'steps') and opponent.steps > 0:
                    # Successfully loaded trained weights
                    return opponent
                else:
                    print(f"[OpponentPool] Warning: Snapshot {snapshot_id} has no training steps")
                    if attempt < max_retries - 1:
                        continue  # Try another snapshot
                    else:
                        return opponent  # Return anyway on last attempt
                        
            except Exception as e:
                print(f"[OpponentPool] Error loading snapshot {snapshot_id} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"[OpponentPool] Failed to load opponent after {max_retries} attempts")
                    return None
                # Try again with a different snapshot
                continue

        return None

    def verify_snapshots(self):
        """
        Verify all snapshots can be loaded.
        Returns list of (snapshot_id, loadable: bool)
        """
        results = []
        print(f"\n[OpponentPool] Verifying {len(self.snapshots)} snapshots...")
        
        for snapshot_path, snapshot_id in self.snapshots:
            try:
                # Try to load checkpoint metadata
                checkpoint = torch.load(snapshot_path, map_location='cpu')
                has_weights = "acnet" in checkpoint
                has_steps = "steps" in checkpoint
                steps = checkpoint.get("steps", 0)
                
                if has_weights:
                    print(f"  Snapshot {snapshot_id}: ✓ Valid (steps={steps})")
                    results.append((snapshot_id, True))
                else:
                    print(f"  Snapshot {snapshot_id}: ✗ Missing weights")
                    results.append((snapshot_id, False))
                    
            except Exception as e:
                print(f"  Snapshot {snapshot_id}: ✗ Load error: {e}")
                results.append((snapshot_id, False))
        
        valid_count = sum(1 for _, valid in results if valid)
        print(f"\n[OpponentPool] {valid_count}/{len(results)} snapshots are valid")
        return results

    def remove_invalid_snapshots(self):
        """Remove snapshots that can't be loaded."""
        valid_snapshots = []
        removed_count = 0
        
        for snapshot_path, snapshot_id in self.snapshots:
            try:
                checkpoint = torch.load(snapshot_path, map_location='cpu')
                if "acnet" in checkpoint:
                    valid_snapshots.append((snapshot_path, snapshot_id))
                else:
                    snapshot_path.unlink()
                    removed_count += 1
                    print(f"[OpponentPool] Removed invalid snapshot {snapshot_id}")
            except Exception:
                try:
                    snapshot_path.unlink()
                except:
                    pass
                removed_count += 1
                print(f"[OpponentPool] Removed corrupted snapshot {snapshot_id}")
        
        self.snapshots = valid_snapshots
        print(f"[OpponentPool] Removed {removed_count} invalid snapshots, {len(self.snapshots)} remain")

    def get_pool_info(self):
        """Get formatted pool information."""
        if not self.snapshots:
            return "Pool: empty"
        snapshot_ids = [sid for _, sid in self.snapshots]
        return f"Pool: {len(self.snapshots)} snapshots (IDs: {min(snapshot_ids)}-{max(snapshot_ids)})"

    def __len__(self):
        return len(self.snapshots)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("OpponentPool Testing")
    print("=" * 60)
    
    # Create test pool
    pool = OpponentPool(
        pool_dir=Path("./test_opponent_pool"),
        agent_module_name="ppo_agent",
        max_size=5,
        seed=42
    )
    
    # Test verification
    if len(pool) > 0:
        print("\nVerifying existing snapshots:")
        pool.verify_snapshots()
        
        print("\nSampling opponent:")
        opponent = pool.sample_opponent()
        if opponent:
            print(f"✓ Successfully loaded opponent")
            print(f"  Steps: {opponent.steps if hasattr(opponent, 'steps') else 'N/A'}")
            print(f"  Eval mode: {opponent.eval_mode if hasattr(opponent, 'eval_mode') else 'N/A'}")
        else:
            print("✗ Failed to load opponent")
    else:
        print("\nNo snapshots in pool to test")
    
    print("\n" + "=" * 60)
