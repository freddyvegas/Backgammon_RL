"""
Model Size Configurations for PPO Agent

Add this to your ppo_agent.py to enable different model sizes.
Replace the Config class and add the get_config() function.
"""

# ---------------- Config ----------------
class Config:
    """Base configuration class."""
    state_dim = 29
    max_actions = 64
    
    # PPO hyperparameters
    lr = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    
    # Exploration
    entropy_coef = 0.02
    entropy_decay = 0.9999
    entropy_min = 0.0
    
    critic_coef = 1.0
    eval_temperature = 0.01
    
    # PPO rollout settings
    rollout_length = 512
    ppo_epochs = 4
    minibatch_size = 128
    
    # Network architecture (ResMLP) - DEFAULT: LARGE
    model_dim = 512
    n_blocks = 6
    
    # Gradient clipping
    grad_clip = 0.5
    max_grad_norm = 0.5
    
    # Weight decay for regularization
    weight_decay = 1e-5
    
    # Reward shaping
    use_reward_shaping = True
    pip_reward_scale = 0.01
    bear_off_reward = 0.05
    hit_reward = 0.02
    shaping_clip = 0.1


class SmallConfig(Config):
    """
    Small model for CPU training / quick testing.
    
    ~80K parameters (vs ~1.6M for large)
    Training speed: ~5-10x faster than large
    Performance: Expect ~5-10% lower win rates
    
    Use for:
    - CPU-only training
    - Quick prototyping
    - Testing code changes
    """
    model_dim = 128      # 512 → 128 (4x smaller)
    n_blocks = 3         # 6 → 3 (half the depth)
    rollout_length = 256 # 512 → 256 (faster updates)
    minibatch_size = 64  # 128 → 64 (fits in memory better)
    lr = 2e-4            # Slightly higher LR for faster learning


class MediumConfig(Config):
    """
    Medium model for modest GPUs / balanced training.
    
    ~400K parameters
    Training speed: ~2-3x faster than large
    Performance: Expect ~2-5% lower win rates
    
    Use for:
    - T4 GPU (Google Colab free tier)
    - Limited GPU time
    - Good balance of speed/performance
    """
    model_dim = 256      # 512 → 256 (2x smaller)
    n_blocks = 4         # 6 → 4 (slightly shallower)
    rollout_length = 384 # 512 → 384 (compromise)
    minibatch_size = 96  # 128 → 96


class LargeConfig(Config):
    """
    Large model (default) for full GPU training.
    
    ~1.6M parameters
    Best performance but slowest training
    
    Use for:
    - A100/V100 GPUs
    - Long training runs
    - Maximum performance
    """
    pass  # Uses defaults from Config


# Model size comparison table
MODEL_SIZES = {
    'small': {
        'config': SmallConfig,
        'params': '~80K',
        'speed': '5-10x faster',
        'performance': '-5 to -10% WR',
        'use_case': 'CPU training, quick tests',
    },
    'medium': {
        'config': MediumConfig,
        'params': '~400K',
        'speed': '2-3x faster',
        'performance': '-2 to -5% WR',
        'use_case': 'T4 GPU, limited time',
    },
    'large': {
        'config': LargeConfig,
        'params': '~1.6M',
        'speed': 'baseline',
        'performance': 'best',
        'use_case': 'Full GPU, long training',
    },
}


def get_config(size='large'):
    """
    Get configuration for specified model size.
    
    Args:
        size: 'small', 'medium', or 'large'
    
    Returns:
        Config instance
    
    Example:
        cfg = get_config('small')  # For CPU training
        agent = PPOAgent(config=cfg)
    """
    size = size.lower()
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(MODEL_SIZES.keys())}")
    
    config_class = MODEL_SIZES[size]['config']
    cfg = config_class()
    
    print(f"Model size: {size.upper()}")
    print(f"  Parameters: {MODEL_SIZES[size]['params']}")
    print(f"  Speed: {MODEL_SIZES[size]['speed']}")
    print(f"  Performance: {MODEL_SIZES[size]['performance']}")
    print(f"  Recommended for: {MODEL_SIZES[size]['use_case']}")
    print(f"  Architecture: dim={cfg.model_dim}, blocks={cfg.n_blocks}")
    print(f"  Rollout: length={cfg.rollout_length}, batch={cfg.minibatch_size}")
    
    return cfg


def print_model_comparison():
    """Print comparison table of all model sizes."""
    print("\n" + "=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)
    print(f"{'Size':<10} {'Params':<12} {'Speed':<15} {'Performance':<15} {'Use Case':<25}")
    print("-" * 80)
    
    for size, info in MODEL_SIZES.items():
        print(f"{size.capitalize():<10} {info['params']:<12} {info['speed']:<15} "
              f"{info['performance']:<15} {info['use_case']:<25}")
    
    print("=" * 80)
    print("\nRecommendations:")
    print("  • CPU or testing → small")
    print("  • Colab T4 GPU → medium")
    print("  • A100/V100 GPU → large")
    print("=" * 80 + "\n")


# Usage example in PPOAgent.__init__:
"""
class PPOAgent:
    def __init__(self, config=None, device=None):
        # Allow passing config or create default
        if config is None:
            config = Config()  # Or get_config('large') for default
        
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network with config's dimensions
        self.acnet = ActorCriticNet(
            in_dim=self.config.state_dim,
            d=self.config.model_dim,        # ← Uses config
            n_blocks=self.config.n_blocks   # ← Uses config
        ).to(self.device)
        
        # ... rest of init ...
"""


# ============================================================================
# COMPLETE UPDATED CONFIG SECTION FOR ppo_agent.py
# Replace lines 21-64 in your ppo_agent.py with this:
# ============================================================================

"""
# ---------------- Config ----------------
class Config:
    '''Base configuration class.'''
    state_dim = 29
    max_actions = 64
    
    # PPO hyperparameters
    lr = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    
    # Exploration
    entropy_coef = 0.02
    entropy_decay = 0.9999
    entropy_min = 0.0
    
    critic_coef = 1.0
    eval_temperature = 0.01
    
    # PPO rollout settings
    rollout_length = 512
    ppo_epochs = 4
    minibatch_size = 128
    
    # Network architecture (ResMLP) - DEFAULT: LARGE
    model_dim = 512
    n_blocks = 6
    
    # Gradient clipping
    grad_clip = 0.5
    max_grad_norm = 0.5
    
    # Weight decay for regularization
    weight_decay = 1e-5
    
    # Reward shaping
    use_reward_shaping = True
    pip_reward_scale = 0.01
    bear_off_reward = 0.05
    hit_reward = 0.02
    shaping_clip = 0.1


class SmallConfig(Config):
    '''Small model for CPU training (~80K params, 5-10x faster).'''
    model_dim = 128
    n_blocks = 3
    rollout_length = 256
    minibatch_size = 64
    lr = 2e-4


class MediumConfig(Config):
    '''Medium model for T4 GPU (~400K params, 2-3x faster).'''
    model_dim = 256
    n_blocks = 4
    rollout_length = 384
    minibatch_size = 96


class LargeConfig(Config):
    '''Large model for full GPU training (~1.6M params, best performance).'''
    pass


def get_config(size='large'):
    '''Get configuration for specified model size.'''
    configs = {
        'small': SmallConfig,
        'medium': MediumConfig,
        'large': LargeConfig,
    }
    
    if size.lower() not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    return configs[size.lower()]()


# Default config instance
CFG = Config()

# Set device at module level
if torch.cuda.is_available():
    CFG.device = "cuda"
else:
    CFG.device = "cpu"

print(f"PPO agent using device: {CFG.device}")
"""
