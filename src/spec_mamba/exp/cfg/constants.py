"""Experiment constants and default values."""

# Default MelSpectrogram config
SAMPLE_RATE = 48_000
N_FFT = 4096
N_MELS = 128

# Default normalization values
DB_MIN = -50
DB_MAX = 60

# Mamba/SSAST pretrained configs
AUDIO_MAMBA_DEFAULT_CONFIG = {
    "tiny": {"depth": 12, "embed_dim": 192},
    "small": {"depth": 12, "embed_dim": 384},
    "base": {"depth": 12, "embed_dim": 768},
}
SSM_CONFIG = {"d_state": 24, "d_conv": 4, "expand": 3}

SSAST_DEFAULT_CONFIG = {
    "tiny": {"depth": 12, "num_heads": 3, "embed_dim": 192},
    "small": {"depth": 12, "num_heads": 6, "embed_dim": 384},
    "base": {"depth": 12, "num_heads": 12, "embed_dim": 768},
}
