__version__ = "1.1.1"

from bi_mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from bi_mamba_ssm.modules.mamba_simple import Mamba
from bi_mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
