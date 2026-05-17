"""BAGEL-SBSR: Saliency-biased Sparse Routing + iMF/DMD2 dual distillation on BAGEL-7B-MoT."""

from .hook import BagelNotInstalledError, patch_bagel, unpatch_bagel
from .saliency import attention_rollout, saliency_from_attentions
from .sbsr import SBSR

__version__ = "0.1.0.dev0"

__all__ = [
    "SBSR",
    "BagelNotInstalledError",
    "__version__",
    "attention_rollout",
    "patch_bagel",
    "saliency_from_attentions",
    "unpatch_bagel",
]
