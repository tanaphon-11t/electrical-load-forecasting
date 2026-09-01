"""Registry for the three V2 methods; Production V1 registry remains untouched."""

from .chatgpt_carr1m import ChatGPTCARR1M
from .gemini_cg_hae import GeminiCGHAEV1
from .kimi_pgf_tea import KimiPGFTEA

METHOD_IDS = ("chatgpt_carr1m", "gemini_cg_hae_v1", "kimi_pgf_tea")

METHOD_REGISTRY = {
    "chatgpt_carr1m": ChatGPTCARR1M,
    "gemini_cg_hae_v1": GeminiCGHAEV1,
    "kimi_pgf_tea": KimiPGFTEA,
}
