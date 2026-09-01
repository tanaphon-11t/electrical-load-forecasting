"""Stable method identifiers and importable placeholder registry."""

from .chatgpt_rsrb import ChatGPTRSRB
from .gemini_arc_xgb import GeminiARCXGB
from .kimi_ramen import KimiRAMEn


METHOD_IDS = (
    "chatgpt_rsrb",
    "gemini_arc_xgb",
    "kimi_ramen",
)

METHOD_REGISTRY = {
    "chatgpt_rsrb": ChatGPTRSRB,
    "gemini_arc_xgb": GeminiARCXGB,
    "kimi_ramen": KimiRAMEn,
}


def get_method_class(method_id: str):
    """Return the registered class without instantiating or training it."""
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown forecasting method: {method_id}") from exc
