"""
llm/__init__.py — Factory functions untuk LLM clients.

Ini adalah satu-satunya pintu masuk ke LLM layer.
Semua agent harus import dari sini:

    from llm import get_llm_main, get_llm_scorer, get_llm_vision

TIDAK BOLEH import client langsung:
    from llm.gemini_client import GeminiClient  ← SALAH
    from llm import get_llm_main                ← BENAR

Dengan pola ini, migrasi dari Gemini ke Claude cukup ganti
LLM_MAIN=claude di .env — tidak ada perubahan di code agent manapun.
"""

from llm.base import BaseLLMClient
import config


def get_llm_main() -> BaseLLMClient:
    """
    Return LLM client untuk task utama:
    Generator, Strategic Planner, Answer Key Agent.
    
    Ditentukan oleh LLM_MAIN di .env:
    - gemini → GeminiClient
    - claude → ClaudeClient
    """
    if config.LLM_MAIN == "gemini":
        from llm.gemini_client import GeminiClient
        return GeminiClient()
    elif config.LLM_MAIN == "claude":
        from llm.claude_client import ClaudeClient
        return ClaudeClient()
    else:
        raise ValueError(
            f"LLM_MAIN='{config.LLM_MAIN}' tidak dikenal. "
            f"Pilihan valid: 'gemini' | 'claude'"
        )


def get_llm_scorer() -> BaseLLMClient:
    """
    Return LLM client untuk scoring & validation:
    Validator, Difficulty Calibrator.
    
    Ditentukan oleh LLM_SCORER di .env:
    - groq  → GroqClient
    - claude → ClaudeClient (dengan use_scorer=True → pakai Haiku)
    """
    if config.LLM_SCORER == "groq":
        from llm.groq_client import GroqClient
        return GroqClient()
    elif config.LLM_SCORER == "claude":
        from llm.claude_client import ClaudeClient
        return ClaudeClient()
    else:
        raise ValueError(
            f"LLM_SCORER='{config.LLM_SCORER}' tidak dikenal. "
            f"Pilihan valid: 'groq' | 'claude'"
        )


def get_llm_vision() -> BaseLLMClient:
    """
    Return LLM client untuk vision/image captioning:
    KB ingestion pipeline untuk dokumen yang mengandung gambar.
    
    Ditentukan oleh LLM_VISION di .env:
    - gemini → GeminiClient (support vision)
    - claude → ClaudeClient (support vision)
    
    CATATAN: Jangan gunakan GroqClient untuk vision —
    complete_vision() akan raise NotImplementedError.
    """
    if config.LLM_VISION == "gemini":
        from llm.gemini_client import GeminiClient
        return GeminiClient()
    elif config.LLM_VISION == "claude":
        from llm.claude_client import ClaudeClient
        return ClaudeClient()
    else:
        raise ValueError(
            f"LLM_VISION='{config.LLM_VISION}' tidak dikenal. "
            f"Pilihan valid: 'gemini' | 'claude'"
        )