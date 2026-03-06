"""
llm/groq_client.py — Implementasi LLM client untuk Groq.

Dipakai saat LLM_SCORER=groq (mode development).
Model yang dipakai: llama-3.3-70b-versatile (free tier).

Groq TIDAK support vision — complete_vision() akan raise NotImplementedError.

Free tier limits:
- 30 RPM (request per menit)
- 500.000 TPD (token per hari)
- Cocok untuk: Validator & Difficulty Calibrator (banyak call, token kecil)
"""

import time
from groq import Groq
from typing import Type
from pydantic import BaseModel

from llm.base import BaseLLMClient
import config


# Model yang dipakai untuk Groq
GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqClient(BaseLLMClient):
    """
    LLM client untuk Groq (Llama 3.3 70B).
    
    Handling khusus:
    - Structured output via instruksi eksplisit di prompt
    - Strip code fence sebelum parse JSON
    - Retry sekali jika parse pertama gagal
    - complete_vision() raise NotImplementedError (Groq tidak support vision)
    """

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY kosong. "
                "Pastikan sudah diisi di file .env"
            )
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def complete(
        self,
        prompt: str,
        system: str = "",
        output_schema: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        use_scorer: bool = False,  # diabaikan untuk Groq
    ) -> str | BaseModel:
        """
        Kirim prompt ke Groq dan dapatkan response.
        
        Flow sama dengan GeminiClient:
        1. Build messages (system + user)
        2. Call API
        3. Parse ke Pydantic jika output_schema diisi
        4. Retry sekali jika parse gagal
        """
        messages = self._build_messages(prompt, system, output_schema)

        # ── Call pertama ─────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API call gagal: {e}")

        # Jika tidak butuh structured output, return langsung
        if output_schema is None:
            return raw_text

        # ── Parse ke Pydantic ────────────────────────────────
        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError:
            # Parse gagal → retry sekali dengan instruksi lebih ketat
            return self._retry_with_strict_json(
                prompt, system, output_schema, temperature
            )

    def complete_vision(
        self,
        prompt: str,
        image_path: str,
        system: str = "",
    ) -> str:
        """
        Groq tidak mendukung vision/multimodal input.
        
        Raises
        ------
        NotImplementedError
            Selalu — Groq tidak punya kemampuan vision.
            Gunakan GeminiClient atau ClaudeClient untuk vision tasks.
        """
        raise NotImplementedError(
            "GroqClient tidak mendukung vision. "
            "Untuk image captioning, gunakan GeminiClient (LLM_VISION=gemini) "
            "atau ClaudeClient (LLM_VISION=claude)."
        )

    # ── Private helpers ───────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        system: str,
        output_schema: Type[BaseModel] | None,
    ) -> list[dict]:
        """
        Susun messages dalam format yang dibutuhkan Groq API.
        Groq menggunakan format OpenAI-compatible messages.
        """
        messages = []

        # System message
        system_content = system or "Kamu adalah asisten yang membantu."
        if output_schema is not None:
            schema_json = output_schema.model_json_schema()
            system_content += (
                f"\n\nPENTING: Respond HANYA dengan JSON yang valid. "
                f"Tidak boleh ada teks sebelum atau sesudah JSON. "
                f"Tidak boleh ada markdown code fence (```). "
                f"JSON harus sesuai schema ini:\n{schema_json}"
            )

        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        return messages

    def _retry_with_strict_json(
        self,
        prompt: str,
        system: str,
        output_schema: Type[BaseModel],
        temperature: float,
    ) -> BaseModel:
        """
        Retry dengan instruksi JSON yang lebih ketat.
        Dipanggil hanya jika parse pertama gagal.
        """
        time.sleep(1)

        strict_prompt = (
            prompt +
            "\n\nPERINGATAN KRITIS: Response sebelumnya tidak valid. "
            "Kali ini HANYA output JSON mentah. "
            "Mulai langsung dengan { dan akhiri dengan }. "
            "Tidak ada kata pengantar, tidak ada penjelasan."
        )

        messages = self._build_messages(strict_prompt, system, output_schema)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq retry API call gagal: {e}")

        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError as e:
            raise ValueError(
                f"Groq gagal menghasilkan JSON valid setelah 2 percobaan.\n"
                f"Schema: {output_schema.__name__}\n"
                f"Error terakhir: {e}"
            )