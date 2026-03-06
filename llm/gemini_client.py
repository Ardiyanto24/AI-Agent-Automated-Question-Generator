"""
llm/gemini_client.py — Implementasi LLM client untuk Google Gemini.

Dipakai saat LLM_MAIN=gemini dan LLM_VISION=gemini (mode development).
Model yang dipakai: gemini-2.0-flash.

Menggunakan package google-genai (bukan google-generativeai yang deprecated).

Free tier limits:
- 15 RPM (request per menit)
- 1.500 RPD (request per hari)
- 1.000.000 TPM (token per menit)
"""

import time
from google import genai
from google.genai import types
from pathlib import Path
from typing import Type
from pydantic import BaseModel

from llm.base import BaseLLMClient
import config


# Model yang dipakai untuk Gemini
GEMINI_MODEL = "gemini-2.0-flash"


class GeminiClient(BaseLLMClient):
    """
    LLM client untuk Google Gemini.

    Handling khusus:
    - Structured output via instruksi eksplisit di prompt
    - Strip code fence sebelum parse JSON
    - Retry sekali jika parse pertama gagal
    
    Menggunakan package google-genai (bukan google-generativeai yang deprecated).
    """

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY kosong. "
                "Pastikan sudah diisi di file .env"
            )
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def complete(
        self,
        prompt: str,
        system: str = "",
        output_schema: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        use_scorer: bool = False,
    ) -> str | BaseModel:
        """
        Kirim prompt ke Gemini dan dapatkan response.
        """
        full_prompt = self._build_prompt(prompt, system, output_schema)

        config_generate = types.GenerateContentConfig(
            temperature=temperature,
        )

        # ── Call pertama ─────────────────────────────────────
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=config_generate,
            )
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call gagal: {e}")

        if output_schema is None:
            return raw_text

        # ── Parse ke Pydantic ────────────────────────────────
        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError:
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
        Kirim prompt + gambar ke Gemini untuk image captioning.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

        # Baca gambar sebagai bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    full_prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini Vision API call gagal: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _build_prompt(
        self,
        prompt: str,
        system: str,
        output_schema: Type[BaseModel] | None,
    ) -> str:
        parts = []
        if system:
            parts.append(f"INSTRUKSI SISTEM:\n{system}")
        parts.append(prompt)
        if output_schema is not None:
            schema_json = output_schema.model_json_schema()
            parts.append(
                f"\nPENTING: Respond HANYA dengan JSON yang valid. "
                f"Tidak boleh ada teks sebelum atau sesudah JSON. "
                f"Tidak boleh ada markdown code fence (```). "
                f"JSON harus sesuai schema ini:\n{schema_json}"
            )
        return "\n\n".join(parts)

    def _retry_with_strict_json(
        self,
        prompt: str,
        system: str,
        output_schema: Type[BaseModel],
        temperature: float,
    ) -> BaseModel:
        time.sleep(1)

        strict_suffix = (
            "\n\nPERINGATAN KRITIS: Response sebelumnya tidak valid. "
            "Kali ini HANYA output JSON mentah. "
            "Mulai langsung dengan { dan akhiri dengan }. "
            "Tidak ada kata pengantar, tidak ada penjelasan, "
            "tidak ada code fence."
        )

        full_prompt = self._build_prompt(
            prompt + strict_suffix, system, output_schema
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini retry API call gagal: {e}")

        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError as e:
            raise ValueError(
                f"Gemini gagal menghasilkan JSON valid setelah 2 percobaan.\n"
                f"Schema: {output_schema.__name__}\n"
                f"Error terakhir: {e}"
            )