"""
llm/claude_client.py — Implementasi LLM client untuk Anthropic Claude.

Dipakai saat LLM_MAIN=claude dan/atau LLM_SCORER=claude (mode production).

Dua model internal:
- claude-sonnet-4-5  → untuk Generator, Planner, Answer Key (use_scorer=False)
- claude-haiku-4-5-20251001   → untuk Validator, Calibrator (use_scorer=True, lebih murah)

Harga (estimasi Maret 2026):
- Sonnet: $3.00/1M input · $15.00/1M output
- Haiku:  $0.80/1M input · $4.00/1M output

NOTE: Client ini sudah disiapkan tapi belum aktif di Phase 1.
Aktifkan dengan set LLM_MAIN=claude dan/atau LLM_SCORER=claude di .env
setelah sistem terbukti stabil dengan Gemini/Groq.
"""

import time
import anthropic
from typing import Type
from pydantic import BaseModel

from llm.base import BaseLLMClient
import config


# Model yang dipakai untuk Claude
CLAUDE_MODEL_MAIN = "claude-sonnet-4-5"         # Untuk Generator, Planner
CLAUDE_MODEL_SCORER = "claude-haiku-4-5-20251001"       # Untuk Validator, Calibrator


class ClaudeClient(BaseLLMClient):
    """
    LLM client untuk Anthropic Claude.
    
    Dua model internal:
    - model_main (Sonnet): reasoning kompleks, generate soal
    - model_scorer (Haiku): scoring & validasi, lebih murah & cepat
    
    Parameter use_scorer=True di complete() menentukan model mana yang dipakai.
    """

    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY kosong. "
                "Pastikan sudah diisi di file .env"
            )
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def complete(
        self,
        prompt: str,
        system: str = "",
        output_schema: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        use_scorer: bool = False,
    ) -> str | BaseModel:
        """
        Kirim prompt ke Claude dan dapatkan response.
        
        use_scorer=True  → pakai Haiku (Validator, Calibrator)
        use_scorer=False → pakai Sonnet (Generator, Planner)
        """
        model = CLAUDE_MODEL_SCORER if use_scorer else CLAUDE_MODEL_MAIN

        # Susun system prompt dengan instruksi JSON jika perlu
        system_content = self._build_system(system, output_schema)

        # ── Call pertama ─────────────────────────────────────
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                system=system_content,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude API call gagal (model={model}): {e}")

        # Jika tidak butuh structured output, return langsung
        if output_schema is None:
            return raw_text

        # ── Parse ke Pydantic ────────────────────────────────
        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError:
            return self._retry_with_strict_json(
                prompt, system, output_schema, temperature, model
            )

    def complete_vision(
        self,
        prompt: str,
        image_path: str,
        system: str = "",
    ) -> str:
        """
        Kirim prompt + gambar ke Claude untuk image captioning.
        Menggunakan model Sonnet (lebih capable untuk vision).
        """
        import base64
        from pathlib import Path

        if not Path(image_path).exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

        # Baca dan encode gambar ke base64
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Deteksi media type dari ekstensi file
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL_MAIN,
                max_tokens=1024,
                system=system or "Kamu adalah asisten yang mendeskripsikan gambar secara teknis.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude Vision API call gagal: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _build_system(
        self,
        system: str,
        output_schema: Type[BaseModel] | None,
    ) -> str:
        """Susun system prompt dengan instruksi JSON jika perlu."""
        content = system or "Kamu adalah asisten yang membantu."

        if output_schema is not None:
            schema_json = output_schema.model_json_schema()
            content += (
                f"\n\nPENTING: Respond HANYA dengan JSON yang valid. "
                f"Tidak boleh ada teks sebelum atau sesudah JSON. "
                f"Tidak boleh ada markdown code fence (```). "
                f"JSON harus sesuai schema ini:\n{schema_json}"
            )
        return content

    def _retry_with_strict_json(
        self,
        prompt: str,
        system: str,
        output_schema: Type[BaseModel],
        temperature: float,
        model: str,
    ) -> BaseModel:
        """Retry dengan instruksi JSON yang lebih ketat."""
        time.sleep(1)

        system_content = self._build_system(system, output_schema)
        strict_prompt = (
            prompt +
            "\n\nPERINGATAN KRITIS: Response sebelumnya tidak valid. "
            "Kali ini HANYA output JSON mentah. "
            "Mulai langsung dengan { dan akhiri dengan }."
        )

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.1,
                system=system_content,
                messages=[{"role": "user", "content": strict_prompt}],
            )
            raw_text = response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude retry API call gagal: {e}")

        try:
            return self._parse_to_schema(raw_text, output_schema)
        except ValueError as e:
            raise ValueError(
                f"Claude gagal menghasilkan JSON valid setelah 2 percobaan.\n"
                f"Schema: {output_schema.__name__}\n"
                f"Error terakhir: {e}"
            )