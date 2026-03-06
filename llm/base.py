"""
llm/base.py — Abstract base class untuk semua LLM client.

Ini adalah "kontrak" yang harus dipenuhi setiap client (Gemini, Groq, Claude).
Semua agent hanya berinteraksi dengan interface ini — tidak pernah langsung
ke client spesifik.

Konsep Abstract Class:
- Tidak bisa di-instantiate langsung
- Subclass WAJIB mengimplementasikan semua method yang ada @abstractmethod
- Kalau subclass tidak implement, Python akan error saat instantiate
"""

from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel


class BaseLLMClient(ABC):
    """
    Interface seragam untuk semua LLM provider.
    
    Setiap provider (Gemini, Groq, Claude) harus mengimplementasikan
    dua method ini dengan perilaku yang konsisten.
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        output_schema: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        use_scorer: bool = False,
    ) -> str | BaseModel:
        """
        Kirim prompt ke LLM dan dapatkan response.

        Parameters
        ----------
        prompt : str
            Pesan utama yang dikirim ke LLM (user message).
        system : str
            System prompt — instruksi konteks untuk LLM.
            Contoh: "Kamu adalah validator soal matematika."
        output_schema : Type[BaseModel] | None
            Jika diisi, client akan parse response ke instance Pydantic.
            Jika None, return raw string.
        temperature : float
            Kreativitas output. 0.0 = deterministik, 1.0 = kreatif.
            Default 0.7 untuk keseimbangan.
        use_scorer : bool
            Khusus Claude: jika True, pakai model Haiku (lebih murah).
            Jika False, pakai model Sonnet.
            Untuk Gemini/Groq: parameter ini diabaikan.

        Returns
        -------
        str
            Raw string response jika output_schema=None.
        BaseModel
            Instance Pydantic jika output_schema diisi.

        Raises
        ------
        ValueError
            Jika response tidak bisa di-parse ke output_schema setelah retry.
        RuntimeError
            Jika API call gagal karena network error atau rate limit.
        """
        pass

    @abstractmethod
    def complete_vision(
        self,
        prompt: str,
        image_path: str,
        system: str = "",
    ) -> str:
        """
        Kirim prompt + gambar ke LLM untuk image captioning.
        
        Dipakai saat KB ingestion untuk mengekstrak deskripsi
        teknis dari gambar dalam dokumen materi.

        Parameters
        ----------
        prompt : str
            Instruksi captioning — apa yang harus dijelaskan dari gambar.
        image_path : str
            Path ke file gambar (jpg, png, dll).
        system : str
            System prompt konteks captioning.

        Returns
        -------
        str
            Deskripsi teknis gambar dalam teks.

        Raises
        ------
        NotImplementedError
            Jika provider tidak support vision (contoh: Groq).
        RuntimeError
            Jika API call gagal.
        """
        pass

    def _strip_code_fence(self, text: str) -> str:
        """
        Helper — bersihkan markdown code fence dari response LLM.
        
        LLM sering membungkus JSON dengan ```json ... ``` 
        yang menyebabkan JSON parsing gagal.
        
        Contoh input  : ```json\n{"key": "value"}\n```
        Contoh output : {"key": "value"}
        """
        text = text.strip()
        # Hapus opening fence (```json atau ``` saja)
        if text.startswith("```"):
            # Cari baris pertama dan hapus
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
        # Hapus closing fence
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _parse_to_schema(
        self,
        raw_text: str,
        output_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Helper — parse raw string ke Pydantic model.
        
        Sudah include strip code fence sebelum parse.
        Dipanggil oleh subclass setelah dapat response dari API.

        Raises
        ------
        ValueError
            Jika parsing tetap gagal setelah strip code fence.
        """
        import json

        cleaned = self._strip_code_fence(raw_text)
        try:
            data = json.loads(cleaned)
            return output_schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Gagal parse response ke {output_schema.__name__}.\n"
                f"Response asli:\n{raw_text}\n"
                f"Error: {e}"
            )