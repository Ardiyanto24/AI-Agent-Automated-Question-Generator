"""
scripts/test_llm_layer.py — Test manual LLM Abstraction Layer.

Jalankan dengan:
    python scripts/test_llm_layer.py

Script ini memverifikasi semua Definisi Done dari Step 1.2:
1. get_llm_main() return GeminiClient saat LLM_MAIN=gemini
2. get_llm_scorer() return GroqClient saat LLM_SCORER=groq
3. GeminiClient.complete() bisa return raw string
4. GeminiClient.complete() bisa return Pydantic instance
5. GroqClient.complete_vision() raise NotImplementedError
6. Kirim prompt sederhana ke Gemini dan Groq — response berhasil diterima
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from llm import get_llm_main, get_llm_scorer, get_llm_vision
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient

# Schema sederhana untuk test structured output
class TestSchema(BaseModel):
    jawaban: str
    angka: int


def test_factory_functions():
    """Test 1 & 2: Factory function return client yang benar."""
    print("\n── Test 1: Factory Functions ─────────────────────")

    main_client = get_llm_main()
    scorer_client = get_llm_scorer()
    vision_client = get_llm_vision()

    assert isinstance(main_client, GeminiClient), \
        f"Expected GeminiClient, got {type(main_client)}"
    print(f"  ✅ get_llm_main()   → {type(main_client).__name__}")

    assert isinstance(scorer_client, GroqClient), \
        f"Expected GroqClient, got {type(scorer_client)}"
    print(f"  ✅ get_llm_scorer() → {type(scorer_client).__name__}")

    assert isinstance(vision_client, GeminiClient), \
        f"Expected GeminiClient, got {type(vision_client)}"
    print(f"  ✅ get_llm_vision() → {type(vision_client).__name__}")


def test_gemini_raw_string():
    """Test 3: GeminiClient.complete() return raw string."""
    print("\n── Test 2: Gemini Raw String Response ────────────")

    client = get_llm_main()
    response = client.complete(
        prompt="Jawab dengan satu kalimat singkat: Apa ibu kota Indonesia?",
        system="Kamu adalah asisten geografi.",
    )

    assert isinstance(response, str), \
        f"Expected str, got {type(response)}"
    assert len(response) > 0, "Response kosong"
    print(f"  ✅ Response: {response[:100]}...")


def test_gemini_structured_output():
    """Test 4: GeminiClient.complete() return Pydantic instance."""
    print("\n── Test 3: Gemini Structured Output ──────────────")

    client = get_llm_main()
    response = client.complete(
        prompt="Berikan jawaban: ibu kota Indonesia adalah apa? Dan berapa jumlah provinsi Indonesia?",
        system="Kamu adalah asisten geografi.",
        output_schema=TestSchema,
        temperature=0.1,
    )

    assert isinstance(response, TestSchema), \
        f"Expected TestSchema, got {type(response)}"
    assert isinstance(response.jawaban, str)
    assert isinstance(response.angka, int)
    print(f"  ✅ Parsed schema:")
    print(f"     jawaban : {response.jawaban}")
    print(f"     angka   : {response.angka}")


def test_groq_raw_string():
    """Test 5: GroqClient.complete() return raw string."""
    print("\n── Test 4: Groq Raw String Response ──────────────")

    client = get_llm_scorer()
    response = client.complete(
        prompt="Jawab dengan satu kalimat: Apa itu Taksonomi Bloom?",
        system="Kamu adalah pakar pendidikan.",
    )

    assert isinstance(response, str), \
        f"Expected str, got {type(response)}"
    assert len(response) > 0, "Response kosong"
    print(f"  ✅ Response: {response[:100]}...")


def test_groq_vision_not_implemented():
    """Test 6: GroqClient.complete_vision() raise NotImplementedError."""
    print("\n── Test 5: Groq Vision NotImplementedError ───────")

    client = get_llm_scorer()
    try:
        client.complete_vision(
            prompt="Deskripsikan gambar ini.",
            image_path="dummy.png",
        )
        print("  ❌ GAGAL: Seharusnya raise NotImplementedError")
    except NotImplementedError as e:
        print(f"  ✅ NotImplementedError raised dengan benar")
        print(f"     Pesan: {e}")


def main():
    print("=" * 55)
    print("  TEST LLM ABSTRACTION LAYER — Step 1.2")
    print("=" * 55)

    tests = [
        test_factory_functions,
        test_gemini_raw_string,
        test_gemini_structured_output,
        test_groq_raw_string,
        test_groq_vision_not_implemented,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ GAGAL: {e}")
            failed += 1

    print("\n" + "=" * 55)
    print(f"  HASIL: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ Step 1.2 — Definisi Done TERCAPAI")
    else:
        print("  ❌ Ada test yang gagal — cek error di atas")
    print("=" * 55)


if __name__ == "__main__":
    main()