"""
config.py — Satu-satunya sumber konfigurasi Examind.

Aturan penting:
- Semua variabel dari .env HANYA dibaca di file ini.
- Modul lain (agent, pipeline, kb, dll.) TIDAK BOLEH import dotenv atau akses os.environ langsung.
- Semua modul lain harus import dari sini: from config import settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env dari root project
load_dotenv()


# ══════════════════════════════════════════════════════════════
# LLM PROVIDER
# ══════════════════════════════════════════════════════════════

LLM_MAIN: str = os.getenv("LLM_MAIN", "gemini")       # gemini | claude
LLM_SCORER: str = os.getenv("LLM_SCORER", "groq")     # groq | claude
LLM_VISION: str = os.getenv("LLM_VISION", "gemini")   # gemini | claude


# ══════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


# ══════════════════════════════════════════════════════════════
# PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════

VALIDATOR_THRESHOLD: int = int(os.getenv("VALIDATOR_THRESHOLD", "75"))
MAX_RETRY: int = int(os.getenv("MAX_RETRY", "3"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
DUPLICATE_SIMILARITY_THRESHOLD: float = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.85"))


# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent

KB_PATH: Path = BASE_DIR / os.getenv("KB_PATH", "knowledge_base")
ASSETS_PATH: Path = BASE_DIR / os.getenv("ASSETS_PATH", "assets")
CHROMA_PATH: Path = BASE_DIR / os.getenv("CHROMA_PATH", "db/chroma")
SQLITE_PATH: Path = BASE_DIR / os.getenv("SQLITE_PATH", "db/sqlite/soal_output.db")


# ══════════════════════════════════════════════════════════════
# CHROMADB COLLECTIONS
# ══════════════════════════════════════════════════════════════
# Pattern key: "{mapel}_{jenjang}_{tipe}"
# Untuk menambah jenjang/mapel baru: cukup tambah entry di dict ini.
# Core pipeline tidak perlu diubah.

CHROMA_COLLECTIONS: dict[str, str] = {
    # ── Phase 1 — SD ──────────────────────────────────────────
    "matematika_sd_materi":       "matematika_sd_materi",
    "matematika_sd_contoh_soal":  "matematika_sd_contoh_soal",
    "ips_sd_materi":              "ips_sd_materi",
    "ips_sd_contoh_soal":         "ips_sd_contoh_soal",

    # ── Phase 2 — SMP (aktifkan saat ekspansi) ────────────────
    # "matematika_smp_materi":      "matematika_smp_materi",
    # "matematika_smp_contoh_soal": "matematika_smp_contoh_soal",
    # "ips_smp_materi":             "ips_smp_materi",
    # "ips_smp_contoh_soal":        "ips_smp_contoh_soal",

    # ── Phase 3 — SMA & IPA (aktifkan saat ekspansi) ──────────
    # "matematika_sma_materi":      "matematika_sma_materi",
    # "matematika_sma_contoh_soal": "matematika_sma_contoh_soal",
    # "ips_sma_materi":             "ips_sma_materi",
    # "ips_sma_contoh_soal":        "ips_sma_contoh_soal",
    # "ipa_sd_materi":              "ipa_sd_materi",
    # "ipa_sd_contoh_soal":         "ipa_sd_contoh_soal",
}

# Collection khusus untuk menyimpan output soal yang sudah di-generate
# Dipakai Duplicate Checker untuk cross-check soal baru vs soal lama
CHROMA_SOAL_OUTPUT_COLLECTIONS: dict[str, str] = {
    "soal_output_matematika_sd": "soal_output_matematika_sd",
    "soal_output_ips_sd":        "soal_output_ips_sd",
}


# ══════════════════════════════════════════════════════════════
# BLOOM DISTRIBUTION PER MODE
# ══════════════════════════════════════════════════════════════
# Ini adalah satu-satunya sumber kebenaran distribusi Bloom.
# Strategic Planner membaca dari sini — tidak hardcode di prompt.
#
# Format: { level_bloom: persentase_desimal }
# Contoh: 0.10 berarti 10% soal harus di level ini.
# Catatan: jumlah total per mode = 1.0 (100%)

BLOOM_DISTRIBUTION: dict[str, dict[str, float]] = {

    # ── MODE OLIMPIADE ─────────────────────────────────────────
    "osnk": {
        "C1": 0.07,
        "C2": 0.27,
        "C3": 0.40,
        "C4": 0.20,
        "C5": 0.04,
        "C6": 0.02,
    },
    "osnp": {
        "C1": 0.00,
        "C2": 0.07,
        "C3": 0.20,
        "C4": 0.47,
        "C5": 0.20,
        "C6": 0.06,
    },
    "osn": {
        "C1": 0.00,
        "C2": 0.00,
        "C3": 0.10,
        "C4": 0.30,
        "C5": 0.35,
        "C6": 0.25,
    },

    # ── MODE SEKOLAH REGULER ───────────────────────────────────
    "latihan": {
        "C1": 0.10,
        "C2": 0.30,
        "C3": 0.40,
        "C4": 0.20,
        "C5": 0.00,
        "C6": 0.00,
    },
    "uh": {
        "C1": 0.05,
        "C2": 0.25,
        "C3": 0.40,
        "C4": 0.25,
        "C5": 0.05,
        "C6": 0.00,
    },
    "pts": {
        "C1": 0.05,
        "C2": 0.20,
        "C3": 0.35,
        "C4": 0.30,
        "C5": 0.10,
        "C6": 0.00,
    },
    "pas": {
        "C1": 0.00,
        "C2": 0.15,
        "C3": 0.30,
        "C4": 0.30,
        "C5": 0.20,
        "C6": 0.05,
    },
}


# ══════════════════════════════════════════════════════════════
# MODE CONTEXT
# ══════════════════════════════════════════════════════════════
# Memetakan setiap mode ke konteks strategi yang dipakai Planner.
# "olimpiade" → prompt template olimpiade
# "sekolah"   → prompt template sekolah reguler

MODE_CONTEXT: dict[str, str] = {
    "osnk":    "olimpiade",
    "osnp":    "olimpiade",
    "osn":     "olimpiade",
    "latihan": "sekolah",
    "uh":      "sekolah",
    "pts":     "sekolah",
    "pas":     "sekolah",
}


# ══════════════════════════════════════════════════════════════
# VALIDASI SAAT STARTUP
# ══════════════════════════════════════════════════════════════

def validate_config() -> None:
    """
    Validasi konfigurasi saat startup.
    Dipanggil di bagian bawah file ini agar error terdeteksi
    sejak awal, bukan saat agent pertama kali dipanggil.
    """
    errors = []

    # Cek provider valid
    valid_main = {"gemini", "claude"}
    valid_scorer = {"groq", "claude"}
    valid_vision = {"gemini", "claude"}

    if LLM_MAIN not in valid_main:
        errors.append(f"LLM_MAIN='{LLM_MAIN}' tidak valid. Pilihan: {valid_main}")
    if LLM_SCORER not in valid_scorer:
        errors.append(f"LLM_SCORER='{LLM_SCORER}' tidak valid. Pilihan: {valid_scorer}")
    if LLM_VISION not in valid_vision:
        errors.append(f"LLM_VISION='{LLM_VISION}' tidak valid. Pilihan: {valid_vision}")

    # Cek API key yang dibutuhkan tersedia
    if LLM_MAIN == "gemini" and not GEMINI_API_KEY:
        errors.append("LLM_MAIN=gemini tapi GEMINI_API_KEY kosong di .env")
    if LLM_MAIN == "claude" and not ANTHROPIC_API_KEY:
        errors.append("LLM_MAIN=claude tapi ANTHROPIC_API_KEY kosong di .env")
    if LLM_SCORER == "groq" and not GROQ_API_KEY:
        errors.append("LLM_SCORER=groq tapi GROQ_API_KEY kosong di .env")
    if LLM_SCORER == "claude" and not ANTHROPIC_API_KEY:
        errors.append("LLM_SCORER=claude tapi ANTHROPIC_API_KEY kosong di .env")
    if LLM_VISION == "gemini" and not GEMINI_API_KEY:
        errors.append("LLM_VISION=gemini tapi GEMINI_API_KEY kosong di .env")

    # Cek nilai numerik masuk akal
    if not (0 <= VALIDATOR_THRESHOLD <= 100):
        errors.append(f"VALIDATOR_THRESHOLD={VALIDATOR_THRESHOLD} harus antara 0-100")
    if not (1 <= MAX_RETRY <= 10):
        errors.append(f"MAX_RETRY={MAX_RETRY} harus antara 1-10")
    if not (1 <= BATCH_SIZE <= 30):
        errors.append(f"BATCH_SIZE={BATCH_SIZE} harus antara 1-30")
    if not (0.0 <= DUPLICATE_SIMILARITY_THRESHOLD <= 1.0):
        errors.append(f"DUPLICATE_SIMILARITY_THRESHOLD={DUPLICATE_SIMILARITY_THRESHOLD} harus antara 0.0-1.0")

    if errors:
        print("\n❌ CONFIG ERROR — Perbaiki .env sebelum menjalankan aplikasi:\n")
        for err in errors:
            print(f"  • {err}")
        print()
        raise ValueError(f"{len(errors)} error konfigurasi ditemukan. Lihat detail di atas.")

    print("✅ Config valid.")
    print(f"   LLM_MAIN    : {LLM_MAIN}")
    print(f"   LLM_SCORER  : {LLM_SCORER}")
    print(f"   LLM_VISION  : {LLM_VISION}")
    print(f"   THRESHOLD   : {VALIDATOR_THRESHOLD}")
    print(f"   MAX_RETRY   : {MAX_RETRY}")
    print(f"   BATCH_SIZE  : {BATCH_SIZE}")
    print(f"   Collections : {len(CHROMA_COLLECTIONS)} aktif")


# ── Jalankan validasi saat file ini diimport ─────────────────
if __name__ == "__main__":
    validate_config()