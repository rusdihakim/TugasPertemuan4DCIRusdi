from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

#KONFIGURASI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "data/Lora.pdf")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "vectorstore")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
# Dipakai otomatis bila kuota harian model utama habis
GEMINI_CADANGAN = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODEL_CADANGAN", "gemini-2.0-flash,gemini-flash-lite-latest"
    ).split(",")
    if m.strip()
]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "8"))              # potongan dari pencarian kemiripan
MMR_K = int(os.getenv("MMR_K", "6"))              # tambahan potongan dari MMR
FETCH_K = int(os.getenv("FETCH_K", "30"))         # kandidat awal yang diseleksi MMR
MAX_KONTEKS = int(os.getenv("MAX_KONTEKS", "14"))  # batas potongan yang dikirim ke LLM
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

#proses embedding dikirim per batch lalu dijeda.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "90"))
EMBED_JEDA = float(os.getenv("EMBED_JEDA", "62"))

LAMBDA_MMR = 0.5          # 1.0 = murni relevansi, 0 = murni keberagaman
VERSI_PIPELINE = "v2"     # diubah bila cara pengolahan teks berubah -> index dibangun ulang

KALIMAT_TIDAK_TAHU_ID = "Saya tidak tahu. Informasi tersebut tidak saya temukan di dalam dokumen."
KALIMAT_TIDAK_TAHU_EN = "I don't know. I could not find that information in the document."

# Fungsi opsional untuk melaporkan kemajuan proses ke antarmuka (mis. terminal)
Pelapor = Callable[[str], None]


class KesalahanChatbot(Exception):
    """Kesalahan yang pesannya layak ditampilkan langsung kepada pengguna."""


def _lapor(pelapor: Pelapor | None, pesan: str) -> None:
    if pelapor:
        pelapor(pesan)


#MEMBACA PDF

def bersihkan_teks(teks: str) -> str:
    """Merapikan hasil ekstraksi PDF: titik penuntun, spasi dan baris berlebih."""
    teks = re.sub(r"\.{4,}", " ", teks)
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def halaman_daftar_isi(teks_asli: str) -> bool:
    """Deteksi halaman daftar isi dari rapatnya titik penuntun ("Bab 1 ..... 12").

    Pada datasheet ini rasio titik halaman daftar isi > 0,6 sedangkan halaman
    berisi konten < 0,04. Halaman daftar isi dibuang agar tidak mengotori hasil
    pencarian dan agar rujukan nomor halaman tidak meleset.
    """
    if not teks_asli:
        return True
    return teks_asli.count(".") / max(len(teks_asli), 1) > 0.30


def muat_pdf(path: Path, pelapor: Pelapor | None = None) -> List[Document]:
    """Membaca PDF per halaman. Metadata `halaman` = nomor halaman (mulai dari 1)."""
    if not path.exists():
        raise KesalahanChatbot(f"File PDF tidak ditemukan: {path}")

    halaman = PyPDFLoader(str(path)).load()

    hasil: List[Document] = []
    dilewati = 0
    for doc in halaman:
        if halaman_daftar_isi(doc.page_content):   # diperiksa SEBELUM dibersihkan
            dilewati += 1
            continue
        isi = bersihkan_teks(doc.page_content)
        if len(isi) < 50:                          # halaman kosong / hanya gambar
            continue
        doc.page_content = isi
        doc.metadata["halaman"] = doc.metadata.get("page", 0) + 1
        doc.metadata["sumber"] = path.name
        hasil.append(doc)

    _lapor(pelapor, f"Halaman terbaca : {len(hasil)} dari {len(halaman)} "
                    f"({dilewati} halaman daftar isi dilewati)")
    return hasil


#CHUNKING

def potong_teks(dokumen: List[Document], pelapor: Pelapor | None = None) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    potongan = [
        d for d in splitter.split_documents(dokumen)
        if len(d.page_content.strip()) >= 80        # buang sisa header/footer
    ]
    for i, d in enumerate(potongan):
        d.metadata["chunk_id"] = i

    _lapor(pelapor, f"Chunk dibuat    : {len(potongan)} "
                    f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return potongan


#EMBEDDING & VECTOR STORE

def buat_embedding():
    """Model embedding: mengubah potongan teks menjadi vektor angka."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith(("isi_", "masukkan_")):
        raise KesalahanChatbot(
            "GOOGLE_API_KEY belum diisi di file .env.\n"
            "Ambil API key gratis di https://aistudio.google.com/app/apikey"
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


def jeda_dari_pesan_galat(pesan: str, bawaan: float = 30.0) -> float:
    """Ambil saran waktu tunggu dari pesan error 429 Google (retryDelay / retry in Xs)."""
    cocok = (re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", pesan)
             or re.search(r"retry in (\d+(?:\.\d+)?)s", pesan))
    return min(float(cocok.group(1)) + 5 if cocok else bawaan, 120)


def sidik_jari_konfigurasi() -> str:
    """Sidik jari PDF + parameter. Bila berubah, index dibangun ulang otomatis."""
    h = hashlib.md5()
    h.update(PDF_PATH.read_bytes() if PDF_PATH.exists() else b"")
    h.update(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}|{VERSI_PIPELINE}".encode())
    return h.hexdigest()


def bangun_index(embedding, pelapor: Pelapor | None = None) -> FAISS:
    """Bangun vector store dari PDF lalu simpan ke disk."""
    potongan = potong_teks(muat_pdf(PDF_PATH, pelapor), pelapor)
    total_batch = (len(potongan) + EMBED_BATCH - 1) // EMBED_BATCH

    store: FAISS | None = None
    for nomor, i in enumerate(range(0, len(potongan), EMBED_BATCH), 1):
        bagian = potongan[i:i + EMBED_BATCH]
        _lapor(pelapor, f"Embedding batch {nomor}/{total_batch} ({len(bagian)} chunk)...")

        for percobaan in range(5):
            try:
                if store is None:
                    store = FAISS.from_documents(bagian, embedding)
                else:
                    store.add_documents(bagian)
                break
            except Exception as e:                  # batas kuota / gangguan jaringan
                if percobaan == 4:
                    raise KesalahanChatbot(f"Gagal membuat embedding: {e}") from e
                jeda = jeda_dari_pesan_galat(str(e), bawaan=30 * (percobaan + 1))
                _lapor(pelapor, f"Batas kuota API, menunggu {jeda:.0f} detik lalu mencoba lagi...")
                time.sleep(jeda)

        if i + EMBED_BATCH < len(potongan):
            _lapor(pelapor, f"Menjaga kuota, jeda {EMBED_JEDA:.0f} detik...")
            time.sleep(EMBED_JEDA)

    if store is None:
        raise KesalahanChatbot("Tidak ada teks yang bisa diindeks dari PDF tersebut.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(
            {
                "fingerprint": sidik_jari_konfigurasi(),
                "pdf": PDF_PATH.name,
                "jumlah_chunk": len(potongan),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _lapor(pelapor, f"Vector store tersimpan di {INDEX_DIR.name}/")
    return store


def muat_atau_bangun_index(paksa_bangun: bool = False, pelapor: Pelapor | None = None):
    """Pakai index tersimpan bila masih cocok, kalau tidak bangun ulang.

    Mengembalikan (vector store, embedding, keterangan index).
    """
    embedding = buat_embedding()
    meta_file = INDEX_DIR / "meta.json"

    if not paksa_bangun and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == sidik_jari_konfigurasi():
                store = FAISS.load_local(
                    str(INDEX_DIR), embedding, allow_dangerous_deserialization=True
                )
                _lapor(pelapor, f"Index dimuat dari cache ({meta.get('jumlah_chunk')} chunk)")
                return store, embedding, meta
            _lapor(pelapor, "PDF/konfigurasi berubah -> index dibangun ulang.")
        except KesalahanChatbot:
            raise
        except Exception:
            _lapor(pelapor, "Cache index tidak terbaca -> index dibangun ulang.")

    store = bangun_index(embedding, pelapor)
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    return store, embedding, meta


#LLM

def buat_llm(nama_model: str | None = None):
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise KesalahanChatbot("DEEPSEEK_API_KEY belum diisi di file .env")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=nama_model or DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=TEMPERATURE,
        )

    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith(("isi_", "masukkan_")):
        raise KesalahanChatbot(
            "GOOGLE_API_KEY belum diisi di file .env.\n"
            "Ambil API key gratis di https://aistudio.google.com/app/apikey"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=nama_model or GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
    )


#PROMPT

PROMPT_JAWAB = ChatPromptTemplate.from_messages([
    (
        "system",
        "Kamu adalah asisten dokumen. Jawab pertanyaan HANYA berdasarkan KONTEKS "
        "potongan dokumen di bawah ini.\n\n"
        "ATURAN WAJIB:\n"
        "1. Jawab HANYA memakai informasi dari KONTEKS. Dilarang memakai pengetahuan "
        "umum di luar dokumen, dilarang menebak atau mengarang.\n"
        f"2. Jika informasi tidak ada di KONTEKS, jawab persis: \"{KALIMAT_TIDAK_TAHU_ID}\" "
        f"(atau \"{KALIMAT_TIDAK_TAHU_EN}\" bila pertanyaannya Bahasa Inggris), "
        "tanpa menambahkan penjelasan dari luar dokumen.\n"
        "3. Selalu cantumkan rujukan halaman dalam kurung siku, contoh: [Halaman 14]. "
        "Nomor halaman WAJIB disalin PERSIS dari label '=== Halaman N ===' pada potongan "
        "yang kamu pakai. DILARANG menyebut nomor halaman yang tidak muncul sebagai label "
        "potongan, dan dilarang memakai nomor halaman yang tertulis di dalam badan teks.\n"
        "4. Tulis SELURUH jawaban dalam bahasa berikut: {bahasa}. Dilarang mencampur "
        "dua bahasa dalam satu jawaban.\n"
        "5. Jawab ringkas, padat, dan faktual. Salin angka beserta satuannya persis "
        "seperti tertulis di dokumen. Gunakan poin-poin bila menjelaskan beberapa hal.\n\n"
        "RIWAYAT PERCAKAPAN (untuk memahami pertanyaan lanjutan):\n{riwayat}\n\n"
        "KONTEKS DOKUMEN:\n{konteks}",
    ),
    ("human", "{pertanyaan}"),
])

PROMPT_RINGKAS_PERTANYAAN = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ubah pertanyaan lanjutan menjadi satu pertanyaan yang berdiri sendiri "
        "(lengkap tanpa perlu riwayat). Balas HANYA pertanyaannya saja. Jika "
        "pertanyaan sudah berdiri sendiri, salin apa adanya.\n\nRIWAYAT:\n{riwayat}",
    ),
    ("human", "{pertanyaan}"),
])


#UTILITAS JAWABAN

KATA_INDONESIA = {
    "apa", "apakah", "berapa", "berapakah", "bagaimana", "siapa", "kapan", "mengapa",
    "kenapa", "yang", "dan", "atau", "dari", "untuk", "dengan", "pada", "ini", "itu",
    "adalah", "bisa", "dapat", "tolong", "jelaskan", "sebutkan", "saja", "dalam",
    "tentang", "digunakan", "mode", "chip",
}


def deteksi_bahasa(teks: str) -> str:
    """Tentukan bahasa pertanyaan agar jawaban tidak tercampur dua bahasa."""
    kata = set(re.findall(r"[a-zA-Z]+", teks.lower()))
    return "Bahasa Indonesia" if kata & KATA_INDONESIA else "English"


def format_konteks(dokumen: List[Document]) -> str:
    """Susun potongan dokumen menjadi konteks berlabel nomor halaman."""
    if not dokumen:
        return "(tidak ada potongan dokumen yang relevan)"
    return "\n\n".join(
        f"=== Halaman {d.metadata.get('halaman', '?')} === "
        f"(potongan {i}, sumber: {d.metadata.get('sumber', '-')})\n"
        f"{d.page_content}\n"
        f"=== akhir potongan Halaman {d.metadata.get('halaman', '?')} ==="
        for i, d in enumerate(dokumen, 1)
    )


def periksa_sitasi(jawaban: str, dokumen: List[Document]) -> List[str]:
    """Pastikan [Halaman N] pada jawaban berasal dari potongan yang benar-benar
    diambil retriever, bukan nomor halaman karangan LLM."""
    tersedia = {str(d.metadata.get("halaman")) for d in dokumen}
    disebut = re.findall(r"\[?Halaman\s+(\d+)", jawaban)
    return sorted({h for h in disebut if h not in tersedia}, key=int)


def menjawab_tidak_tahu(jawaban: str) -> bool:
    """True bila seluruh jawaban adalah pernyataan 'tidak tahu'."""
    awal = jawaban.lower().lstrip("*_# ")
    return awal.startswith(("saya tidak tahu", "i don't know", "i do not know"))


#CHATBOT

class ChatbotRAG:
    """Menyatukan retriever + prompt + LLM menjadi satu alur tanya jawab."""

    def __init__(self, store: FAISS, embedding, llm, info_index: Dict | None = None):
        self.store = store
        self.embedding = embedding
        self.info_index = info_index or {}
        self.model_aktif = GEMINI_MODEL if LLM_PROVIDER == "google" else DEEPSEEK_MODEL
        self.model_cadangan = list(GEMINI_CADANGAN) if LLM_PROVIDER == "google" else []
        self.riwayat: List[Tuple[str, str]] = []
        self.catatan: List[str] = []          # pesan info untuk ditampilkan antarmuka
        self._pasang_rantai(llm)

    # -- penyiapan rantai LCEL ---------------------------------------------- #

    def _pasang_rantai(self, llm) -> None:
        self.llm = llm
        self._rantai = {
            "jawab": PROMPT_JAWAB | llm | StrOutputParser(),
            "ringkas": PROMPT_RINGKAS_PERTANYAAN | llm | StrOutputParser(),
        }

    def _pindah_model_cadangan(self) -> bool:
        while self.model_cadangan:
            nama = self.model_cadangan.pop(0)
            try:
                self._pasang_rantai(buat_llm(nama))
                self.model_aktif = nama
                self.catatan.append(f"Batas penggunaan tercapai — beralih ke model cadangan '{nama}'.")
                return True
            except Exception:
                continue
        return False

    def _jalankan(self, nama_rantai: str, data: dict) -> str:
        """Panggil LLM dengan penanganan batas kuota per menit maupun per hari."""
        for _ in range(6):
            try:
                return self._rantai[nama_rantai].invoke(data)
            except Exception as e:
                pesan = str(e)
                tidak_tersedia = "NOT_FOUND" in pesan or "no longer available" in pesan
                kuota = "429" in pesan or "RESOURCE_EXHAUSTED" in pesan
                if not kuota and not tidak_tersedia:
                    raise
                if tidak_tersedia or "PerDay" in pesan or "per day" in pesan.lower():
                    if not self._pindah_model_cadangan():
                        raise KesalahanChatbot(
                            "Kuota harian semua model Google sudah habis. Coba lagi besok, "
                            "atau ganti LLM_PROVIDER=deepseek di file .env."
                        ) from e
                    continue
                jeda = jeda_dari_pesan_galat(pesan)
                self.catatan.append(f"Batas penggunaan sementara tercapai, menunggu {jeda:.0f} detik...")
                time.sleep(jeda)
        raise KesalahanChatbot("Gagal memanggil LLM: batas kuota API terus terlampaui.")

    # -- retriever ----------------------------------------------------------- #

    def cari_potongan(self, kueri: str) -> List[Document]:
        """Cari potongan teks paling relevan dengan pertanyaan.

        Dua strategi digabung karena saling menutupi kelemahan: pencarian
        kemiripan unggul bila jawaban terkumpul di satu tabel, sedangkan MMR
        menambahkan potongan dari bagian dokumen lain sehingga pertanyaan
        bertopik ganda tetap terjawab lengkap. Kueri hanya di-embedding sekali.
        """
        vektor = self.embedding.embed_query(kueri)
        hasil = self.store.similarity_search_by_vector(vektor, k=TOP_K)
        hasil += self.store.max_marginal_relevance_search_by_vector(
            vektor, k=MMR_K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MMR
        )

        unik: List[Document] = []
        sudah = set()
        for d in hasil:                       # buang duplikat, urutan relevansi dijaga
            kunci = d.metadata.get("chunk_id", d.page_content[:60])
            if kunci not in sudah:
                sudah.add(kunci)
                unik.append(d)
        return unik[:MAX_KONTEKS]

    # -- tanya jawab --------------------------------------------------------- #

    def _teks_riwayat(self, n: int = 3) -> str:
        if not self.riwayat:
            return "(belum ada percakapan sebelumnya)"
        return "\n".join(f"Pengguna: {t}\nAsisten: {j}" for t, j in self.riwayat[-n:])

    def tanya(self, pertanyaan: str) -> Tuple[str, List[Document]]:
        """Jawab satu pertanyaan. Mengembalikan (jawaban, potongan sumber)."""
        self.catatan.clear()

        # Pertanyaan lanjutan ("kalau yang itu berapa?") dijadikan mandiri dulu
        kueri = pertanyaan
        if self.riwayat:
            try:
                kueri = self._jalankan(
                    "ringkas", {"riwayat": self._teks_riwayat(), "pertanyaan": pertanyaan}
                ).strip() or pertanyaan
            except Exception:
                kueri = pertanyaan

        dokumen = self.cari_potongan(kueri)
        jawaban = self._jalankan(
            "jawab",
            {
                "bahasa": deteksi_bahasa(pertanyaan),
                "riwayat": self._teks_riwayat(),
                "konteks": format_konteks(dokumen),
                "pertanyaan": pertanyaan,
            },
        ).strip()

        self.riwayat.append((pertanyaan, jawaban))
        return jawaban, dokumen


def buat_chatbot(paksa_bangun: bool = False, pelapor: Pelapor | None = None) -> ChatbotRAG:
    """Siapkan chatbot siap pakai: index vektor + LLM."""
    store, embedding, info = muat_atau_bangun_index(paksa_bangun, pelapor)
    return ChatbotRAG(store, embedding, buat_llm(), info)
