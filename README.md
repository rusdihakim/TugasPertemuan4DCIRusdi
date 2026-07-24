## Cara Install

```bash
# 1. Masuk ke folder project
cd bangrus

# 2. (disarankan) buat virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install semua library
pip install -r requirements.txt

# 4. Siapkan konfigurasi
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

Lalu isi `GOOGLE_API_KEY` pada file `.env` (ambil gratis di
<https://aistudio.google.com/app/apikey>).

## Cara Menjalankan

```bash
# Mulai chatbot di terminal
.\.venv\Scripts\python.exe terminal.py

# Mulai chatbot di browser dengan Streamlit
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502

# Bangun ulang index vektor (mis. setelah PDF atau chunk_size diganti)
.\.venv\Scripts\python.exe terminal.py --rebuild
```

Alternatif Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_streamlit.ps1
```

Alternatif Command Prompt / double click:

```bat
run_streamlit.bat
```



### Perintah di dalam chat

| Perintah | Fungsi |
|---|---|
| `/bantuan` | Tampilkan daftar perintah |
| `/info` | Tampilkan info dokumen dan pengaturan yang dipakai |
| `/sumber` | Tampilkan / sembunyikan kutipan teks sumber |
| `/reset` | Mulai percakapan baru (hapus riwayat) |
| `/keluar` | Keluar dari aplikasi (bisa juga `exit` / `quit` / Ctrl+C) |
