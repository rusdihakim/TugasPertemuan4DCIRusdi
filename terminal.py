from __future__ import annotations

import argparse
import sys
import warnings

for _aliran in (sys.stdout, sys.stderr):
    try:
        _aliran.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", message=".*langchain-community.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document          # noqa: E402
from rich.console import Console                       # noqa: E402
from rich.markdown import Markdown                     # noqa: E402
from rich.panel import Panel                           # noqa: E402
from rich.table import Table                           # noqa: E402
from rich.text import Text                             # noqa: E402

from rag_chatbot import (                              # noqa: E402
    MAX_KONTEKS,
    ChatbotRAG,
    KesalahanChatbot,
    buat_chatbot,
    menjawab_tidak_tahu,
    periksa_sitasi,
)

console = Console()

PERINTAH = {
    "/bantuan": "Tampilkan daftar perintah",
    "/info": "Tampilkan status dan konfigurasi yang sedang berjalan",
    "/sumber": "Tampilkan atau sembunyikan kutipan rujukan",
    "/reset": "Mulai percakapan baru",
    "/keluar": "Akhiri sesi",
}

# Warna tema (ungu cyan)
AKSEN = "#8B5CF6"
AKSEN_TERANG = "#A78BFA"
AKSEN_CYAN = "#22D3EE"
BAYANGAN = "#3B3564"
GRADASI = ["#A78BFA", "#8B5CF6", "#7C7CF8", "#6E9BF7", "#5FB4F0", "#4FC7E4", "#3FD2D8"]

# Font blok 5x5 untuk logo sambutan
FONT_BLOK = {
    "C": ["11111", "10000", "10000", "10000", "11111"],
    "H": ["10001", "10001", "11111", "10001", "10001"],
    "A": ["01110", "10001", "11111", "10001", "10001"],
    "T": ["11111", "00100", "00100", "00100", "00100"],
    "B": ["11110", "10001", "11110", "10001", "11110"],
    "O": ["01110", "10001", "10001", "10001", "01110"],
}


# Layar sambutan

def _logo_blok(kata: str, warna_awal: int = 0) -> Text:
    """Gambar kata memakai blok besar dengan bayangan tipis di belakangnya."""
    tinggi, sisi, jarak = 5, 5, 2
    lebar = len(kata) * sisi + (len(kata) - 1) * jarak
    grid = [[0] * (lebar + 1) for _ in range(tinggi + 1)]

    for i, huruf in enumerate(kata):
        pola = FONT_BLOK[huruf]
        geser = i * (sisi + jarak)
        for baris in range(tinggi):
            for kolom in range(sisi):
                if pola[baris][kolom] == "1":
                    grid[baris][geser + kolom] = 1

    for baris in range(tinggi - 1, -1, -1):          # bayangan digeser 1 ke kanan-bawah
        for kolom in range(lebar - 1, -1, -1):
            if grid[baris][kolom] == 1 and grid[baris + 1][kolom + 1] == 0:
                grid[baris + 1][kolom + 1] = 2

    skala = 2 if console.width >= (lebar + 1) * 2 else 1

    teks = Text()
    for baris, isi in enumerate(grid):
        warna = GRADASI[min(warna_awal + baris, len(GRADASI) - 1)]
        for nilai in isi:
            if nilai == 1:
                teks.append("█" * skala, style=warna)
            elif nilai == 2:
                teks.append("░" * skala, style=BAYANGAN)
            else:
                teks.append(" " * skala)
        teks.append("\n")
    return teks


def tampilkan_layar_pembuka() -> None:
    """Layar pembuka: logo, penjelasan singkat, lalu menunggu Enter."""
    console.clear()
    console.print()

    judul = Text.assemble(
        ("✻ ", AKSEN_TERANG),
        ("Selamat datang di ", "bold white"),
        ("ChatBot", f"bold {AKSEN_TERANG}"),
    )
    console.print(Panel(judul, border_style=AKSEN, expand=False, padding=(0, 2)))
    console.print()
    console.print(_logo_blok("CHATBOT"))

    console.print("Asisten tanya jawab yang menjawab dari sumber, bukan dari tebakan.",
                  style="bold white")
    console.print()
    console.print("Setiap jawaban dilengkapi rujukan halamannya, sehingga dapat", style=AKSEN_CYAN)
    console.print("ditelusuri kembali ke bagian aslinya.", style=AKSEN_CYAN)
    console.print()
    console.print(Text.assemble(
        ("Tekan ", "dim"), ("Enter", f"bold {AKSEN_TERANG}"), (" untuk memulai", "dim"),
    ))

    if sys.stdin.isatty():
        try:
            console.input("")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Sampai jumpa![/dim]")
            sys.exit(0)
    console.clear()


def tampilkan_kepala_ruang_chat() -> None:
    """Baris informasi ringkas di bagian atas ruang chat."""
    isi = Text()
    isi.append("✻ ChatBot", style=f"bold {AKSEN_TERANG}")
    isi.append("   Basis pengetahuan siap digunakan.\n", style="white")
    isi.append("Ajukan pertanyaan Anda, atau ketik ", style="dim")
    isi.append("/bantuan", style=AKSEN_CYAN)
    isi.append(" untuk melihat perintah yang tersedia.", style="dim")
    console.print(Panel(isi, border_style=AKSEN, padding=(0, 2)))


def tampilkan_bantuan() -> None:
    tabel = Table(show_header=True, header_style=f"bold {AKSEN_TERANG}", border_style="dim", box=None)
    tabel.add_column("Perintah", style=AKSEN_CYAN, no_wrap=True)
    tabel.add_column("Fungsi")
    for perintah, fungsi in PERINTAH.items():
        tabel.add_row(perintah, fungsi)
    console.print(Panel(tabel, title="Bantuan", title_align="left",
                        border_style=AKSEN, padding=(1, 2)))
    console.print(
        "[dim]Pertanyaan lanjutan dapat diajukan langsung — konteks percakapan "
        "tetap dijaga. Tekan Ctrl+C kapan saja untuk berhenti.[/dim]"
    )


def tampilkan_info(bot: ChatbotRAG) -> None:
    tabel = Table(show_header=False, border_style="dim", box=None)
    tabel.add_column(style="dim", no_wrap=True)
    tabel.add_column(style="white")
    tabel.add_row("Basis pengetahuan", f"{bot.info_index.get('jumlah_chunk', '-')} bagian terindeks")
    tabel.add_row("Model jawaban", bot.model_aktif)
    tabel.add_row("Rujukan per jawaban", f"hingga {MAX_KONTEKS} bagian dipertimbangkan")
    tabel.add_row("Riwayat percakapan", f"{len(bot.riwayat)} pertanyaan")
    console.print(Panel(tabel, title="Info", title_align="left",
                        border_style=AKSEN, padding=(1, 2)))


def tampilkan_jawaban(jawaban: str, dokumen: list[Document], tampil_kutipan: bool) -> None:
    console.print()
    console.print(Panel(Markdown(jawaban), title="Jawaban", title_align="left",
                        border_style=AKSEN_CYAN, padding=(1, 2)))

    if menjawab_tidak_tahu(jawaban):
        console.print("[dim]Tidak ada rujukan — informasi ini tidak tersedia pada sumber.[/dim]")
        return

    meleset = periksa_sitasi(jawaban, dokumen)
    if meleset:
        console.print(
            f"[yellow]Perhatian:[/yellow] rujukan Halaman {', '.join(meleset)} tidak berasal "
            "dari bagian yang ditelusuri. Mohon acu daftar rujukan berikut."
        )

    console.print(f"[bold {AKSEN_TERANG}]Rujukan:[/bold {AKSEN_TERANG}]")
    for i, d in enumerate(dokumen[:5], 1):
        console.print(
            f"  [{AKSEN_TERANG}]{i}.[/{AKSEN_TERANG}] {d.metadata.get('sumber', '-')} — "
            f"[bold]Halaman {d.metadata.get('halaman', '?')}[/bold]"
        )
        if tampil_kutipan:
            kutipan = " ".join(d.page_content.split())[:240]
            console.print(f'     [dim]"{kutipan}..."[/dim]')
    sisa = len(dokumen) - 5
    if sisa > 0:
        console.print(f"  [dim]dan {sisa} bagian lain yang turut dipertimbangkan[/dim]")


# Percakapan


def jalankan_percakapan(bot: ChatbotRAG) -> None:
    tampilkan_kepala_ruang_chat()
    tampil_kutipan = True

    while True:
        try:
            masukan = console.input(f"\n[bold {AKSEN_TERANG}]❯[/bold {AKSEN_TERANG}] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Terima kasih telah menggunakan ChatBot.[/dim]")
            return

        if not masukan:
            continue

        perintah = masukan.lower()
        if perintah in ("/keluar", "/exit", "/quit", "exit", "quit", "keluar"):
            console.print("[dim]Terima kasih telah menggunakan ChatBot.[/dim]")
            return
        if perintah in ("/bantuan", "/help", "?"):
            tampilkan_bantuan()
            continue
        if perintah == "/info":
            tampilkan_info(bot)
            continue
        if perintah == "/sumber":
            tampil_kutipan = not tampil_kutipan
            console.print(f"[dim]Kutipan rujukan {'ditampilkan' if tampil_kutipan else 'disembunyikan'}.[/dim]")
            continue
        if perintah == "/reset":
            bot.riwayat.clear()
            console.print("[dim]Percakapan baru dimulai.[/dim]")
            continue
        if perintah.startswith("/"):
            console.print(f"[yellow]Perintah '{masukan}' tidak dikenal.[/yellow] "
                          f"Ketik [{AKSEN_CYAN}]/bantuan[/{AKSEN_CYAN}] untuk daftar perintah.")
            continue

        try:
            with console.status(f"[{AKSEN_CYAN}]Menelusuri sumber dan menyusun jawaban...[/{AKSEN_CYAN}]"):
                jawaban, dokumen = bot.tanya(masukan)
        except KeyboardInterrupt:
            console.print("[dim]Dibatalkan.[/dim]")
            continue
        except KesalahanChatbot as e:
            console.print(Panel(str(e), title="Gagal menjawab", border_style="red"))
            continue
        except Exception as e:
            console.print(Panel(f"{type(e).__name__}: {e}", title="Terjadi kesalahan",
                                border_style="red"))
            continue

        for catatan in bot.catatan:
            console.print(f"[dim]{catatan}[/dim]")
        tampilkan_jawaban(jawaban, dokumen, tampil_kutipan)


# Titik masuk aplikasi

def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot RAG dokumen PDF (terminal).")
    parser.add_argument("--rebuild", action="store_true",
                        help="bangun ulang index vektor dari PDF")
    args = parser.parse_args()

    tampilkan_layar_pembuka()

    try:
        with console.status(f"[{AKSEN_CYAN}]Menyiapkan ChatBot...[/{AKSEN_CYAN}]") as status:
            bot = buat_chatbot(
                paksa_bangun=args.rebuild,
                pelapor=lambda pesan: status.update(f"[{AKSEN_CYAN}]{pesan}[/{AKSEN_CYAN}]"),
            )
    except KesalahanChatbot as e:
        console.print(Panel(str(e), title="Tidak bisa memulai", border_style="red"))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Dibatalkan.[/dim]")
        sys.exit(1)

    jalankan_percakapan(bot)


if __name__ == "__main__":
    main()
