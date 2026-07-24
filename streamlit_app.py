from __future__ import annotations

import time
from typing import Any

import streamlit as st

from rag_chatbot import (
    MAX_KONTEKS,
    ChatbotRAG,
    KesalahanChatbot,
    buat_chatbot,
    menjawab_tidak_tahu,
    periksa_sitasi,
)


st.set_page_config(
    page_title="ChatBot RAG PDF",
    page_icon="docs",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.18);
    }
    .app-title {
        margin: 0 0 0.25rem 0;
        font-size: 2rem;
        line-height: 1.15;
        font-weight: 750;
    }
    .app-subtitle {
        color: rgba(100, 100, 100, 0.95);
        margin-bottom: 1.2rem;
    }
    .source-line {
        color: rgba(95, 95, 95, 0.95);
        font-size: 0.92rem;
    }
    .metric-note {
        color: rgba(95, 95, 95, 0.95);
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _progress_reporter(container: Any):
    def report(message: str) -> None:
        container.info(message)

    return report


def load_bot(force_rebuild: bool = False) -> ChatbotRAG:
    progress_box = st.empty()
    try:
        with st.spinner("Menyiapkan basis pengetahuan..."):
            bot = buat_chatbot(
                paksa_bangun=force_rebuild,
                pelapor=_progress_reporter(progress_box),
            )
    finally:
        progress_box.empty()
    return bot


def ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "bot" not in st.session_state:
        st.session_state.bot = load_bot()


def reset_chat() -> None:
    st.session_state.messages = []
    if "bot" in st.session_state:
        st.session_state.bot.riwayat.clear()


def rebuild_index() -> None:
    st.session_state.bot = load_bot(force_rebuild=True)
    reset_chat()
    st.toast("Index selesai dibangun ulang.")


def render_sources(documents: list, show_quotes: bool) -> None:
    if not documents:
        st.caption("Tidak ada rujukan yang ditemukan.")
        return

    for index, doc in enumerate(documents[:5], start=1):
        source = doc.metadata.get("sumber", "-")
        page = doc.metadata.get("halaman", "?")
        with st.expander(f"{index}. {source} - Halaman {page}", expanded=False):
            if show_quotes:
                quote = " ".join(doc.page_content.split())
                st.write(quote[:900] + ("..." if len(quote) > 900 else ""))
            else:
                st.caption("Kutipan sumber sedang disembunyikan.")

    extra = len(documents) - 5
    if extra > 0:
        st.caption(f"Dan {extra} bagian lain turut dipertimbangkan.")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            notes = message.get("notes") or []
            for note in notes:
                st.info(note)

            documents = message.get("documents") or []
            if documents and not menjawab_tidak_tahu(message["content"]):
                mismatch = periksa_sitasi(message["content"], documents)
                if mismatch:
                    st.warning(
                        "Rujukan halaman pada jawaban perlu dicek. "
                        f"Acu daftar sumber di bawah ini untuk halaman: {', '.join(mismatch)}."
                    )
                render_sources(documents, st.session_state.show_sources)


ensure_state()
bot: ChatbotRAG = st.session_state.bot

with st.sidebar:
    st.subheader("Status")
    info = bot.info_index or {}
    st.metric("Bagian terindeks", info.get("jumlah_chunk", "-"))
    st.metric("Model", bot.model_aktif)
    st.markdown(f"<p class='metric-note'>Rujukan per jawaban: hingga {MAX_KONTEKS} bagian.</p>", unsafe_allow_html=True)

    st.divider()
    st.toggle("Tampilkan kutipan sumber", key="show_sources")
    st.button("Reset percakapan", use_container_width=True, on_click=reset_chat)
    st.button("Bangun ulang index", use_container_width=True, on_click=rebuild_index)

    st.divider()
    st.caption("Gunakan tombol rebuild setelah file PDF atau pengaturan chunk berubah.")

st.markdown("<h1 class='app-title'>ChatBot RAG PDF</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-subtitle'>Tanyakan isi dokumen. Jawaban dibuat dari potongan PDF yang terindeks dan menampilkan rujukan halaman.</p>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info("Mulai dengan pertanyaan seperti: Apa itu LoRa?")

for item in st.session_state.messages:
    render_message(item)

prompt = st.chat_input("Tulis pertanyaan tentang dokumen...")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("Menelusuri sumber dan menyusun jawaban...")
        started = time.time()
        try:
            answer, documents = bot.tanya(prompt)
            elapsed = time.time() - started
            placeholder.empty()
            st.markdown(answer)

            notes = list(bot.catatan)
            if elapsed > 1:
                notes.append(f"Selesai dalam {elapsed:.1f} detik.")
            for note in notes:
                st.info(note)

            if documents and not menjawab_tidak_tahu(answer):
                mismatch = periksa_sitasi(answer, documents)
                if mismatch:
                    st.warning(
                        "Rujukan halaman pada jawaban perlu dicek. "
                        f"Acu daftar sumber di bawah ini untuk halaman: {', '.join(mismatch)}."
                    )
                render_sources(documents, st.session_state.show_sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "documents": documents,
                    "notes": notes,
                }
            )
        except KesalahanChatbot as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.append({"role": "assistant", "content": str(exc)})
        except Exception as exc:
            placeholder.empty()
            message = f"{type(exc).__name__}: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
