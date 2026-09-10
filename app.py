"""
AI Video & Meeting Intelligence - Streamlit UI
===============================================
A modern, responsive, and intuitive Streamlit frontend for:
- Transcribing video and audio content
- Generating titles and executive summaries
- Extracting action items, key decisions, and open questions
- Interactive RAG-based chat with meetings and videos
"""

import os
import json
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Attempt imports from local project modules
try:
    from utils.audio_processor import process_input
    from core.transcriber import transcribe_all
    from core.summarizer import summarize, generate_title
    from core.extractor import (
        extract_action_items,
        extract_key_decisions,
        extract_questions,
    )
    from core.rag_engine import build_rag_chain, ask_question
    BACKEND_AVAILABLE = True
    BACKEND_ERROR = None
except ImportError as err:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(err)


# ==============================================================================
# 1. PAGE CONFIGURATION & MODERN THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Video & Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern typography, cards, and clean tabs
st.markdown(
    """
    <style>
    /* Main Layout Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.15);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    /* Content Cards */
    .insight-card {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, 0.9);
        margin: 6px 0 0 0;
        font-size: 1.02rem;
    }

    /* Chat bubble spacing */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================================================================
# 3. PIPELINE EXECUTION FUNCTION
# ==============================================================================
def run_pipeline(source: str, language: str = "english") -> dict:
    """
    Executes the 5-step processing pipeline with real-time visual progress steps.
    """
    with st.status("🚀 Processing Video / Meeting...", expanded=True) as status:
        try:
            # Step 1: Process and chunk audio
            st.write("🎵 **Step 1/5:** Chunking audio / extracting stream...")
            chunks = process_input(source)
            st.write("✓ Audio processed into chunks successfully.")

            # Step 2: Speech transcription
            st.write(f"🎙️ **Step 2/5:** Transcribing speech (Language: `{language}`)...")
            transcript = transcribe_all(chunks, language)
            word_count = len(transcript.split())
            st.write(f"✓ Transcription generated ({word_count:,} words).")

            # Step 3: Title & Executive Summary
            st.write("📝 **Step 3/5:** Generating title and executive summary...")
            title = generate_title(transcript)
            summary = summarize(transcript)
            st.write("✓ Title and executive summary created.")

            # Step 4: Extract Key Takeaways
            st.write("🔍 **Step 4/5:** Extracting action items, decisions, and questions...")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            st.write("✓ Key takeaways & task assignments extracted.")

            # Step 5: Build RAG Vector Index
            st.write("🧠 **Step 5/5:** Building RAG semantic search engine...")
            rag_chain = build_rag_chain(transcript)
            st.write("✓ RAG engine active and indexed.")

            status.update(label="🎉 Analysis Complete! Ready to explore.", state="complete", expanded=False)

            return {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "language": language,
                "source": source,
            }

        except Exception as e:
            status.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)
            st.error(f"Error details: {str(e)}")
            return None


# ==============================================================================
# 4. SIDEBAR CONFIGURATION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("🎙️ AI Video Assistant")
    st.markdown("Transform video and audio recordings into intelligent structured notes and an interactive RAG chatbot.")
    st.divider()

    # Alert if backend modules are missing
    if not BACKEND_AVAILABLE:
        st.error(f"⚠️ Backend modules not found: {BACKEND_ERROR}")
        st.info("Ensure `app.py` is in your project root where `core/` and `utils/` are located.")

    st.subheader("1. Select Media Source")
    source_type = st.radio(
        "Source Input Mode:",
        ["🔗 YouTube URL", "📁 Upload Media File", "💻 Local File Path"],
        help="Select where your video or audio originates."
    )

    source_path = None

    if source_type == "🔗 YouTube URL":
        yt_input = st.text_input(
            "Enter YouTube Link:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a public YouTube video link."
        )
        if yt_input.strip():
            source_path = yt_input.strip()

    elif source_type == "📁 Upload Media File":
        uploaded_media = st.file_uploader(
            "Upload Audio or Video File:",
            type=["mp4", "mp3", "wav", "m4a", "webm", "mkv", "aac", "ogg"],
            help="Supported: MP4, MP3, WAV, M4A, WEBM, MKV, etc."
        )
        if uploaded_media is not None:
            # Save uploaded buffer to a temporary file on disk for process_input
            ext = Path(uploaded_media.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_file.write(uploaded_media.getbuffer())
                source_path = temp_file.name
            size_mb = uploaded_media.size / (1024 * 1024)
            st.caption(f"📁 Selected: `{uploaded_media.name}` ({size_mb:.1f} MB)")

    else:  # Local File Path
        local_input = st.text_input(
            "Enter File Path on Disk:",
            placeholder="C:/path/to/meeting.mp4 or /home/user/call.mp3",
            help="Direct path to an audio or video file on your computer."
        )
        if local_input.strip():
            source_path = local_input.strip()

    st.subheader("2. Target Language")
    language = st.selectbox(
        "Audio / Speech Language:",
        ["english", "hinglish", "spanish", "french", "german", "hindi"],
        index=0,
        help="Target language for the transcriber."
    )

    st.divider()

    # Run Analysis Button
    run_btn = st.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True,
        disabled=not bool(source_path) or not BACKEND_AVAILABLE,
    )

    if run_btn and source_path:
        st.session_state.chat_history = []  # Reset chat session
        result = run_pipeline(source_path, language)
        if result:
            st.session_state.pipeline_result = result
            st.rerun()

    # Reset Button
    if st.session_state.pipeline_result is not None:
        if st.button("🔄 Start New Video Analysis", use_container_width=True):
            st.session_state.pipeline_result = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown(
        """
        <small style='color: gray;'>
        🔒 Files are processed locally or via configured LLM/STT APIs in <code>.env</code>.
        </small>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 5. MAIN CONTENT DISPLAY
# ==============================================================================

# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎙️ AI Video & Meeting Assistant</h1>
        <p>Instant summaries, action items, key decisions, and interactive chat from any video or audio source.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# State A: No video analyzed yet (Welcome Screen)
if st.session_state.pipeline_result is None:
    st.markdown("### 🌟 Getting Started")
    st.info("👈 Enter a **YouTube URL**, **upload a media file**, or provide a **local file path** in the sidebar to begin.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 📝 Smart Summaries")
        st.caption("AI-generated titles and comprehensive executive briefs distilled from hours of audio.")
    with c2:
        st.markdown("#### ✅ Action Item Detection")
        st.caption("Automatically detect tasks, assignments, and follow-ups with owners and deliverables.")
    with c3:
        st.markdown("#### ⚖️ Key Decisions")
        st.caption("Isolate core agreements, architectural choices, and consensus items.")
    with c4:
        st.markdown("#### 💬 RAG Chat Assistant")
        st.caption("Ask questions and search quotes grounded directly in the source transcript.")

    st.divider()
    with st.expander("🛠️ Workflow Architecture", expanded=False):
        st.markdown(
            """
            - **Ingestion (`utils.audio_processor`):** Audio extraction and dynamic chunking.
            - **Transcription (`core.transcriber`):** Multi-language speech recognition.
            - **Summarizer (`core.summarizer`):** High-level summary & title generation.
            - **Extraction (`core.extractor`):** Action items, key decisions, and open questions.
            - **RAG Engine (`core.rag_engine`):** Semantic vector search and conversational Q&A.
            """
        )

# State B: Video analysis completed (Tabbed Dashboard)
else:
    res = st.session_state.pipeline_result

    # Display Title
    st.markdown(f"## 📌 {res.get('title', 'Meeting Analysis')}")

    # Top Metric Counters
    words = len(res.get("transcript", "").split())
    def count_elements(val):
        if isinstance(val, list):
            return len(val)
        if isinstance(val, str):
            lines = [l for l in val.strip().splitlines() if l.strip()]
            return len(lines)
        return 0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Transcript Words", f"{words:,}")
    with col_m2:
        st.metric("Action Items", f"{count_elements(res.get('action_items'))}")
    with col_m3:
        st.metric("Key Decisions", f"{count_elements(res.get('key_decisions'))}")
    with col_m4:
        st.metric("Open Questions", f"{count_elements(res.get('open_questions'))}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 5 Main Feature Tabs
    tab_summary, tab_insights, tab_questions, tab_rag_chat, tab_transcript = st.tabs([
        "📋 Executive Summary",
        "✅ Actions & Decisions",
        "❓ Open Questions",
        "💬 Chat with Video (RAG)",
        "📜 Transcript & Export",
    ])

    # ----------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ----------------------------------------------------
    with tab_summary:
        st.subheader("📋 Executive Summary")
        summary_text = res.get("summary", "No summary available.")
        st.markdown(
            f"""
            <div class="insight-card">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------
    # TAB 2: ACTION ITEMS & DECISIONS
    # ----------------------------------------------------
    with tab_insights:
        col_tasks, col_decisions = st.columns(2)

        with col_tasks:
            st.subheader("✅ Action Items")
            actions = res.get("action_items")
            if isinstance(actions, list) and actions:
                for i, item in enumerate(actions, 1):
                    st.markdown(f"**{i}.** {item}")
            elif isinstance(actions, str) and actions.strip():
                st.markdown(actions)
            else:
                st.info("No action items identified.")

        with col_decisions:
            st.subheader("🔑 Key Decisions")
            decisions = res.get("key_decisions")
            if isinstance(decisions, list) and decisions:
                for i, decision in enumerate(decisions, 1):
                    st.markdown(f"**{i}.** {decision}")
            elif isinstance(decisions, str) and decisions.strip():
                st.markdown(decisions)
            else:
                st.info("No key decisions identified.")

    # ----------------------------------------------------
    # TAB 3: OPEN QUESTIONS
    # ----------------------------------------------------
    with tab_questions:
        st.subheader("❓ Open & Unresolved Questions")
        questions = res.get("open_questions")
        if isinstance(questions, list) and questions:
            for i, q in enumerate(questions, 1):
                st.markdown(f"**{i}.** {q}")
        elif isinstance(questions, str) and questions.strip():
            st.markdown(questions)
        else:
            st.info("No unresolved questions detected.")

    # ----------------------------------------------------
    # TAB 4: INTERACTIVE RAG CHAT
    # ----------------------------------------------------
    with tab_rag_chat:
        st.subheader("💬 Chat with your Meeting")
        st.caption("Ask questions and search through the discussion. Responses are verified against the transcript.")

        # Quick Suggestion Chips
        st.markdown("**Suggested Quick Questions:**")
        q_col1, q_col2, q_col3 = st.columns(3)
        quick_prompt = None

        if q_col1.button("🎯 What were the main takeaways?", use_container_width=True):
            quick_prompt = "What were the main takeaways from this discussion?"
        if q_col2.button("📅 Were any deadlines mentioned?", use_container_width=True):
            quick_prompt = "Were there any specific deadlines, deliverables, or target dates discussed?"
        if q_col3.button("👥 Who was assigned which task?", use_container_width=True):
            quick_prompt = "Who was assigned which tasks or next steps?"

        # Render conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Bar
        chat_query = st.chat_input("Ask a question about the video...")
        active_query = chat_query or quick_prompt

        if active_query:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # Query RAG Engine
            with st.chat_message("assistant"):
                with st.spinner("Searching transcript & generating grounded answer..."):
                    try:
                        rag = res.get("rag_chain")
                        if rag:
                            answer = ask_question(rag, active_query)
                        else:
                            answer = "RAG knowledge base is not initialized."
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as err:
                        err_msg = f"Failed to answer question: {str(err)}"
                        st.error(err_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    # ----------------------------------------------------
    # TAB 5: TRANSCRIPT & EXPORT
    # ----------------------------------------------------
    with tab_transcript:
        st.subheader("📜 Full Transcript")
        full_transcript = res.get("transcript", "")

        search_kw = st.text_input("🔍 Search transcript for keyword:", placeholder="Type a term to filter...")
        if search_kw:
            matches = [line for line in full_transcript.splitlines() if search_kw.lower() in line.lower()]
            st.caption(f"Found {len(matches)} matching segment(s):")
            st.text_area("Search Results", "\n\n".join(matches), height=280)
        else:
            st.text_area("Transcript Content", full_transcript, height=350)

        st.divider()
        st.subheader("📥 Export & Download Report")

        # Markdown Report Generation
        export_md = f"""# {res.get('title', 'Meeting Analysis')}

## Executive Summary
{res.get('summary', 'N/A')}

---

## Action Items
{res.get('action_items', 'N/A')}

---

## Key Decisions
{res.get('key_decisions', 'N/A')}

---

## Open Questions
{res.get('open_questions', 'N/A')}

---

## Full Transcript
{full_transcript}
"""

        # JSON Data Export
        export_json = json.dumps(
            {
                "title": res.get("title"),
                "summary": res.get("summary"),
                "action_items": res.get("action_items"),
                "key_decisions": res.get("key_decisions"),
                "open_questions": res.get("open_questions"),
                "transcript": full_transcript,
            },
            indent=2,
        )

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            clean_title = "".join(c for c in res.get('title', 'meeting') if c.isalnum() or c in (' ', '_', '-')).rstrip()
            st.download_button(
                label="📄 Download Report (.md)",
                data=export_md,
                file_name=f"{clean_title.replace(' ', '_')}_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📝 Download Raw Transcript (.txt)",
                data=full_transcript,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                label="📊 Download JSON Export (.json)",
                data=export_json,
                file_name="meeting_export.json",
                mime="application/json",
                use_container_width=True,
            )