# 🎙️ AI Video & Meeting Intelligence

An end-to-end pipeline that turns raw video/audio (YouTube links or local files) into structured, searchable meeting intelligence — transcripts, executive summaries, action items, key decisions, open questions, and a RAG-powered chat interface to query the content directly.

Built with **LangChain (LCEL)**, **Google Gemini**, **OpenAI Whisper**, **Sarvam AI**, **ChromaDB**, and **Streamlit**.

---

## ✨ Features

- 📝 **Smart Summaries** — AI-generated titles and executive briefs distilled from hours of audio
- ✅ **Action Item Detection** — automatically extracts tasks, owners, and deadlines
- 🔑 **Key Decisions** — isolates core agreements and consensus points from the discussion
- 💬 **RAG Chat Assistant** — ask questions about the content; answers are grounded directly in the transcript, not hallucinated
- 🌐 **Multi-source input** — YouTube URL, uploaded media file, or local file path
- 🗣️ **Dual speech-to-text engine, chosen by language:**
  - **English** → OpenAI Whisper, running fully locally (no API cost, no data leaves your machine)
  - **Hinglish** → Sarvam AI's speech-to-text-translate API, which transcribes and translates Hindi/Hinglish speech to English in one step
- 📥 **Export** — download results as Markdown report, raw transcript, or structured JSON
- 🔍 **In-transcript keyword search**

---

## 🏗️ Architecture

```
                ┌──────────────────┐
   YouTube URL  │                  │
   or Local File├─► audio_processor├─► Chunked WAV audio (10-min chunks)
                │  (yt-dlp/pydub)  │
                └──────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │      transcriber        │
              │  routes by language:    │
              │                         │
              │  english  → Whisper     │   (local model, no API cost)
              │  hinglish → Sarvam AI   │   (cloud STT + translation,
              │             (25s pieces)│    25s pieces per API limit)
              └────────────────────────┘
                          │
                          ▼
                    Full Transcript
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
    ┌───────────┐  ┌─────────────┐  ┌──────────────┐
    │summarizer │  │  extractor   │  │  rag_engine   │
    │(title +   │  │(action items,│  │ (ChromaDB +   │
    │ summary)  │  │ decisions,   │  │  Gemini chat)  │
    │           │  │ questions —  │  │               │
    │           │  │ 1 combined   │  │               │
    │           │  │ API call)    │  │               │
    └───────────┘  └─────────────┘  └──────────────┘
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                 Streamlit UI (app.py)
         Tabs: Summary · Actions/Decisions ·
          Open Questions · RAG Chat · Export
```

All Gemini API calls are routed through a custom **rate limiter with exponential backoff** (`core/rate_limiter.py`) to stay within free-tier quota limits and gracefully retry on transient failures.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`gemini-3.5-flash-lite`) via `langchain-google-genai` |
| Orchestration | LangChain (LCEL — pipe-based chains) |
| Speech-to-Text (English) | OpenAI Whisper — local model, no API cost |
| Speech-to-Text + Translation (Hinglish) | Sarvam AI (`saaras:v2.5`) — cloud API |
| Vector Store / RAG | ChromaDB + HuggingFace sentence-transformer embeddings |
| Frontend | Streamlit |
| Audio ingestion | yt-dlp, pydub, ffmpeg |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-video-meeting-assistant.git
cd ai-video-meeting-assistant
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r Requirements.txt
```

### 4. Install FFmpeg (required for audio processing)
- **Windows:** `choco install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### 5. Set up your API keys
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```
- Gemini key (free, no credit card): [Google AI Studio](https://aistudio.google.com/apikey)
- Sarvam key (only needed if you plan to use Hinglish transcription): [Sarvam AI](https://www.sarvam.ai/)

> **Note:** `SARVAM_API_KEY` is only required if you select "hinglish" as the transcription language. English-only usage works with just the Gemini key.

### 6. Run it

**CLI version:**
```bash
python main.py
```

**Streamlit web app:**
```bash
streamlit run app.py
```

---

## 📊 Rate Limits & Free Tier

This project runs entirely on Google Gemini's **free tier** (`gemini-3.5-flash-lite`):
- 15 requests/minute
- 500 requests/day

To stay within these limits, the project includes:
- A custom rate limiter that spaces out API calls and retries failures with exponential backoff
- A **combined extraction call** — action items, key decisions, and open questions are extracted in a single API call instead of three, cutting quota usage for that stage by ~66%
- Tunable transcript chunking (`chunk_size` / `chunk_overlap`) to balance summary quality against number of API calls

Whisper transcription (English) runs locally and has no API quota at all. Sarvam transcription (Hinglish) is subject to Sarvam's own separate rate limits — check their docs if you plan on heavy Hinglish usage.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web UI
├── main.py                 # CLI entry point
├── core/
│   ├── transcriber.py      # Dual-engine transcription: Whisper (English) / Sarvam (Hinglish)
│   ├── summarizer.py       # Title & executive summary generation
│   ├── extractor.py        # Action items / decisions / questions (combined call)
│   ├── rag_engine.py       # RAG chain — retrieval + grounded Q&A
│   ├── vector_store.py     # ChromaDB embedding + retrieval
│   └── rate_limiter.py     # API call throttling + retry logic
├── utils/
│   └── audio_processor.py  # YouTube/local file ingestion, WAV conversion & chunking
├── Requirements.txt
└── .env                    # API keys (not committed — see .gitignore)
```

---

## 🌍 How Hinglish Transcription Works

Sarvam's synchronous speech-to-text-translate API only accepts audio clips up to 30 seconds long. To work around this, `transcriber.py` automatically:
1. Splits each 10-minute audio chunk into 25-second pieces (with a 5-second safety margin)
2. Sends each piece to Sarvam individually
3. Stitches the translated English transcripts back together in order

This is fully automatic — the caller just passes `language="hinglish"` and never has to think about the underlying piece-splitting.

---

## 🚧 Known Limitations

- Free-tier daily quota (500 requests/day on Gemini) means heavy usage or many concurrent users could hit rate limits
- Whisper transcription runs on CPU by default — larger files take longer without a GPU
- Hinglish transcription depends on Sarvam AI's API availability and its own separate rate limits
- RAG retrieval quality depends on transcript chunking granularity

---

## 🔮 Possible Future Improvements

- Speaker diarization (who said what)
- Support for additional languages beyond English/Hinglish
- Persistent storage for multiple past meetings
- Deployment with billing-capped Gemini tier for public access

---

## 📄 License

MIT — feel free to use, modify, and learn from this project.
