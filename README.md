# LectureLens
# LectureLens

**Turn hours of YouTube lectures into searchable, intelligent study material in minutes.**

LectureLens is an offline-first multimodal AI engine that indexes long educational videos (JEE, NEET, university lectures) at 5-second granularity. It combines speech (ASR), text on slides (OCR + LaTeX), and visual objects (YOLO/CLIP) into a unified vector database, allowing natural-language search across modalities and instant export to PPT, Notion, or Obsidian.

Built for Indian students who waste hours scrubbing through 2–3 hour unindexed lectures.

---

## ✨ Core Functions

- **Multimodal 5-Second Search**  
  Search by what was **spoken** (ASR), **written** on slides (OCR + LaTeX), or **shown** (objects/diagrams via YOLO/CLIP).

- **Smart JEE/NEET Notes**  
  Auto-detects mathematical equations → converts to clean LaTeX. One-click PPT or Obsidian export with timestamps and clickable YouTube deep links.

- **Chat with the Lecture (RAG)**  
  Ask questions like “Explain K-Map minimization from this video” and get answers grounded in exact timestamps.

- **Zero-Shot Visual Search**  
  Find anything without retraining — “professor drawing neural network diagram” or “satellite orbit equation on board”.

- **Safe-for-Study Filter**  
  Auto-hides buckets containing weapons/blood (optional) so parents can safely share with students.

- **Live Stream & Local File Support**  
  Works with YouTube live streams (caps at “aired so far”) and already-downloaded MP4s.

---

## 🛠 Tech Stack & Architecture

**Backend Pipeline (Python 3.10)**
- Video & ASR: `yt-dlp` + `aria2c` (16 parallel connections)
- Frame Extraction: FFmpeg (`-skip_frame nokey`) + exact PTS timestamps → `timestamps.json`
- Object Detection: Ultralytics YOLOv8n + CLIP zero-shot (SigLIP)
- OCR + LaTeX: PaddleOCR-VL (multilingual + math)
- Vector Database: ChromaDB (persistent SQLite) with three content types:
  - `asr` → spoken text
  - `visual_ocr` → slide text + LaTeX
  - `visual_objects` → YOLO/CLIP detections

**Frontend & Export**
- Gradio (clean tabs + tri-modal table with clickable timestamps + yellow LaTeX highlighting)
- Obsidian Export: Markdown with `$$` LaTeX blocks + deep links
- Notion Export: Direct API push (blocks + database)
- PPT Export: `python-pptx` + SymPy-rendered equations
- RAG Chat: Gemini-Flash (cached in ChromaDB)

**Key Optimizations**
- I-frame skipping → 100+ FPS extraction
- Spillover-aware 5s bucketing
- Parallel GPU workers (12 for analysis)
- Zero-garbage filtering (OCR ≥ 80%, YOLO ≥ 55%)

---

## 🎯 Scope & Target Use Cases

**Primary Focus**  
JEE/NEET & university students in India who watch 2–4 hour unindexed lectures daily.

**Secondary Use Cases** (bonus)  
- Long corporate training / Zoom recordings  
- Any educational video where slide content + speech matters

**What it is NOT**  
A general movie/series clip finder or wildlife camera-trap tool (those are saturated spaces and dilute the core education story).

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/LectureLens.git
cd LectureLens

# 2. Install
conda create -n lecturelens python=3.10
conda activate lecturelens
pip install -r requirements.txt

# 3. Run full pipeline on any YouTube link
python fast_fetch.py "https://youtube.com/watch?v=EXAMPLE"

# 4. Launch the interface
python app.py
