"""
FastFetch - Ultra-Parallel Video Downloader + ChromaDB Indexer
======================================================
Downloads video + ASR (Google auto-captions) only.
NO manual subtitles. NO audio.

Step 1 — Video file (fast, 16 workers)
           Supports both regular videos AND live streams.
           For live streams: captures only what has aired so far
           (from stream start up to the moment you run the script).
Step 2 — ASR only (lightweight, separate call)
Step 3 — ChromaDB Indexing (5-second multimodal buckets)
           Each bucket gets THREE entries:
             _asr            → spoken text (filled now)
             _visual_ocr     → OCR/slide text placeholder (filled by image_model.py)
             _visual_objects → YOLO object labels placeholder (filled by image_model.py)
           Kept separate so "satellite" (object search) never bleeds into
           OCR text results and vice versa. Enables filtered queries:
             where={"content_type": "asr"}            → speech only
             where={"content_type": "visual_ocr"}     → slide/screen text only
             where={"content_type": "visual_objects"} → detected objects only
             no filter                                → full multimodal search

5 ASR Quality Improvements:
    FIX 1 — Robust language detection (handles dots in video titles)
    FIX 2 — Text cleaning (removes fillers, short junk, spacing noise)
    FIX 3 — Separate per-language fields in metadata (query en/hi independently)
    FIX 4 — Spillover handling (long segments split across buckets proportionally)
    FIX 5 — Error handling for large/malformed JSON files
"""

import subprocess
import argparse
import time
import shutil
import sys
import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import chromadb
except ImportError:
    print("[ERROR] chromadb not found. Run: pip install chromadb")
    sys.exit(1)

SCRIPT_DIR   = Path(__file__).parent.resolve()
COOKIES_FILE = SCRIPT_DIR / "cookies.txt"
DB_PATH      = str(SCRIPT_DIR / "stream_db")

# ── FIX 2: Filler words to strip from ASR (English + Hindi) ──────────────────
FILLERS = {
    "um", "uh", "ah", "er", "hmm", "hm", "like", "okay", "ok",
    "toh", "bas", "aur", "na", "haan", "nahi", "matlab",
}


def check_deps():
    if not shutil.which("yt-dlp"):
        print("[ERROR] yt-dlp not found.  Run: pip install -U yt-dlp")
        sys.exit(1)
    if not shutil.which("aria2c"):
        print("[ERROR] aria2c not found.")
        print("        Windows: winget install aria2")
        print("        Mac:     brew install aria2")
        sys.exit(1)


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else "unknown_id"


# ── LIVE STREAM DETECTION ─────────────────────────────────────────────────────
def check_if_live(url: str) -> tuple[bool, float]:
    """
    Ask yt-dlp whether this URL is a live stream.
    Returns (is_live: bool, elapsed_seconds: float).
    elapsed_seconds = how many seconds of the stream have aired so far.
    For regular videos, elapsed_seconds = full duration.
    """
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--cookies", str(COOKIES_FILE),
        "--dump-json",
        "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not result.stdout.strip():
            return False, 0.0

        info     = json.loads(result.stdout.strip().splitlines()[0])
        is_live  = bool(info.get("is_live") or info.get("live_status") == "is_live")

        if is_live:
            # For a live stream, yt-dlp reports the start timestamp
            # We compute elapsed = now - stream_start_epoch
            start_epoch = info.get("release_timestamp") or info.get("timestamp")
            if start_epoch:
                elapsed = time.time() - float(start_epoch)
                elapsed = max(0.0, elapsed)
            else:
                # Fallback: use whatever duration yt-dlp reports (partial)
                elapsed = float(info.get("duration") or 0)
        else:
            elapsed = float(info.get("duration") or 0)

        return is_live, elapsed

    except Exception as e:
        print(f"[WARNING] Could not probe live status: {e}")
        return False, 0.0


# ── FIX 1: Robust language detection ─────────────────────────────────────────
def detect_lang(filename: str) -> str:
    """
    Reliably extract language code from yt-dlp filename by searching
    backward from .json3. Handles dots in video titles safely.
    e.g. "Some.Video.With.Dots [ID].en.json3"  ->  "en"
         "Video [ID].hi.json3"                 ->  "hi"
         "Video [ID].en-IN.json3"              ->  "en"
    """
    stem = filename
    if stem.endswith(".json3"):
        stem = stem[:-6]
    last_dot = stem.rfind(".")
    if last_dot != -1:
        candidate = stem[last_dot + 1:]
        if re.match(r'^[a-zA-Z]{2,9}(-[a-zA-Z]{2,4})?$', candidate):
            return candidate.lower().split("-")[0]   # "en-IN" -> "en"
    print(f"      [WARNING] Could not detect language from: {filename} — defaulting to 'unknown'")
    return "unknown"


# ── FIX 2: Text cleaning ──────────────────────────────────────────────────────
def clean_text(raw: str) -> str:
    text = " ".join(raw.replace("\n", " ").split())
    text = re.sub(r"[।,\.!?]{2,}", "", text)
    text = text.strip(" .,।")
    if len(text) < 3:
        return ""
    words = text.lower().split()
    if words and all(w.strip(".,।?!") in FILLERS for w in words):
        return ""
    return text


# ── ChromaDB indexer ──────────────────────────────────────────────────────────
def index_to_chromadb(out_dir: Path, video_id: str, is_live: bool, total_duration: float, embedder=None):
    """
    Parses all ASR JSON3 files and indexes into 5-sec buckets.
    Each bucket gets two ChromaDB documents:
      {video_id}_chunk_{n}_asr    → spoken ASR text  (content_type = "asr")
      {video_id}_chunk_{n}_visual → empty placeholder (content_type = "visual",
                                    filled later by image_model.py)
    """
    print(f"\n[3/3] Parsing ASR and Indexing to ChromaDB...")

    # ── Safe database wipe using native ChromaDB methods ────────────────────
    print(f"      -> Connecting to Vector DB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Try to safely delete the old collection to clear it instead of deleting the directory
    try:
        client.delete_collection("video_index")
        print("      -> Safely flushed old collection 'video_index'")
    except Exception:
        pass # Collection didn't exist yet

    json_files = list(out_dir.glob(f"*{video_id}*.json3"))
    if not json_files:
        print("      -> No JSON3 transcript files found. Skipping.")
        return

    # buckets[b_idx][lang] = [clean text segments]
    buckets = defaultdict(lambda: defaultdict(list))

    for j_path in json_files:
        lang         = detect_lang(j_path.name)                 # FIX 1
        file_size_mb = j_path.stat().st_size / (1024 * 1024)

        if file_size_mb > 10:
            print(f"      [WARNING] {j_path.name} is {file_size_mb:.1f} MB — large file")

        try:                                                      # FIX 5
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"      [SKIP] Malformed JSON in {j_path.name}: {e}")
            continue
        except Exception as e:
            print(f"      [SKIP] Could not read {j_path.name}: {e}")
            continue

        events = data.get("events", [])
        if not events:
            print(f"      [SKIP] {j_path.name} has no events.")
            continue

        print(f"      Processing {len(events)} events  |  lang='{lang}'  |  {file_size_mb:.2f} MB")

        for event in events:
            if "tStartMs" not in event:
                continue

            start_ms = event["tStartMs"]
            dur_ms   = event.get("dDurationMs", 1000)
            end_ms   = start_ms + dur_ms
            b_start  = start_ms // 5000
            b_end    = end_ms   // 5000

            # For live streams: skip any segment beyond what has aired so far
            if is_live and total_duration > 0:
                if (start_ms / 1000) > total_duration:
                    continue

            for seg in event.get("segs", []):
                if "utf8" not in seg:
                    continue

                text = clean_text(seg["utf8"])                   # FIX 2
                if not text:
                    continue

                # FIX 4: spillover across bucket boundary
                if b_end > b_start and (end_ms - start_ms) > 4000:
                    words       = text.split()
                    span_ms     = end_ms - start_ms
                    boundary_ms = (b_start + 1) * 5000
                    ratio       = (boundary_ms - start_ms) / span_ms
                    split_at    = max(1, int(len(words) * ratio))
                    first_half  = " ".join(words[:split_at])
                    second_half = " ".join(words[split_at:])
                    if first_half:
                        buckets[b_start][lang].append(first_half)
                    if second_half:
                        buckets[b_end][lang].append(second_half)
                else:
                    buckets[b_start][lang].append(text)

    # ── Build ChromaDB payload ────────────────────────────────────────────────
    documents = []
    metadatas = []
    ids       = []

    for b_idx, lang_dict in sorted(buckets.items()):

        # FIX 3: per-language metadata fields
        text_en = " ".join(lang_dict.get("en", []))
        text_hi = " ".join(lang_dict.get("hi", []))

        parts = []
        for lang, words in lang_dict.items():
            sentence = " ".join(words)
            if sentence:
                parts.append(f"[{lang.upper()}]: {sentence}")
        final_text = " | ".join(parts)

        if not final_text.strip():
            continue

        start_sec  = b_idx * 5
        end_sec    = start_sec + 5
        word_count = len(final_text.split())

        # ── ASR entry (spoken text, searchable now) ───────────────────────────
        documents.append(final_text)
        metadatas.append({
            "content_type": "asr",          # <- enables filtered search
            "start_sec":    start_sec,
            "end_sec":      end_sec,
            "video_id":     video_id,
            "text_en":      text_en,         # FIX 3: query English only
            "text_hi":      text_hi,         # FIX 3: query Hindi only
            "languages":    ",".join(sorted(lang_dict.keys())),
            "word_count":   word_count,
            "is_live":      str(is_live),
        })
        ids.append(f"{video_id}_chunk_{b_idx}_asr")

        # ── Visual placeholders — TWO per bucket (OCR + Objects separate) ────
        # Kept separate so semantic search on "satellite" (object) doesn't
        # bleed into OCR results and vice versa.

        # Placeholder 1: OCR text (screen/slide written content)
        documents.append("")
        metadatas.append({
            "content_type":  "visual_ocr",   # filter: where={"content_type":"visual_ocr"}
            "start_sec":     start_sec,
            "end_sec":       end_sec,
            "video_id":      video_id,
            "visual_ready":  "false",        # image_model.py sets this to "true"
            "ocr_text":      "",             # filled later by image_model.py
            "is_live":       str(is_live),
        })
        ids.append(f"{video_id}_chunk_{b_idx}_visual_ocr")

        # Placeholder 2: YOLO detected objects
        documents.append("")
        metadatas.append({
            "content_type":  "visual_objects",  # filter: where={"content_type":"visual_objects"}
            "start_sec":     start_sec,
            "end_sec":       end_sec,
            "video_id":      video_id,
            "visual_ready":  "false",            # image_model.py sets this to "true"
            "objects":       "",                 # filled later by image_model.py
            "is_live":       str(is_live),
        })
        ids.append(f"{video_id}_chunk_{b_idx}_visual_objects")

    if not documents:
        print("      -> Transcripts were empty after cleaning. Nothing to index.")
        return

    n_buckets = len(documents) // 3   # each bucket = 1 ASR + 1 visual_ocr + 1 visual_objects

    print(f"      -> Connecting to Vector DB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    # ── GPU-accelerated embedding via sentence-transformers + PyTorch CUDA ────
    # Bypasses onnxruntime entirely (avoids TensorRT/cublas DLL errors).
    # Requires: pip install sentence-transformers torch (with CUDA build)
    # Setup:    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ef = embedder
    device = "cpu"
    if ef is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model directly onto GPU — bypasses onnxruntime/TensorRT completely
            _st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

            # Wrap as a ChromaDB-compatible embedding function
            class _DirectGPUEmbedder:
                def name(self) -> str:
                    return "_DirectGPUEmbedder"

                def embed_query(self, input: str | list[str], **kwargs) -> list[list[float]]:
                    if isinstance(input, str):
                        input = [input]
                    return _st_model.encode(input).tolist()

                def embed_documents(self, input: list[str], **kwargs) -> list[list[float]]:
                    if isinstance(input, str):
                        input = [input]
                    return _st_model.encode(input).tolist()

                def __call__(self, input):
                    return _st_model.encode(
                        input,
                        batch_size=256,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).tolist()

            ef = _DirectGPUEmbedder()

            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                print(f"      -> Embedding device: CUDA  [{gpu_name}]  <-- GPU active!")
            else:
                print("      -> Embedding device: CPU  (torch installed but no CUDA GPU found)")
                print("         Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")

        except ImportError as e:
            print(f"      -> sentence-transformers not found ({e})")
            print("         Falling back to default onnxruntime CPU embedder")
            print("         To fix: pip install sentence-transformers")
            print("                 pip install torch --index-url https://download.pytorch.org/whl/cu121")

    collection = (
        client.get_or_create_collection(name="video_index", embedding_function=ef)
        if ef else
        client.get_or_create_collection(name="video_index")
    )

    # Upsert in batches — larger batch = more GPU parallelism (sweet spot ~256)
    batch_size = 256
    total      = len(documents)
    t_embed    = time.time()
    for i in range(0, total, batch_size):
        collection.upsert(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print(f"      -> Upserted {min(i + batch_size, total)}/{total} entries...")

    embed_secs = time.time() - t_embed
    print(f"      -> Done! {n_buckets} buckets indexed in {embed_secs:.1f}s")
    print(f"         ({n_buckets} ASR + {n_buckets} visual_ocr + {n_buckets} visual_objects placeholders)")
    print("      -> Query tips:")
    print('         Speech only   : where={"content_type": "asr"}')
    print('         OCR/slides    : where={"content_type": "visual_ocr"}')
    print('         YOLO objects  : where={"content_type": "visual_objects"}')
    print("         Full search   : no filter")


# ── Main download orchestrator ────────────────────────────────────────────────
def download_video(url: str, workers: int, out_dir: Path, embedder=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start      = time.time()
    out_template = str(out_dir / "%(title)s [%(id)s].%(ext)s")
    video_id     = extract_video_id(url)

    # ── Probe: regular video or live stream? ──────────────────────────────────
    print(f"\n[0/3] Probing URL...")
    is_live, elapsed_secs = check_if_live(url)

    if is_live:
        print(f"      -> LIVE STREAM detected!")
        print(f"      -> Capturing from stream start up to now ({elapsed_secs/60:.1f} min aired)")
    else:
        print(f"      -> Regular video  (duration: {elapsed_secs/60:.1f} min)")

    print(f"\n{'='*60}")
    print(f"  FastFetch  |  Video + ASR + DB Indexer")
    print(f"{'='*60}")
    print(f"  URL      : {url[:55]}{'...' if len(url)>55 else ''}")
    print(f"  Video ID : {video_id}")
    print(f"  Mode     : {'LIVE STREAM' if is_live else 'Regular Video'}")
    if is_live:
        print(f"  Capture  : 0s -> {elapsed_secs:.0f}s ({elapsed_secs/60:.1f} min so far)")
    print(f"  Output   : {out_dir.absolute()}")
    print(f"  Workers  : {workers} parallel connections")
    print(f"  DB Path  : {DB_PATH}")
    print(f"{'='*60}\n")

    # ── STEP 1: Video ─────────────────────────────────────────────────────────
    print("[1/3] Downloading video...\n")

    video_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--cookies",        str(COOKIES_FILE),
        "--extractor-args", "youtube:client=ios,android",
        "--extractor-args", "youtube:client=android,web",
        "--force-ipv4",
        "-f",               "bv*[height<=720][ext=mp4]/bv*",
        "-N",               str(workers),
        "--no-write-subs",
        "--no-write-auto-subs",
    ]

    # Live stream: cap download at elapsed time so we only get what's aired
    if is_live and elapsed_secs > 0:
        video_cmd += ["--download-sections", f"*0-{int(elapsed_secs)}"]

    video_cmd += ["-o", out_template, url]

    video_ok = False
    try:
        subprocess.run(video_cmd, check=True)
        video_ok = True
        print("\n[1/3] Video saved.")
    except subprocess.CalledProcessError:
        print("\n[ERROR] Video download failed.")

    # ── STEP 2: ASR ───────────────────────────────────────────────────────────
    if video_ok:
        print("\n[2/3] Fetching ASR auto-captions (en / en-IN / hi)...\n")

        asr_cmd = [
            "yt-dlp",
            "--no-playlist",
            "--cookies",        str(COOKIES_FILE),
            "--extractor-args", "youtube:client=ios,android",
            "--force-ipv4",
            "--skip-download",
            "--write-auto-subs",
            "--sub-langs",      "en,en-IN,hi",
            "--sub-format",     "json3",
            "--no-write-subs",
            "-o",               out_template,
            url,
        ]

        try:
            subprocess.run(asr_cmd, check=True)
            print("[2/3] ASR saved.")
        except subprocess.CalledProcessError:
            print("[2/3] WARNING: ASR fetch failed (429 or not available).")
            print("      Video is still saved — captions are missing.")

        # ── STEP 3: Index ─────────────────────────────────────────────────────
        index_to_chromadb(out_dir, video_id, is_live, elapsed_secs, embedder=embedder)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s  |  Output: {out_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    check_deps()

    ap = argparse.ArgumentParser(
        description="Download video/livestream + ASR and index into ChromaDB (5-sec buckets)."
    )
    ap.add_argument("url",
                    help="YouTube URL (regular video or live stream)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel connections for video (default: 16)")
    ap.add_argument("--out",     type=str, default="downloads",
                    help="Output folder (default: ./downloads)")

    args = ap.parse_args()
    download_video(args.url, args.workers, Path(args.out))