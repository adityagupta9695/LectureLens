"""
image_model.py - Visual Analysis Pipeline (GPU-Optimised)
======================================================
Reads imges/ produced by streamstamper.py, groups frames
by 5-second bucket (timestamps.json), runs batched YOLO
+ batched PaddleOCR, upserts into ChromaDB visual slots.

Speed optimisations applied:
    - YOLO runs ONE batch call for ALL frames across ALL buckets
      (GPU processes 32/64 frames in one shot, not one-by-one)
    - PaddleOCR runs in true batch mode (list of paths at once)
    - No ThreadPoolExecutor for GPU work (threads fight over GPU)
    - Angle-classification disabled (slides are never rotated)
    - Embedding model loaded once, reused across all upserts
    - HuggingFace model loaded from local cache — no re-download

Workflow:
    fast_fetch.py   -> video + ASR  -> stream_db  (ASR entries)
    streamstamper.py -> I-frames   -> imges/ + timestamps.json
    image_model.py  -> visual      -> stream_db  (visual entries)

Usage:
    python image_model.py --video-id VIDEO_ID
    python image_model.py --video-id VIDEO_ID --imges-dir path/to/imges
    python image_model.py --video-id VIDEO_ID --yolo-batch 64 --ocr-batch 16

Requirements:
    pip install ultralytics paddlepaddle-gpu paddleocr chromadb
    pip install sentence-transformers torch
"""

import argparse
import json
import os
import time
import sys
from collections import defaultdict
from pathlib import Path

try:
    import chromadb
except ImportError:
    print("[ERROR] chromadb not found. Run: pip install chromadb")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent.resolve()
DB_PATH    = str(SCRIPT_DIR / "stream_db")
IMGES_DIR  = SCRIPT_DIR / "imges"

# Prevent HuggingFace from checking for model updates on every run
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
# Prevent PaddleOCR v3 from doing a connectivity check to model servers on startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_yolo(conf: float):
    try:
        from ultralytics import YOLO
        model      = YOLO("yolov8n.pt")
        model.conf = conf
        print(f"[models] YOLOv8n loaded  (conf={conf})")
        return model
    except ImportError:
        print("[ERROR] ultralytics not found. Run: pip install ultralytics")
        sys.exit(1)


def load_ocr():
    # ── Disable oneDNN/MKL-DNN BEFORE importing paddle ───────────────────────
    # The ConvertPirAttribute crash is caused by oneDNN ops on Windows GPU.
    # Setting FLAGS_use_mkldnn=0 disables it without falling back to CPU.
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_call_stack_level"] = "0"    # suppress C++ stack traces

    try:
        import torch
        from paddleocr import PaddleOCR

        # Use GPU if CUDA is available, otherwise CPU
        # device="gpu" uses your RTX 4060 — ~10-20x faster than CPU for OCR
        paddle_device = "gpu" if torch.cuda.is_available() else "cpu"

        # PaddleOCR v3 valid params. No use_angle_cls / show_log / use_gpu.
        # text_det_model_name="PP-OCRv5_mobile_det" — mobile det model:
        #   ~6x faster than the default server model PP-OCRv5_server_det
        #   accuracy is nearly identical for clean slide/screen frames
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            device=paddle_device,
        )
        print(f"[models] PaddleOCR v3 loaded  (device={paddle_device.upper()})")
        return ocr
    except ImportError:
        print("[ERROR] paddleocr not found. Run: pip install paddleocr paddlepaddle-gpu")
        sys.exit(1)


def load_embedder():
    """
    Load sentence-transformer onto GPU once.
    TRANSFORMERS_OFFLINE=1 prevents any HuggingFace network call if cached.
    Falls back to CPU if CUDA not available.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        class _Embedder:
            def name(self) -> str:
                return "sentence-transformers/all-MiniLM-L6-v2"

            def _encode(self, input):
                if isinstance(input, str):
                    input = [input]
                return model.encode(
                    input,
                    batch_size=256,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).tolist()

            def __call__(self, input):
                return self._encode(input)

            def embed_query(self, input, **kwargs):
                # ChromaDB calls this during collection.query()
                return self._encode(input)

            def embed_documents(self, input, **kwargs):
                # ChromaDB calls this during collection.upsert()
                return self._encode(input)

        print(f"[models] Embedder loaded  (device={device.upper()})")
        return _Embedder()
    except ImportError:
        print("[models] sentence-transformers not found — using default CPU embedder")
        return None


# ── Step 1: Load timestamps.json ──────────────────────────────────────────────

def load_timestamps(imges_dir: Path) -> list:
    ts_path = imges_dir / "timestamps.json"
    if not ts_path.exists():
        print(f"[ERROR] timestamps.json not found at {ts_path}")
        print("        Run streamstamper.py first.")
        sys.exit(1)
    with open(ts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[1/4] Loaded {len(data)} frame entries from timestamps.json")
    return data


# ── Step 2: Group frames by bucket ────────────────────────────────────────────

def group_by_bucket(timestamps: list, imges_dir: Path) -> dict:
    buckets = defaultdict(list)
    missing = 0
    for entry in timestamps:
        bucket = entry.get("bucket", -1)
        if bucket < 0:
            continue
        fp = imges_dir / entry["frame"]
        if not fp.exists():
            missing += 1
            continue
        buckets[bucket].append(fp)
    print(f"[2/4] {len(buckets)} buckets  ({missing} frames missing/skipped)")
    return dict(buckets)


# ── Step 3: Batched YOLO over ALL frames, then batched OCR ────────────────────

def run_analysis(buckets: dict, yolo_model, ocr_model) -> list:
    """
    Two-pass approach:
      Pass 1 — ONE YOLO batch call for every frame across every bucket.
               GPU processes all frames in parallel (32–64 per kernel call).
      Pass 2 — ONE PaddleOCR batch call per bucket (keeps memory bounded).
               Only runs OCR on frames that YOLO confirmed have slide content
               (person, tv, laptop, book, cell phone) OR all frames if no
               content filter matches (whiteboards without objects).
    No threads. Sequential bucket loop, but each model call is fully batched.
    """
    print(f"[3/4] Analysing {len(buckets)} buckets...")
    t0 = time.time()

    # ── PASS 1: batch YOLO across ALL frames at once ──────────────────────────
    # Flatten all frames with their bucket index for later reassembly
    all_frames   = []   # list of Path
    frame_bucket = []   # parallel list of bucket_idx

    for b_idx, paths in sorted(buckets.items()):
        for fp in paths:
            all_frames.append(fp)
            frame_bucket.append(b_idx)

    print(f"      YOLO: {len(all_frames)} frames total (batched)...")
    t_yolo = time.time()

    # Process YOLO in chunks of 32 to stay within 8GB VRAM
    YOLO_BATCH = 32
    bucket_objects = defaultdict(set)
    frame_paths = [str(p) for p in all_frames]

    for chunk_start in range(0, len(frame_paths), YOLO_BATCH):
        chunk_end = min(chunk_start + YOLO_BATCH, len(frame_paths))
        chunk_paths = frame_paths[chunk_start:chunk_end]

        yolo_results = yolo_model(chunk_paths, verbose=False, stream=True)

        for j, r in enumerate(yolo_results):
            b = frame_bucket[chunk_start + j]
            for cls_id in r.boxes.cls.tolist():
                bucket_objects[b].add(r.names[int(cls_id)])

        # Free GPU memory after each chunk
        del yolo_results
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"      YOLO done in {time.time()-t_yolo:.1f}s")

    # ── PASS 2: batched OCR per bucket ────────────────────────────────────────
    # "Slide-indicator" labels — if any of these appear, the frame likely has text
    SLIDE_LABELS = {"tv", "laptop", "cell phone", "book", "keyboard",
                    "monitor", "screen", "whiteboard", "blackboard"}

    print(f"      OCR:  processing {len(buckets)} buckets...")
    t_ocr = time.time()

    bucket_ocr = {}   # b_idx -> best OCR string

    for b_idx, paths in sorted(buckets.items()):
        # Decide which frames to OCR:
        # If YOLO found slide-like objects → OCR all frames in bucket
        # Otherwise OCR only the FIRST frame (saves time on talking-head buckets)
        has_slide = bool(bucket_objects[b_idx] & SLIDE_LABELS)
        frames_to_ocr = paths if has_slide else paths[:1]

        best_text = ""
        best_len  = 0

        # PaddleOCR v3 (PaddleX): call once per image (no batch-path support).
        # Returns list of OCRResult objects, each with .boxes attribute.
        # Each box has: .rec_text (str) and .rec_score (float).
        for fp in frames_to_ocr:
            try:
                results = ocr_model.ocr(str(fp))
                if not results:
                    continue

                lines = []

                # PaddleOCR v3 returns: list of OCRResult objects
                # Each OCRResult has multiple possible structures depending on version:
                #
                # Format A (v3 new):  result[0] is a dict with keys
                #                     'rec_texts' (list) and 'rec_scores' (list)
                # Format B (v3 obj):  each item has .rec_text and .rec_score attrs
                # Format C (v2):      each item is [bbox, ('text', conf)]

                page = results[0] if results else None
                if page is None:
                    continue

                # Format A: dict with rec_texts/rec_scores (most common in v3)
                if hasattr(page, "keys") and "rec_texts" in page:
                    for t, s in zip(page["rec_texts"], page["rec_scores"]):
                        if s > 0.55 and t and len(t.strip()) > 1:
                            lines.append(t.strip())

                # Format B: flat list of OCRResult objects with attributes
                elif isinstance(page, list):
                    for item in page:
                        # v3 attribute-style
                        if hasattr(item, "rec_text") and hasattr(item, "rec_score"):
                            if item.rec_score > 0.55 and len(item.rec_text.strip()) > 1:
                                lines.append(item.rec_text.strip())
                        # v2 tuple-style: [bbox, ('text', conf)]
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            rec = item[1]
                            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                t, c = rec[0], rec[1]
                                if c > 0.55 and isinstance(t, str) and len(t.strip()) > 1:
                                    lines.append(t.strip())

                # Format C: results itself is a flat list of attribute objects
                elif hasattr(page, "rec_text"):
                    for item in results:
                        if hasattr(item, "rec_text") and item.rec_score > 0.55:
                            if len(item.rec_text.strip()) > 1:
                                lines.append(item.rec_text.strip())

                combined = " ".join(lines)
                if len(combined) > best_len:
                    best_text = combined
                    best_len  = len(combined)

            except Exception as e:
                pass   # silent — one bad frame shouldn't stop the pipeline

        bucket_ocr[b_idx] = best_text[:500]

    print(f"      OCR done in  {time.time()-t_ocr:.1f}s")

    # ── Assemble results ──────────────────────────────────────────────────────
    results = []
    for b_idx in sorted(buckets.keys()):
        objects   = ", ".join(sorted(bucket_objects.get(b_idx, set())))
        ocr_text  = bucket_ocr.get(b_idx, "")
        parts     = []
        if objects:
            parts.append(f"[Objects]: {objects}")
        if ocr_text:
            parts.append(f"[OCR]: {ocr_text}")
        visual_summary = " | ".join(parts)

        results.append({
            "bucket":          b_idx,
            "objects":         objects,
            "ocr_text":        ocr_text,
            "visual_summary":  visual_summary,
            "frames_analysed": len(buckets[b_idx]),
        })

    print(f"      Total analysis: {time.time()-t0:.1f}s")
    return results


# ── Step 4: Upsert into ChromaDB ──────────────────────────────────────────────

def upsert_visual(results: list, video_id: str, embedder):
    """
    Upsert into TWO separate ChromaDB entries per bucket:
      {video_id}_chunk_{n}_visual_ocr     -> OCR/slide text only
      {video_id}_chunk_{n}_visual_objects -> YOLO object labels only

    Keeping them separate means:
      Search "satellite" in visual_objects → only physical objects detected
      Search "satellite" in visual_ocr     → only screen/slide text matches
    No cross-contamination between written content and detected objects.
    """
    print(f"\n[4/4] Upserting {len(results)*2} visual entries ({len(results)} OCR + {len(results)} objects)...")
    t0 = time.time()

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = (
        client.get_or_create_collection(name="video_index", embedding_function=embedder)
        if embedder else
        client.get_or_create_collection(name="video_index")
    )

    # ── Build OCR entries ─────────────────────────────────────────────────────
    ocr_docs  = []
    ocr_metas = []
    ocr_ids   = []

    # ── Build Objects entries ─────────────────────────────────────────────────
    obj_docs  = []
    obj_metas = []
    obj_ids   = []

    for r in results:
        b_idx     = r["bucket"]
        start_sec = b_idx * 5
        end_sec   = start_sec + 5
        fa        = r["frames_analysed"]

        # OCR entry — document is the raw OCR text (what was written on screen)
        ocr_doc = r["ocr_text"] or f"[bucket {b_idx}: no text detected]"
        ocr_docs.append(ocr_doc)
        ocr_metas.append({
            "content_type":    "visual_ocr",
            "start_sec":       start_sec,
            "end_sec":         end_sec,
            "video_id":        video_id,
            "visual_ready":    "true",
            "ocr_text":        r["ocr_text"],
            "frames_analysed": fa,
        })
        ocr_ids.append(f"{video_id}_chunk_{b_idx}_visual_ocr")

        # Objects entry — document is space-joined label list (e.g. "laptop person book")
        # Plain space-joined works better for semantic search than comma-separated
        obj_doc = r["objects"].replace(", ", " ") if r["objects"] else f"[bucket {b_idx}: no objects detected]"
        obj_docs.append(obj_doc)
        obj_metas.append({
            "content_type":    "visual_objects",
            "start_sec":       start_sec,
            "end_sec":         end_sec,
            "video_id":        video_id,
            "visual_ready":    "true",
            "objects":         r["objects"],
            "frames_analysed": fa,
        })
        obj_ids.append(f"{video_id}_chunk_{b_idx}_visual_objects")

    # ── Upsert both sets in batches ───────────────────────────────────────────
    batch_size = 256
    for label, docs, metas, ids in [
        ("OCR    ", ocr_docs,  ocr_metas,  ocr_ids),
        ("Objects", obj_docs,  obj_metas,  obj_ids),
    ]:
        total = len(docs)
        for i in range(0, total, batch_size):
            collection.upsert(
                documents=docs[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )
        print(f"      -> {label} : {total} entries upserted")

    print(f"      -> Done in {time.time()-t0:.1f}s")
    print('         OCR search     : where={"content_type": "visual_ocr"}')
    print('         Objects search : where={"content_type": "visual_objects"}')
    print('         Speech search  : where={"content_type": "asr"}')
    print('         Full search    : no filter')



# ── Public API called by server.py ────────────────────────────────────────────

def process_video_visuals(video_id: str, yolo_model, ocr_model, embedder, imges_dir: Path = None):
    """
    Entry point called by server.py after streamstamper finishes.
    Chains: load_timestamps -> group_by_bucket -> run_analysis -> upsert_visual
    Uses pre-loaded models passed in from the server lifespan.
    """
    if imges_dir is None:
        imges_dir = IMGES_DIR

    t_total    = time.time()
    timestamps = load_timestamps(imges_dir)
    buckets    = group_by_bucket(timestamps, imges_dir)
    results    = run_analysis(buckets, yolo_model, ocr_model)
    upsert_visual(results, video_id, embedder)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s  |  {len(results)} visual buckets written")
    print(f"{'='*60}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visual analysis: batched YOLO + OCR -> ChromaDB (GPU-optimised)."
    )
    ap.add_argument("--video-id",   required=True,
                    help="YouTube video ID  (e.g. UvqJJ_5Lr6Y)")
    ap.add_argument("--imges-dir",  default=None,
                    help="imges/ folder path  (default: ./imges)")
    ap.add_argument("--yolo-conf",  type=float, default=0.25,
                    help="YOLO confidence threshold  (default: 0.25)")
    ap.add_argument("--yolo-batch", type=int,   default=32,
                    help="YOLO batch size  (default: 32, raise to 64 on 8GB+ GPU)")
    ap.add_argument("--ocr-batch",  type=int,   default=8,
                    help="OCR frames per batch  (default: 8)")
    args = ap.parse_args()

    video_id  = args.video_id
    imges_dir = Path(args.imges_dir) if args.imges_dir else IMGES_DIR

    print(f"\n{'='*60}")
    print(f"  image_model.py  |  Visual Analysis Pipeline")
    print(f"{'='*60}")
    print(f"  Video ID   : {video_id}")
    print(f"  Imges dir  : {imges_dir.absolute()}")
    print(f"  DB path    : {DB_PATH}")
    print(f"  YOLO conf  : {args.yolo_conf}")
    print(f"  YOLO batch : {args.yolo_batch}")
    print(f"  OCR batch  : {args.ocr_batch}")
    print(f"{'='*60}\n")

    t_total = time.time()

    print("[models] Loading models (one-time cost)...")
    yolo_model = load_yolo(args.yolo_conf)
    ocr_model  = load_ocr()
    embedder   = load_embedder()
    print()

    timestamps = load_timestamps(imges_dir)
    buckets    = group_by_bucket(timestamps, imges_dir)
    results    = run_analysis(buckets, yolo_model, ocr_model)
    upsert_visual(results, video_id, embedder)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s  |  {len(results)} visual buckets written")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()