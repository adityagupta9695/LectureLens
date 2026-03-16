"""
StreamStamper - Ultra Fast Parallel I-Frame Extractor
======================================================
Works with YouTube URLs AND local video files.
Optimized for LONG streams (4-10 hour videos).

Cookie setup (YouTube only, one-time):
    1. Install "Get cookies.txt LOCALLY" Chrome extension
    2. Go to youtube.com while logged into your Google account
    3. Click "Export All Cookies" (the large button at top — NOT "Export")
    4. Save as cookies.txt next to this script (should be 300+ KB)

Usage:
    python streamstamper.py "https://youtube.com/watch?v=VIDEOID"
    python streamstamper.py "C:\\Videos\\myvideo.mp4"
    python streamstamper.py "C:\\Videos\\myvideo.mkv" --workers 16 --seg-duration 300
    python streamstamper.py "URL_or_PATH" --max 100
    python streamstamper.py "URL_or_PATH" --lowres 1

Requirements:
    pip install yt-dlp
    winget install ffmpeg
"""

import sys
import argparse
import subprocess
import shutil
import time
import os
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

print_lock = Lock()

SCRIPT_DIR   = Path(__file__).parent.resolve()
COOKIES_FILE = SCRIPT_DIR / "cookies.txt"
MIN_COOKIES_KB = 10

# Video file extensions recognised as local files
LOCAL_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv",
    ".m4v", ".ts", ".mts", ".m2ts", ".mpg", ".mpeg", ".3gp",
    ".ogv", ".vob", ".divx", ".xvid", ".hevc",
}


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def is_local_file(source: str) -> bool:
    """Return True if source looks like a local file path rather than a URL."""
    p = Path(source)
    # Explicit check: exists on disk
    if p.exists() and p.is_file():
        return True
    # Heuristic: has a known video extension and no http/https scheme
    if not source.startswith(("http://", "https://", "rtmp://", "rtsp://")):
        if p.suffix.lower() in LOCAL_VIDEO_EXTENSIONS:
            return True
    return False


# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps(need_ytdlp: bool = True):
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg   ->  winget install ffmpeg")
    if need_ytdlp and not shutil.which("yt-dlp"):
        missing.append("yt-dlp   ->  pip install yt-dlp")
    if missing:
        print("\n[ERROR] Missing required tools:")
        for m in missing:
            print(f"  * {m}")
        sys.exit(1)


def get_cookie_args() -> list:
    if COOKIES_FILE.exists():
        size_kb = COOKIES_FILE.stat().st_size / 1024
        if size_kb >= MIN_COOKIES_KB:
            print(f"[cookies] cookies.txt  ({size_kb:.0f} KB)  -> OK")
            return ["--cookies", str(COOKIES_FILE)]
        else:
            print(f"[cookies] WARNING: cookies.txt is only {size_kb:.1f} KB — use 'Export All Cookies'")
    else:
        print(f"[cookies] No cookies.txt found — YouTube videos may require it")
    return []


def ytdlp_base(cookie_args: list) -> list:
    return [
        "yt-dlp",
        "--no-playlist",
        "--no-check-certificates",
        "--extractor-retries", "3",
    ] + cookie_args


# ── Local file: probe helpers ────────────────────────────────────────────────

def find_ffprobe() -> str:
    """
    Locate ffprobe. Checks (in order):
      1. Same folder as this script  (e.g. you dropped ffprobe.exe next to streamstamper.py)
      2. Same folder as ffmpeg.exe   (standard Windows ffmpeg install)
      3. PATH search
    """
    search_dirs = []

    # 1. Script's own directory
    search_dirs.append(SCRIPT_DIR)

    # 2. Same folder as ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        search_dirs.append(Path(ffmpeg_path).parent)

    for d in search_dirs:
        for ext in ("", ".exe"):
            candidate = d / ("ffprobe" + ext)
            if candidate.exists():
                return str(candidate)

    # 3. PATH
    found = shutil.which("ffprobe")
    return found if found else "ffprobe"


def probe_local_file(file_path: str) -> tuple:
    """
    Probe duration, codec, height, fps from a local file.
    Method 1: ffprobe JSON (accurate).
    Method 2: parse ffmpeg -i stderr (fallback, always works if ffmpeg is present).
    Returns (duration_float, vcodec_str, height_str, fps_str).
    """
    fp = find_ffprobe()

    # ── Method 1: ffprobe JSON (only if ffprobe is functional) ──────────────
    try:
        # Quick sanity-check: does ffprobe actually run?
        test = subprocess.run([fp, "-version"], capture_output=True, timeout=5)
        if test.returncode == 0:
            cmd = [
                fp, "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-select_streams", "v:0",
                file_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                data     = json.loads(r.stdout)
                fmt      = data.get("format", {})
                streams  = data.get("streams", [])
                duration = float(fmt.get("duration") or 0)
                vcodec, height, fps = "unknown", "?", "?"
                if streams:
                    s      = streams[0]
                    vcodec = s.get("codec_name", "unknown")
                    height = str(s.get("height", "?"))
                    rfr    = s.get("r_frame_rate", "")
                    if rfr and "/" in rfr:
                        n, d = rfr.split("/")
                        if int(d) > 0:
                            fps = f"{int(n)/int(d):.2f}"
                if duration > 0:
                    return duration, vcodec, height, fps
        else:
            print(f"[1/4] ffprobe not functional (rc={test.returncode}) — skipping, using ffmpeg fallback")
    except Exception as e:
        print(f"[1/4] ffprobe skipped ({e})")

    # ── Method 2: ffmpeg -i (stderr contains file info when no output given) ─
    # IMPORTANT: do NOT pass -v error or -v quiet — that suppresses the info.
    # ffmpeg prints "At least one output file must be specified" to stderr but
    # also prints the full file info. We intentionally trigger that non-zero exit.
    try:
        r2 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", file_path],
            capture_output=True, text=True, timeout=30
        )
        # ffmpeg writes file info to stderr (exit code 1 is expected — no output file)
        combined = r2.stderr + r2.stdout

        duration, vcodec, height, fps = 0.0, "unknown", "?", "?"

        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", combined)
        if m:
            duration = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))

        # Match: "Stream #0:0(und): Video: h264 (High), yuv420p, 1920x1080, 30 fps"
        vm = re.search(
            r"Stream #[^:]+: Video: (\w+)[^\n]*?(\d{3,4})x(\d{3,4})[^\n]*?([\d.]+)\s*fps",
            combined
        )
        if vm:
            vcodec = vm.group(1)
            w, h   = int(vm.group(2)), int(vm.group(3))
            height = str(min(w, h))
            fps    = vm.group(4)

        if duration > 0:
            return duration, vcodec, height, fps

        print(f"[1/4] ffmpeg info output was empty or unparseable")
        print(f"[1/4] raw stderr: {r2.stderr[:400]!r}")

    except Exception as e:
        print(f"[1/4] ffmpeg fallback failed ({e})")

    return 0.0, "unknown", "?", "?"


def resolve_local_file(source: str) -> tuple:
    """
    Validate a local file and probe its metadata.
    Returns (absolute_path, duration_seconds, codec_string).
    """
    p = Path(source).resolve()
    if not p.exists():
        print(f"\n[ERROR] File not found: {p}")
        print(f"  Tip: wrap the path in double quotes if it contains spaces")
        sys.exit(1)
    if not p.is_file():
        print(f"\n[ERROR] Not a file: {p}")
        sys.exit(1)

    print(f"[1/4] Local file : {p.name}")
    print(f"[1/4] Full path  : {p}")
    print(f"[1/4] ffprobe    : {find_ffprobe()}")

    duration, vcodec, height, fps = probe_local_file(str(p))

    print(f"[1/4] Stream     : {vcodec}  {height}p  {fps}fps")

    if duration > 0:
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = int(duration % 60)
        print(f"[1/4] Duration   : {h:02d}:{m:02d}:{s:02d}  ({duration:.0f}s total)")
    else:
        print(f"[1/4] Duration   : COULD NOT PROBE — parallelism disabled (1 worker)")
        print(f"[1/4]   To fix: confirm ffprobe is installed alongside ffmpeg")

    return str(p), duration, str(vcodec)


# ── YouTube: format discovery ─────────────────────────────────────────────────

STORYBOARD_PATTERNS = re.compile(
    r'\bsb\d*\b|storyboard|mhtml|text|audio only|images',
    re.IGNORECASE
)

def pick_best_format(url: str, cookie_args: list) -> str:
    """
    Query available formats and pick the best REAL video stream.
    Excludes storyboards, audio-only, text tracks, and mhtml formats.
    Prefers H264 720p > VP9 720p > H264 1080p > VP9 1080p > anything real.
    """
    print("[1/4] Querying available formats...")

    cmd = ytdlp_base(cookie_args) + [
        "--list-formats", "--no-warnings", "--no-playlist", url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Sign in" in stderr or "bot" in stderr.lower():
            _print_cookie_help()
            sys.exit(1)
        print("[1/4] Could not list formats — using safe fallback")
        return "bestvideo[height<=720]/bestvideo/best"

    candidates = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") or "---" in line or line.startswith("ID "):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        fmt_id = parts[0]
        if not re.match(r'^[\w\-\.]+$', fmt_id):
            continue

        # Skip storyboard format IDs
        if re.match(r'^sb\d*$', fmt_id, re.IGNORECASE):
            continue

        line_lower = line.lower()

        if any(kw in line_lower for kw in [
            "storyboard", "mhtml", "audio only", "images", "text", "drc",
        ]):
            continue

        has_res    = bool(re.search(r'\d{3,4}x\d{3,4}', line))
        has_height = bool(re.search(r'\b\d{3,4}p\b', line))
        is_video   = "video only" in line_lower or "video" in line_lower

        if not (has_res or has_height or is_video):
            continue

        m = re.search(r'\b(\d{3,4})p\b', line)
        if not m:
            m = re.search(r'\d{3,4}x(\d{3,4})', line)
        height = int(m.group(1)) if m else 0

        # Skip suspiciously low heights (storyboard artifact)
        if 0 < height < 144:
            continue

        is_h264 = bool(re.search(r'\bavc1?\b|\bh\.?264\b', line_lower))
        is_vp9  = bool(re.search(r'\bvp0?9\b', line_lower))
        codec_score = 3 if is_h264 else (2 if is_vp9 else 1)
        is_combined = "video only" not in line_lower and is_video

        candidates.append({
            "id": fmt_id, "height": height,
            "codec": codec_score, "combined": is_combined,
        })

    if not candidates:
        print("[1/4] No real video formats found — using safe fallback")
        return "bestvideo[height<=720]/bestvideo/best"

    def score(c):
        h = c["height"]
        if   h == 720:  h_score = 200
        elif h == 1080: h_score = 180
        elif h == 480:  h_score = 150
        elif h == 360:  h_score = 100
        elif h == 1440: h_score = 140
        elif h == 2160: h_score = 120
        elif h == 0:    h_score = 50
        else:           h_score = h // 10
        combined_penalty = -500 if c["combined"] else 0
        return c["codec"] * 1000 + h_score + combined_penalty

    candidates.sort(key=score, reverse=True)

    print(f"[1/4] Available video formats (top picks):")
    for c in candidates[:5]:
        codec_name = "H264" if c["codec"]==3 else ("VP9" if c["codec"]==2 else "AV1/other")
        marker = " <-- SELECTED" if c == candidates[0] else ""
        print(f"       ID={c['id']:6s}  {c['height']}p  {codec_name}{marker}")

    best = candidates[0]
    codec_name = "H264" if best["codec"]==3 else ("VP9" if best["codec"]==2 else "AV1/other")
    if best["codec"] < 3:
        print(f"[1/4] NOTE: No H264 stream — using {codec_name} (slightly slower, still works)")

    return best["id"]


def resolve_youtube(url: str, cookie_args: list) -> tuple[str, float, str]:
    fmt_id = pick_best_format(url, cookie_args)

    print(f"\n[1/4] Resolving CDN URL (format {fmt_id})...")
    cmd = ytdlp_base(cookie_args) + [
        "--dump-json", "--no-warnings", "-f", fmt_id, url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if not stdout or result.returncode != 0:
        print(f"[1/4] Format {fmt_id} failed — retrying with bestvideo[height<=720]/bestvideo/best...")
        cmd2 = ytdlp_base(cookie_args) + [
            "--dump-json", "--no-warnings",
            "-f", "bestvideo[height<=720]/bestvideo/best", url,
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

    if not stdout:
        error_lines = [l for l in stderr.splitlines() if "ERROR:" in l]
        print(f"\n[ERROR] Could not resolve stream URL.")
        for ln in (error_lines or [stderr[:400]]):
            print(f"  {ln.strip()}")
        if "Sign in" in stderr or "bot" in stderr.lower():
            _print_cookie_help()
        else:
            print("  Try: pip install yt-dlp --upgrade")
        sys.exit(1)

    return _parse_yt_json(stdout)


def _parse_yt_json(stdout: str) -> tuple[str, float, str]:
    try:
        info = json.loads(stdout.splitlines()[0])
    except json.JSONDecodeError:
        info = json.loads(stdout)

    direct_url = info.get("url", "")
    if not direct_url:
        for f in (info.get("requested_formats") or info.get("formats") or []):
            if f.get("url"):
                direct_url = f["url"]
                break

    if not direct_url:
        raise ValueError("No stream URL in yt-dlp JSON")

    if "i.ytimg.com" in direct_url or "storyboard" in direct_url.lower():
        raise ValueError(f"yt-dlp returned a storyboard URL, not real video: {direct_url[:100]}")

    duration = float(info.get("duration") or 0)
    title    = info.get("title", "Unknown")
    vcodec   = info.get("vcodec") or "unknown"
    height   = info.get("height") or "?"
    fps      = info.get("fps") or "?"

    try:
        if float(fps) < 1.0:
            raise ValueError(f"fps={fps} too low — storyboard detected")
    except (TypeError, ValueError) as e:
        if "storyboard" in str(e) or "fps" in str(e):
            raise

    preview = direct_url[:80] + ("..." if len(direct_url) > 80 else "")
    print(f"[1/4] Title    : {title[:70]}")
    print(f"[1/4] Stream   : {vcodec}  {height}p  {fps}fps")
    print(f"[1/4] URL      : {preview}")

    if duration > 0:
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = int(duration % 60)
        print(f"[1/4] Duration : {h:02d}:{m:02d}:{s:02d}  ({duration:.0f}s total)")
    else:
        print("[1/4] Duration : unknown (single segment mode)")

    if vcodec and not str(vcodec).startswith("avc"):
        print(f"[1/4] Codec    : {vcodec} (not H264 — decode slightly slower, still fine)")

    return direct_url, duration, str(vcodec)


def _print_cookie_help():
    print(f"""
  [FIX] YouTube requires authentication:
    1. Open Chrome and go to youtube.com (stay logged in)
    2. Click "Get cookies.txt LOCALLY" extension
    3. Click "Export All Cookies" (the LARGE button at top)
    4. Save as: {COOKIES_FILE}
    5. File should be 300+ KB. If smaller, repeat step 3.
    """)


# ── Step 2: Build segments ────────────────────────────────────────────────────

def build_segments(duration: float, seg_duration: int, workers: int) -> list:
    if duration <= 0:
        print("[2/4] Segments : 1  (duration unknown)")
        return [(0.0, None, 0)]

    raw = []
    start = 0.0
    idx   = 0
    while start < duration:
        end = min(start + seg_duration, duration)
        raw.append((start, end, idx))
        start = end
        idx  += 1

    max_segs = workers * 3
    if len(raw) > max_segs:
        step = duration / max_segs
        raw  = []
        for i in range(max_segs):
            s = round(i * step, 2)
            e = round(min(s + step, duration), 2)
            raw.append((s, e, i))

    avg = int(duration / len(raw))
    print(f"[2/4] Segments : {len(raw)}  (~{avg}s each, {workers} parallel workers)")
    return raw


# ── Step 3: Extract I-frames ──────────────────────────────────────────────────

def extract_segment(
    source: str,
    start: float,
    end,
    seg_idx: int,
    out_dir: Path,
    quality: int,
    max_frames_per_seg: int,
    lowres: int,
) -> list:
    """
    Extract I-frames from one time segment.
    Returns list of (frame_filename, pts_sec_absolute) tuples.
    pts_sec is relative to the START OF THE FULL VIDEO (not the segment).
    """
    seg_out = out_dir / f"seg_{seg_idx:04d}"
    seg_out.mkdir(parents=True, exist_ok=True)
    out_pat = str(seg_out / "frame_%08d.jpg")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-skip_frame",       "nokey",
        "-skip_loop_filter", "all",
        "-skip_idct",        "all",
        "-threads",          "1",
    ]

    if lowres > 0:
        cmd += ["-lowres", str(lowres)]

    if start > 0:
        cmd += ["-ss", str(start)]

    cmd += ["-i", source]

    if end is not None:
        cmd += ["-t", str(round(end - start, 2))]

    if max_frames_per_seg > 0:
        cmd += ["-vframes", str(max_frames_per_seg)]

    # showinfo filter writes pts_time to stderr for exact timestamps
    # Line format: [Parsed_showinfo_1 @ ...] n:  0 pts:512 pts_time:0.512 ...
    cmd += [
        "-an", "-sn", "-dn",
        "-vsync", "vfr",
        "-vf", r"select=eq(pict_type\,I),showinfo",
        "-qscale:v", str(quality),
        out_pat,
    ]

    results = []
    try:
        proc  = subprocess.run(cmd, capture_output=True, text=True)
        saved = sorted(seg_out.glob("frame_*.jpg"))

        # Parse pts_time values from showinfo stderr
        pts_times = re.findall(r"pts_time:([\d.]+)", proc.stderr)

        for i, frame_path in enumerate(saved):
            if i < len(pts_times):
                abs_pts = float(pts_times[i]) + start   # absolute video timestamp
            else:
                abs_pts = start + i * 2.0                # fallback: rough 2s gap
            results.append((frame_path.name, round(abs_pts, 4)))

        end_str = f"{round(end)}s" if end is not None else "end"
        safe_print(f"  [seg {seg_idx:03d}]  {len(saved):4d} frames  "
                   f"| {start:.0f}s -> {end_str}")
    except Exception as e:
        safe_print(f"  [seg {seg_idx:03d}]  ERROR: {e}")

    return results

# ── Step 4: Merge + write timestamps.json ────────────────────────────────────

def merge_segments(out_dir: Path, n_segments: int, seg_results: list) -> int:
    """
    Rename all seg_NNNN/frame_*.jpg into out_dir/frame_NNNNNNNN.jpg.
    Writes timestamps.json using the pts data collected during extraction.
    seg_results: list of lists, one per segment, each [(filename, pts_sec), ...]
    """
    # Build a flat map: original_seg_filename -> pts_sec (keyed by seg_idx)
    # seg_results[seg_idx] = [(frame_name_in_seg, pts_sec), ...]
    pts_map = {}   # seg_dir_name/frame_name -> pts_sec
    for seg_idx, seg_data in enumerate(seg_results):
        for frame_name, pts_sec in seg_data:
            key = f"seg_{seg_idx:04d}/{frame_name}"
            pts_map[key] = pts_sec

    counter    = 1
    timestamps = []   # final list for timestamps.json

    for idx in range(n_segments):
        seg_dir = out_dir / f"seg_{idx:04d}"
        if not seg_dir.exists():
            continue
        for frame in sorted(seg_dir.glob("frame_*.jpg")):
            dest      = out_dir / f"frame_{counter:08d}.jpg"
            final_name = dest.name
            if dest.exists():
                dest.unlink()
            frame.rename(dest)

            # Look up pts for this frame
            lookup_key = f"seg_{idx:04d}/{frame.name}"
            pts_sec    = pts_map.get(lookup_key, -1.0)
            bucket_idx = int(pts_sec // 5) if pts_sec >= 0 else -1

            timestamps.append({
                "frame":   final_name,
                "pts_sec": pts_sec,
                "bucket":  bucket_idx,
            })
            counter += 1

        try:
            seg_dir.rmdir()
        except Exception:
            pass

    # Write timestamps.json next to the frames
    ts_path = out_dir / "timestamps.json"
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=2)
    print(f"[4/4] timestamps.json written  ({len(timestamps)} entries)  ->  {ts_path}")

    return counter - 1

# ── Main ──────────────────────────────────────────────────────────────────────

def run(source, out_dir, workers, seg_duration, max_frames, quality, lowres):
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_count = os.cpu_count() or 4
    t_start   = time.time()

    lowres_labels = ["off (full res)", "1/2 res", "1/4 res", "1/8 res"]
    local = is_local_file(source)

    print(f"\n{'='*60}")
    print(f"  StreamStamper  |  Ultra Fast Parallel I-Frame Extractor")
    print(f"{'='*60}")
    print(f"  Source       : {'[LOCAL] ' if local else '[YouTube] '}{source[:50]}{'...' if len(source)>50 else ''}")
    print(f"  Output       : {out_dir.absolute()}")
    print(f"  Workers      : {workers}  (CPU cores: {cpu_count})")
    print(f"  Seg duration : {seg_duration}s per chunk")
    print(f"  Max frames   : {'unlimited' if not max_frames else max_frames}")
    print(f"  JPEG quality : {quality}  (2=best, 31=worst)")
    print(f"  Lowres mode  : {lowres_labels[lowres]}")
    print(f"{'='*60}\n")

    if local:
        check_deps(need_ytdlp=False)
        video_source, duration, vcodec = resolve_local_file(source)
    else:
        check_deps(need_ytdlp=True)
        cookie_args = get_cookie_args()
        print()
        try:
            video_source, duration, vcodec = resolve_youtube(source, cookie_args)
        except ValueError as e:
            print(f"\n[ERROR] {e}")
            print("  Try: pip install yt-dlp --upgrade")
            sys.exit(1)

    segments   = build_segments(duration, seg_duration, workers)
    n_segments = len(segments)

    max_per_seg = 0
    if max_frames > 0:
        max_per_seg = max(1, max_frames // n_segments)

    print(f"\n[3/4] Launching {min(workers, n_segments)} parallel FFmpeg workers...\n")

    # seg_results[seg_idx] = [(frame_name, pts_sec), ...]  — used for timestamps.json
    seg_results = [[] for _ in range(n_segments)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                extract_segment,
                video_source, start, end, idx,
                out_dir, quality, max_per_seg, lowres
            ): idx
            for start, end, idx in segments
        }
        for f in as_completed(futures):
            seg_idx = futures[f]
            seg_results[seg_idx] = f.result()

    print(f"\n[4/4] Merging segments -> {out_dir.name}/...")
    final_count = merge_segments(out_dir, n_segments, seg_results)

    elapsed  = time.time() - t_start
    fps_rate = final_count / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Source mode    : {'local file' if local else 'YouTube stream'}")
    print(f"  Stream codec   : {vcodec}")
    print(f"  Frames saved   : {final_count}")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print(f"  Throughput     : {fps_rate:.2f} frames/sec")
    print(f"  Output dir     : {out_dir.absolute()}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    cpu_count       = os.cpu_count() or 4
    default_workers = min(cpu_count, 8)

    ap = argparse.ArgumentParser(
        prog="streamstamper",
        description="Ultra-fast parallel I-frame extractor — YouTube URLs and local files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python streamstamper.py "https://youtube.com/watch?v=VIDEOID"
  python streamstamper.py "C:\\Videos\\gameplay.mp4"
  python streamstamper.py "C:\\Videos\\lecture.mkv" --workers 16 --seg-duration 300
  python streamstamper.py "URL_or_PATH" --max 100 --out frames
  python streamstamper.py "URL_or_PATH" --lowres 1 --workers 16

Supported local formats: mp4, mkv, mov, avi, webm, flv, wmv, m4v, ts, mpg, and more.

CPU cores on this machine: {cpu_count}  ->  recommended --workers: {default_workers}
        """
    )
    ap.add_argument("source",
                    help="YouTube URL or local video file path")
    ap.add_argument("--out",          type=str, default="imges",         metavar="DIR",
                    help="Output folder (default: ./imges)")
    ap.add_argument("--workers",      type=int, default=default_workers, metavar="N",
                    help=f"Parallel ffmpeg processes (default: {default_workers})")
    ap.add_argument("--seg-duration", type=int, default=300,             metavar="SEC",
                    help="Seconds per segment (default: 300)")
    ap.add_argument("--max",          type=int, default=0,               metavar="N",
                    help="Max I-frames to save, 0=unlimited (default: 0)")
    ap.add_argument("--quality",      type=int, default=2,               metavar="Q",
                    help="JPEG quality 2=best 31=worst (default: 2)")
    ap.add_argument("--lowres",       type=int, default=0,               metavar="N",
                    choices=[0, 1, 2, 3],
                    help="Decode res: 0=full 1=half 2=quarter 3=eighth (default: 0)")
    args = ap.parse_args()

    run(
        source=args.source,
        out_dir=Path(args.out),
        workers=args.workers,
        seg_duration=args.seg_duration,
        max_frames=args.max,
        quality=args.quality,
        lowres=args.lowres,
    )


if __name__ == "__main__":
    main()