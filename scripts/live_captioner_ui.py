# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "rich>=12.6.0",
#     "numpy",
#     "pocket-tts",
#     "gradio",
#     "matplotlib",
#     "yt-dlp",
#     "einops>=0.8.1",
#     "accelerate",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchcodec==0.4.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Live Captioning UI

Watch the live captioner actually run live: the video plays at native speed while the
model captions frames as fast as it can, streaming tokens over a single KV cache.

Usage:
    uv run scripts/live_captioner_ui.py [--model_id CASA-Qwen2_5-VL-3B-LiveCC] [--port 7860]
    uv run scripts/live_captioner_ui.py --help
"""

import base64
import dataclasses
import hashlib
import html
import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

# GRADIO_TEMP_DIR is read at import time so it must be set before importing gradio
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path.home() / ".cache" / "casa_gradio"))
Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from fire import Fire  # noqa: E402
from PIL import Image  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.generation.logits_process import LogitsProcessor  # noqa: E402

# So that `live_captioner` imports below resolve when the script is run from anywhere
_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from live_captioner import (  # noqa: E402
    QWEN_SYSTEM_PROMPT,
    build_logits_processors,
    load_model,
    resolve_stop_tokens,
    tokenize_frame_messages,
)

ACCENT = "#AEF239"
MAX_DISPLAY_FPS = 24


# ---------------------------------------------------------------------------
# Captioning engine (no UI dependency)
# ---------------------------------------------------------------------------


@dataclass
class GenerationParams:
    """Sampling parameters for streaming caption generation"""

    fps: int = 2
    max_new_tokens: int = 20
    repetition_penalty: float = 1.15
    prompt: str = "In this video"
    temp: float = 0.4
    top_k: int = 256
    eos_bias: float = 0.95
    ngram_size: int = 4
    ngram_window: int = 60


@dataclass
class Session:
    """Caption state shared between the worker thread and the display loop

    Also owns the playback clock: :meth:`elapsed` is the time since the run started with
    paused time removed, so the video, the captions and the per-frame compute budget are all
    measured against the same clock and stop and resume together.
    """

    pause: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _started_at: float = field(default_factory=time.perf_counter)
    _paused_at: float = 0.0
    _paused_for: float = 0.0
    _live_text: str = ""
    _live_ts: float = 0.0
    _history: list[tuple[float, str]] = field(default_factory=list)  # (video_ts, caption)
    # (video_ts, lag_s); lag is negative when the captioner ran ahead of playback
    _delays: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pause.set()  # start unpaused

    def elapsed(self) -> float:
        """Playback time in seconds, excluding time spent paused"""
        with self._lock:
            now = time.perf_counter()
            paused_now = 0.0 if self.pause.is_set() else now - self._paused_at
            return now - self._started_at - self._paused_for - paused_now

    def is_paused(self) -> bool:
        """Whether playback is paused (the `pause` event is set while playing)"""
        return not self.pause.is_set()

    def set_paused(self, paused: bool) -> bool:
        """Pause or resume playback, idempotently; returns True if now paused

        The clock accounting lives here rather than in the waiters because both threads
        block on the same event, so charging the pause on wake-up would count it twice.
        """
        with self._lock:
            if paused == self.is_paused():
                return paused
            if paused:
                self._paused_at = time.perf_counter()
                self.pause.clear()
            else:
                self._paused_for += time.perf_counter() - self._paused_at
                self.pause.set()
            return paused

    def toggle_pause(self) -> bool:
        """Pause if playing, resume if paused; returns True if now paused"""
        return self.set_paused(not self.is_paused())

    def publish_live(self, ts: float, text: str) -> None:
        with self._lock:
            self._live_ts, self._live_text = ts, text

    def finish_frame(self, ts: float, text: str, lag: float) -> None:
        with self._lock:
            self._history.append((ts, text))
            self._delays.append((ts, lag))
            self._live_text = ""

    def snapshot(self) -> tuple[str, float, list[tuple[float, str]], list[tuple[float, float]]]:
        with self._lock:
            return self._live_text, self._live_ts, list(self._history), list(self._delays)


class CaptionEngine:
    """Streaming captioner: one image prefill per frame, one KV cache across the video"""

    def __init__(self, model_id: str, image_size: int = 448):
        self.model, self.processor = load_model(model_id, image_size=image_size)
        self.is_qwen = "qwen" in model_id.lower()
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.stop_tokens = resolve_stop_tokens(self.model.generation_config.eos_token_id)
        self._stop_token_set = set(self.stop_tokens)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        return self.processor._image_processor.process_images(frames).to("cuda")

    def _positions(self, start: int, length: int) -> torch.Tensor:
        pos = torch.arange(start, start + length, dtype=torch.long, device="cuda")
        if self.is_qwen:
            return pos.view(1, 1, length).expand(3, 1, length).contiguous()
        return pos.unsqueeze(0)

    def _decode(self, token_ids: torch.Tensor, kv_cache: DynamicCache) -> Any:
        """Forward text-only tokens, extending the KV cache"""
        with torch.no_grad():
            return self.model.forward(
                token_ids,
                pixel_values=None,
                past_key_values=kv_cache,
                use_cache=True,
                attention_mask=torch.ones((1, token_ids.shape[1]), device="cuda"),
                position_ids=self._positions(kv_cache._seen_tokens, token_ids.shape[1]),
                __is_first_gen_call__=False,
            )

    def _sample(
        self,
        logits: torch.Tensor,
        context_ids: torch.Tensor,
        processors: list[LogitsProcessor],
        params: GenerationParams,
    ) -> torch.Tensor:
        """Pick the next token id, as a (1, 1) tensor"""
        for proc in processors:
            logits = proc(context_ids, logits)
        if params.temp <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / params.temp
        if params.top_k > 0:
            kth = torch.topk(logits, min(params.top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    def stream(self, frames: torch.Tensor, params: GenerationParams, session: Session) -> None:
        """Caption pre-processed frames sequentially, publishing tokens into the session

        Runs in a worker thread. Generation for a frame is cut short as soon as its
        real-time budget (1/fps) is spent, so captions keep up with playback.
        """
        model, processor = self.model, self.processor
        system_prompt = QWEN_SYSTEM_PROMPT if self.is_qwen else None
        logits_processors = build_logits_processors(
            stop_tokens=self.stop_tokens,
            repetition_penalty=params.repetition_penalty,
            eos_bias=params.eos_bias,
            ngram_size=params.ngram_size,
            ngram_window=params.ngram_window,
        )
        # Generation is one long assistant turn, so we never emit turn-end tokens
        processor.asst_end_tokens = []
        frame_interval = 1.0 / params.fps
        model.reset_ca_streaming_states()
        kv_cache = DynamicCache()
        stop_token = self.stop_tokens[0]

        for frame_idx in range(len(frames)):
            session.pause.wait()
            if session.stop.is_set():
                return
            model_ts = frame_idx / params.fps

            inputs = tokenize_frame_messages(
                processor, frame_idx, params.prompt, system_prompt, self.is_qwen
            )
            input_ids = inputs["input_ids"].cuda()
            insertion_points = inputs.get("image_embeds_insertion_points")
            if insertion_points is not None:
                insertion_points = [p.cuda() for p in insertion_points]
            past_len = kv_cache._seen_tokens

            # Prefill: process the frame + all input tokens in one forward pass
            model.start_ca_streaming_states()
            with torch.no_grad():
                outputs = model.forward(
                    input_ids,
                    pixel_values=list(frames[frame_idx : frame_idx + 1].to(torch.bfloat16)),
                    past_key_values=kv_cache,
                    use_cache=True,
                    attention_mask=torch.ones((1, past_len + input_ids.shape[1]), device="cuda"),
                    position_ids=self._positions(past_len, input_ids.shape[1]),
                    image_embeds_insertion_points=insertion_points,
                    pre_image_tokens=list(model.config.pre_image_tokens),
                    post_image_tokens=list(model.config.post_image_tokens),
                )

            current_text = (params.prompt + " ") if (frame_idx == 0 and params.prompt) else ""
            current_ids = input_ids
            deadline = session.elapsed() + frame_interval

            for _ in range(params.max_new_tokens):
                next_id = self._sample(
                    outputs.logits[:, -1, :], current_ids, logits_processors, params
                )
                token = int(next_id.item())
                if token in self._stop_token_set:
                    # Remember which stop token the model actually minted: flushing a
                    # different one into the cache below makes the context look like
                    # concatenated documents, and the per-frame cadence degrades.
                    stop_token = token
                    break

                current_text += processor.tokenizer.decode([token], skip_special_tokens=True)
                session.publish_live(model_ts, current_text)
                # Forwarded before the budget check, so the KV cache always holds every
                # token the caption shows
                current_ids = torch.cat([current_ids, next_id], dim=-1)
                outputs = self._decode(next_id, kv_cache)

                session.pause.wait()
                if session.stop.is_set():
                    return
                if session.elapsed() >= deadline:
                    break

            # Negative lag means the captioner finished the frame before playback reached
            # it, which is what the pace plot shows as running under the real-time line
            session.finish_frame(model_ts, current_text.strip(), session.elapsed() - model_ts)

            # Close the frame's generation in the KV cache
            self._decode(torch.tensor([[stop_token]], device="cuda"), kv_cache)
            model.reset_ca_streaming_states()


# The model is fixed for the process lifetime and loaded once in main(), before the
# UI is served, so a run never has to wait on a cold model load.
_ENGINE: CaptionEngine | None = None
_SESSION: Session | None = None  # single-user demo: session of the current run


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


def _resolve_video(path_or_url: str) -> str:
    """Download from URL with yt-dlp if needed; return a local path"""
    path_or_url = path_or_url.strip()
    if not path_or_url.startswith(("http://", "https://")):
        return path_or_url
    dl_dir = Path(os.environ["GRADIO_TEMP_DIR"]) / "downloads"
    dl_dir.mkdir(exist_ok=True)
    out = dl_dir / f"{hashlib.md5(path_or_url.encode()).hexdigest()[:12]}.mp4"
    if not out.exists():
        subprocess.run(["yt-dlp", "-o", str(out), path_or_url], check=True)
    return str(out)


@dataclass
class LoadedVideo:
    """An open video, decoded frame by frame as it is displayed

    Playback frames are read on demand rather than decoded up front: a few minutes of
    1080p is tens of GB as raw tensors, and the player only ever shows one frame at a time.
    """

    duration: float
    display_fps: int
    decoder: Any
    start_time: float

    def frame_at(self, ts: float) -> torch.Tensor:
        """The frame shown at playback time `ts` seconds, as a (C, H, W) uint8 tensor"""
        ts = min(max(ts, 0.0), max(self.duration - 1e-3, 0.0))
        return self.decoder.get_frame_played_at(self.start_time + ts).data

    def decode_at(self, fps: int) -> torch.Tensor:
        """Every frame at `fps` as one (N, C, H, W) uint8 batch, for model preprocessing"""
        times = [self.start_time + i / fps for i in range(int(fps * self.duration))]
        return self.decoder.get_frames_played_at(times).data


def _load_video(path_or_url: str) -> LoadedVideo:
    from torchcodec.decoders._video_decoder import VideoDecoder

    decoder = VideoDecoder(_resolve_video(path_or_url), device="cpu")
    start_time = decoder.metadata.begin_stream_seconds
    duration = decoder.metadata.end_stream_seconds - start_time
    display_fps = int(min(decoder.metadata.average_fps or MAX_DISPLAY_FPS, MAX_DISPLAY_FPS))
    return LoadedVideo(duration, display_fps, decoder, start_time)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SUBTITLE_WINDOW = 14  # total words shown in the live subtitle
_SUBTITLE_BRIGHT = 5  # how many of the newest words are highlighted
# The player is a fraction of a browser window, and every frame is re-sent as a data URI,
# so full-resolution JPEGs only cost bandwidth
_MAX_PLAYER_WIDTH = 960


def _player_html(frame: torch.Tensor | None, words: list[str], status: str = "") -> str:
    """Render one video frame with a subtitle overlay and a small status line"""
    if frame is None:
        img = ""
    else:
        image = Image.fromarray(frame.permute(1, 2, 0).numpy())
        if image.width > _MAX_PLAYER_WIDTH:
            height = round(image.height * _MAX_PLAYER_WIDTH / image.width)
            image = image.resize((_MAX_PLAYER_WIDTH, height), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        img = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block">'

    visible = words[-_SUBTITLE_WINDOW:]
    spans = []
    for i, w in enumerate(visible):
        age = len(visible) - 1 - i  # 0 = newest word
        color = ACCENT if age < _SUBTITLE_BRIGHT else "#999"
        spans.append(f'<span style="color:{color}">{html.escape(w)}</span>')
    subtitle = " ".join(spans) or "&nbsp;"

    return (
        '<div style="position:relative;border-radius:8px;overflow:hidden;'
        'background:#111;min-height:220px">'
        f"{img}"
        '<div style="position:absolute;left:0;right:0;bottom:0;padding:10px 16px;'
        "background:linear-gradient(transparent,rgba(0,0,0,.85) 40%);text-align:center;"
        'font-size:1.25em;font-weight:bold;text-shadow:0 1px 3px #000;color:#eee">'
        f"{subtitle}</div>"
        '<div style="position:absolute;top:8px;right:10px;font-size:.75em;color:#ccc;'
        f'background:rgba(0,0,0,.55);padding:2px 8px;border-radius:4px">{status}</div>'
        "</div>"
    )


def _message_html(msg: str) -> str:
    return (
        '<div style="display:flex;align-items:center;justify-content:center;min-height:220px;'
        'background:#111;border-radius:8px;font-size:1.1em;color:#aaa">'
        f"{html.escape(msg)}</div>"
    )


def _words_up_to(history: list[tuple[float, str]], video_ts: float) -> list[str]:
    """The caption words belonging to the video up to `video_ts`

    The captioner usually runs ahead of playback (it only spends its full per-frame budget
    on frames it has a lot to say about), and showing a caption the moment it is produced
    would put a subtitle for a frame several seconds in the future over the current one.
    Each caption is held back until playback reaches the frame it describes.
    """
    return [word for ts, text in history if ts <= video_ts for word in text.split()]


def _transcript_text(history: list[tuple[float, str]]) -> str:
    return "\n".join(f"[{ts:6.1f}s] {text}" for ts, text in history if text)


def _style_axes(fig: plt.Figure, ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel(xlabel, fontsize=7, color="#aaa")
    ax.set_ylabel(ylabel, fontsize=7, color="#aaa")
    ax.tick_params(colors="#aaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout(pad=0.4)


def _pace_plot(delays: list[tuple[float, float]]) -> plt.Figure:
    """Caption completion time vs video time, against the y=x real-time reference

    A curve under the diagonal means the captioner finished a frame before playback got
    there; on the diagonal is exactly real time, above it is falling behind.
    """
    fig, ax = plt.subplots(figsize=(4, 1.8))
    if delays:
        xs = [ts for ts, _ in delays]
        ys = [ts + lag for ts, lag in delays]  # wall completion time
        limit = max(max(xs), max(ys), 1e-3) * 1.05
        ax.plot([0, limit], [0, limit], color="#888", linewidth=1, linestyle="--", zorder=1)
        ax.text(limit * 0.98, limit * 0.9, "real-time", color="#888", fontsize=6, ha="right")
        ax.fill_between(xs, xs, ys, alpha=0.2, color=ACCENT, zorder=2)
        ax.plot(xs, ys, color=ACCENT, linewidth=1.5, marker="o", markersize=3, zorder=3)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
    else:
        ax.text(0.5, 0.5, "waiting…", ha="center", va="center", color="#666", fontsize=8)
    _style_axes(fig, ax, "video time (s)", "completion time (s)")
    return fig


def _memory_plot(mem: list[tuple[float, float]]) -> plt.Figure:
    """GPU memory over video time, scaled to the variation rather than to zero"""
    fig, ax = plt.subplots(figsize=(4, 1.8))
    if mem:
        xs, ys = zip(*mem)
        # A CA model's footprint barely moves, so an axis anchored at 0 shows a flat line;
        # zoom to the data and keep a floor so a constant trace still gets a sane window
        margin = max(max(ys) - min(ys), 0.05) * 0.15
        low, high = min(ys) - margin, max(ys) + margin
        ax.plot(xs, ys, color="#4fc3f7", linewidth=1.5, marker="o", markersize=3)
        ax.fill_between(xs, low, ys, alpha=0.15, color="#4fc3f7")
        ax.set_ylim(low, high)
    else:
        ax.text(0.5, 0.5, "waiting…", ha="center", va="center", color="#666", fontsize=8)
    _style_axes(fig, ax, "video time (s)", "GPU mem (GiB)")
    return fig


# ---------------------------------------------------------------------------
# Run loop (Gradio generator)
# ---------------------------------------------------------------------------


def run_captioning(video_path: str, params: GenerationParams) -> Generator:
    """Caption a video while displaying it at native speed

    Yields (player_html, transcript, pace_fig, mem_fig, scrub_update, timeline_state).
    The caption worker runs in a background thread, decoupled from the display pacing.
    """
    global _SESSION
    assert _ENGINE is not None, "main() must load the engine before the UI is built"

    def _updates(player: str, **kwargs: Any) -> tuple:
        noop = gr.update()
        return (
            player,
            kwargs.get("transcript", noop),
            kwargs.get("pace", noop),
            kwargs.get("mem", noop),
            kwargs.get("scrub", noop),
            kwargs.get("timeline", noop),
        )

    if not video_path or not video_path.strip():
        yield _updates(_message_html("Please provide a video path or URL."))
        return

    yield _updates(_message_html("Loading video…"))
    try:
        video = _load_video(video_path)
    except Exception as e:  # surface decode/download errors in the player
        yield _updates(_message_html(f"Failed to load video: {e}"))
        return

    yield _updates(
        _message_html("Preprocessing frames…"),
        scrub=gr.update(maximum=round(video.duration, 1), step=round(1.0 / params.fps, 3)),
    )
    model_frames = _ENGINE.preprocess(video.decode_at(params.fps))

    session = Session()
    _SESSION = session
    worker = threading.Thread(
        target=_ENGINE.stream, args=(model_frames, params, session), daemon=True
    )
    worker.start()

    mem_history: list[tuple[float, float]] = []
    pace_fig, mem_fig = _pace_plot([]), _memory_plot([])
    frame_period = 1.0 / video.display_fps
    shown_frames, shown_mem_sec = -1, -1  # -1: send the empty plots on the first frame

    try:
        while True:
            session.pause.wait()
            # The frame to show is whichever one the playback clock is on, so a slow
            # display step drops frames instead of drifting behind the captions
            video_ts = session.elapsed()
            if video_ts >= video.duration:
                return
            live_text, live_ts, history, delays = session.snapshot()

            # Everything but the player changes once per caption or once per second, so it
            # is not re-rendered on every displayed frame
            changed: dict[str, Any] = {}
            if len(history) != shown_frames:
                shown_frames = len(history)
                plt.close(pace_fig)
                pace_fig = _pace_plot(delays)
                changed |= {
                    "transcript": _transcript_text(history),
                    "pace": pace_fig,
                    "timeline": {"video": video, "history": history},
                }
            if int(video_ts) != shown_mem_sec:
                shown_mem_sec = int(video_ts)
                mem_history.append((video_ts, torch.cuda.memory_allocated() / 1024**3))
                plt.close(mem_fig)
                mem_fig = _memory_plot(mem_history)
                changed["mem"] = mem_fig

            # The in-progress caption is only shown once playback has reached its frame,
            # which is the case exactly when the captioner is behind
            words = _words_up_to(history, video_ts)
            if live_ts <= video_ts:
                words += live_text.split()
            caption_ts = max(live_ts, history[-1][0] if history else 0.0)
            lead = caption_ts - video_ts
            status = f"video {video_ts:5.1f}s · captioner {abs(lead):4.1f}s " + (
                "ahead" if lead >= 0 else "behind"
            )
            # The scrub bar is deliberately not driven from here: writing a new value 24
            # times a second snaps the handle back out from under a drag, which made it
            # look inert. The playhead lives in the status line instead.
            yield _updates(_player_html(video.frame_at(video_ts), words, status), **changed)
            time.sleep(max(0.0, frame_period - (session.elapsed() - video_ts)))
    finally:
        session.stop.set()
        session.pause.set()  # unblock the worker if it was paused


def _pause_label(paused: bool) -> Any:
    return gr.update(value="▶ Resume" if paused else "⏸ Pause")


def _toggle_pause() -> Any:
    if _SESSION is None:
        return gr.update()
    return _pause_label(_SESSION.toggle_pause())


def _on_scrub(ts: float, timeline: dict | None) -> tuple[Any, Any]:
    """Show a past frame with the captions that had been produced by then

    Reviewing pauses playback: the display loop writes the same player on every frame, so
    a reviewed frame would otherwise be overwritten within milliseconds.
    """
    if timeline is None:
        return gr.update(), gr.update()
    video: LoadedVideo = timeline["video"]
    paused = _SESSION.set_paused(True) if _SESSION is not None else True
    return (
        _player_html(
            video.frame_at(ts), _words_up_to(timeline["history"], ts), f"review {ts:5.1f}s"
        ),
        _pause_label(paused),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _build_ui(defaults: GenerationParams, model_id: str) -> gr.Blocks:
    with gr.Blocks(title="Live Video Captioner", analytics_enabled=False) as demo:
        gr.Markdown(f"## Live Video Captioner — `{model_id}`")
        timeline_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Textbox(
                    label="Video path / URL",
                    placeholder="/path/to/video.mp4  or  https://youtu.be/…",
                )
                with gr.Row():
                    run_btn = gr.Button("▶ Start", variant="primary")
                    pause_btn = gr.Button("⏸ Pause")
                    stop_btn = gr.Button("⏹ Stop")
                with gr.Accordion("Generation settings", open=True):
                    with gr.Row():
                        fps_sl = gr.Slider(1, 8, value=defaults.fps, step=1, label="FPS")
                        tok_sl = gr.Slider(
                            5, 100, value=defaults.max_new_tokens, step=5, label="Max tokens/frame"
                        )
                    with gr.Row():
                        temp_sl = gr.Slider(
                            0.0, 1.0, value=defaults.temp, step=0.05, label="Temperature"
                        )
                        topk_sl = gr.Slider(1, 512, value=defaults.top_k, step=1, label="Top-K")
                    with gr.Row():
                        rp_sl = gr.Slider(
                            1.0,
                            2.0,
                            value=defaults.repetition_penalty,
                            step=0.05,
                            label="Rep. penalty",
                        )
                        eos_sl = gr.Slider(
                            0.1, 2.0, value=defaults.eos_bias, step=0.05, label="EOS bias"
                        )
                    with gr.Row():
                        ngram_sl = gr.Slider(
                            2, 8, value=defaults.ngram_size, step=1, label="N-gram size"
                        )
                        ngramw_sl = gr.Slider(
                            20, 200, value=defaults.ngram_window, step=10, label="N-gram window"
                        )
                    prompt_in = gr.Textbox(label="Initial prompt", value=defaults.prompt)

            with gr.Column(scale=2):
                player_out = gr.HTML(_message_html("Enter a video and press Start."))
                scrub_sl = gr.Slider(
                    0,
                    600,
                    step=0.1,
                    value=0,
                    label="Review (s) — drag to look back at a frame; pauses playback",
                )
                with gr.Row():
                    pace_out = gr.Plot(label="Captioning pace")
                    mem_out = gr.Plot(label="GPU memory")
                with gr.Accordion("Transcript", open=False):
                    transcript_out = gr.Textbox(
                        show_label=False, container=False, lines=6, max_lines=14, interactive=False
                    )

        def _run(
            video: str,
            fps: float,
            toks: float,
            temp: float,
            top_k: float,
            rp: float,
            eos: float,
            ngram: float,
            ngramw: float,
            prompt: str,
        ) -> Generator:
            params = GenerationParams(
                fps=int(fps),
                max_new_tokens=int(toks),
                repetition_penalty=rp,
                prompt=prompt,
                temp=temp,
                top_k=int(top_k),
                eos_bias=eos,
                ngram_size=int(ngram),
                ngram_window=int(ngramw),
            )
            yield from run_captioning(video, params)

        run_event = run_btn.click(
            fn=_run,
            inputs=[
                video_in,
                fps_sl,
                tok_sl,
                temp_sl,
                topk_sl,
                rp_sl,
                eos_sl,
                ngram_sl,
                ngramw_sl,
                prompt_in,
            ],
            outputs=[player_out, transcript_out, pace_out, mem_out, scrub_sl, timeline_state],
        )
        pause_btn.click(fn=_toggle_pause, outputs=[pause_btn])
        stop_btn.click(fn=None, cancels=[run_event])
        # .release only fires on user interaction, not on programmatic slider updates
        scrub_sl.release(
            fn=_on_scrub, inputs=[scrub_sl, timeline_state], outputs=[player_out, pause_btn]
        )

    return demo


def main(
    model_id: str = "CASA-Qwen2_5-VL-3B-LiveCC",
    port: int = 10060,
    image_size: int = 448,
    **gen_params: Any,
):
    """Launch the live captioning Gradio UI

    The model is loaded once, up front, so the UI never blocks on a cold load.

    :param model_id: released model name, hub repo id, or local checkpoint directory
    :param port: Gradio server port
    :param image_size: frame resolution fed to the vision tower. Defaults to the 448 the
        livecc checkpoints were fine-tuned on; 896 costs 4x the vision compute and 4x the
        cross-attention image KV, which this real-time loop cannot afford
    :param gen_params: Default generation parameters (see GenerationParams: fps,
        max_new_tokens, repetition_penalty, prompt, temp, top_k, eos_bias,
        ngram_size, ngram_window)
    """
    unknown = set(gen_params) - {f.name for f in dataclasses.fields(GenerationParams)}
    if unknown:
        raise ValueError(f"Unknown generation parameters: {sorted(unknown)}")

    global _ENGINE
    print(f"Loading {model_id}…")
    _ENGINE = CaptionEngine(model_id, image_size=image_size)
    print("Model loaded.")

    demo = _build_ui(GenerationParams(**gen_params), model_id)
    demo.launch(server_port=port, server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    Fire(main)
