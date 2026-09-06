# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "rich>=12.6.0",
#     "numpy",
#     "pocket-tts",
#     "einops>=0.8.1",
#     "accelerate",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchcodec==0.4.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Live Captioning Inference

uv run scripts/live_captioner.py --help
"""

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from typing import cast as type_cast

# Triton compiles kernels on first use (torch.compile'd vision tower). Its cache is kept on
# local disk: on a networked home directory the compilation lock is slow and can hang.
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_dir_{os.environ.get('USER', '')}")

import rich
import torch
from fire import Fire
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessor

RELEASED_MODELS = (
    "CASA-Helium1-VL-2B",
    "CASA-Helium1-VL-2B-Shared",
    "Helium1-VL-2B",
    "CASA-Qwen2_5-VL-3B",
    "CASA-Qwen2_5-VL-3B-Shared",
    "CASA-Qwen2_5-VL-3B-Frozen",
    "CASA-Qwen2_5-VL-3B-LiveCC",
)


def resolve_repo(model_id: str) -> str:
    """Expand a released model name into its Hugging Face repo id.

    Full repo ids ("org/name") and local checkpoint directories are returned unchanged.

    :param model_id: released model name, hub repo id, or local directory
    """
    if "/" in model_id or Path(model_id).exists():
        return model_id
    if model_id not in RELEASED_MODELS:
        raise ValueError(f"Unknown model {model_id!r}, expected one of {RELEASED_MODELS}")
    return f"kyutai/{model_id}"


class TimingHook:
    """Hook to measure token generation time.

    Uses CUDA events instead of a per-forward ``torch.cuda.synchronize()``: a host sync after
    every decode step serializes the pipeline and taxes exactly the loop we are timing. Events
    are enqueued on the stream at zero host cost; the single sync happens when ``timings`` is
    read at the end of the frame.
    """

    def __init__(self):
        self._events: list[torch.cuda.Event] = []
        self.mems = []

    def reset(self):
        self._events = [torch.cuda.Event(enable_timing=True)]
        self._events[0].record()
        self.mems = []

    def __call__(self, module: torch.nn.Module, input: Any, output: Any):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.append(event)
        self.mems.append(torch.cuda.memory.max_memory_allocated() / (1024**3))
        torch.cuda.reset_peak_memory_stats()

    @property
    def timings(self) -> list[float]:
        """Per-forward wall times in seconds; syncs once on the last recorded event."""
        if len(self._events) < 2:
            return []
        self._events[-1].synchronize()
        return [
            self._events[i].elapsed_time(self._events[i + 1]) / 1000.0
            for i in range(len(self._events) - 1)
        ]


def __convert_to_progressive_subtitles__(
    input_file: str | Path, output_file: str | Path, max_line_length: int = 50
) -> None:
    """
    Convert SRT to progressive chunk by chunk

    :param input_file: Input .srt file path
    :param output_file: Output .srt file path
    :param max_line_length: Maximum character length before wrapping (default 50)
    """

    def parse_srt_time(time_str: str) -> datetime:
        """Parse SRT timestamp to datetime object"""
        return datetime.strptime(time_str.strip(), "%H:%M:%S,%f")

    def format_srt_time(dt: datetime) -> str:
        """Format datetime to SRT timestamp"""
        return dt.strftime("%H:%M:%S,%f")[:-3]

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into subtitle blocks
    blocks = re.split(r"\n\n+", content.strip())

    new_subtitles = []
    subtitle_counter = 1
    previous_sentence = ""
    current_sentence = ""

    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")

        # Check if this line contains timestamps
        if "-->" not in lines[1]:
            continue

        # Extract timestamps
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1]
        )

        if not time_match:
            continue

        # align end to next non-empty block
        next_start_time = None
        if i < len(blocks) - 1:
            next_block = blocks[i + 1]
            next_lines = next_block.strip().split("\n")

            if len(" ".join(next_lines[2:])) == 0 and i < len(blocks) - 2:
                next_block = blocks[i + 2]
                next_lines = next_block.strip().split("\n")

            if "-->" not in next_lines[1]:
                continue

            # Extract timestamps
            next_time_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", next_lines[1]
            )
            if next_time_match is None:
                continue
            next_start_str, _ = next_time_match.groups()
            next_start_time = parse_srt_time(next_start_str)

        start_str, end_str = time_match.groups()
        start_time = parse_srt_time(start_str)
        if next_start_time is not None:
            end_time = next_start_time
        else:
            end_time = parse_srt_time(end_str)
        if start_time >= end_time:
            continue
        # assert start_time < end_time, f"{start_time} >= {end_time}"

        # Get subtitle text (everything after line 1)
        subtitle_text = " ".join(lines[2:])

        # Split into words
        if len(subtitle_text) == 0:
            continue

        if len(subtitle_text) + len(current_sentence) > max_line_length:
            previous_sentence = current_sentence
            current_sentence = ""

        current_sentence += " " + subtitle_text.replace("\n", "")

        # Build subtitle with previous sentence context
        full_text = current_sentence + " " * (max_line_length - len(current_sentence))
        if previous_sentence:
            full_text = f"{previous_sentence}\n{full_text}"
        splt = full_text.split("\n")
        if len(splt) > 2:
            splt = splt[-2:]
        full_text = "\n".join(splt)

        # Create subtitle entry
        new_subtitles.append(
            f"{subtitle_counter}\n{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n{full_text}"
        )
        subtitle_counter += 1

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(new_subtitles) + "\n")

    print(f"Progressive subtitles saved to {output_file}")
    print(f"Created {subtitle_counter - 1} subtitle entries")


class EMARepetitionPenalty(LogitsProcessor):
    """
    Repetition penalty using an EMA (exponential moving average) over sampled tokens.

    Usage:
        processor = EMARepetitionPenalty(penalty=1.2, decay=0.99)
        ...
        while generating:
            logits = processor(input_ids, logits)
            next_token = sample(logits)
    """

    def __init__(
        self,
        penalty: float,
        decay: float = 0.99,
        ignore_tokens: list[int] | None = None,
        max_clamp: float = 5.0,
    ):
        super().__init__()
        if not (0 < decay < 1):
            raise ValueError("decay must be in (0,1).")

        self.penalty = float(penalty)
        self.decay = float(decay)
        self.ignore_tokens = [] if ignore_tokens is None else ignore_tokens
        self.max_clamp = max_clamp
        # All state is allocated lazily from the first `scores`, on its device: its width is the
        # only trustworthy vocabulary size (a checkpoint whose lm_head was extended is wider than
        # its tokenizer), and the penalty runs once per decode step so it must not trigger any
        # host<->device sync (no .item(), no int indexing from python ids).
        self.ema_counts: torch.Tensor | None = None
        self._ignore_idx: torch.Tensor | None = None
        self._one: torch.Tensor | None = None

    def reset(self):
        """Clear EMA state between independent generations."""
        self.ema_counts = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):  # type: ignore[override]
        """Decay the EMA counts, charge the last sampled token, and penalize logits"""
        if self.penalty == 1.0:
            return scores

        if (
            self.ema_counts is None
            or self.ema_counts.shape[0] != scores.shape[-1]
            or self.ema_counts.device != scores.device
        ):
            self.ema_counts = torch.zeros(scores.shape[-1], device=scores.device)
            self._one = torch.ones(1, device=scores.device)
            self._ignore_idx = torch.tensor(
                self.ignore_tokens, dtype=torch.long, device=scores.device
            )

        # Update, entirely device-side: `int(...item())` here would sync the host every step
        assert input_ids.shape[0] == 1, "EMARepetitionPeanlty expects batch size 1"
        self.ema_counts.mul_(self.decay)
        self.ema_counts.index_add_(0, input_ids[0, -1:], self._one)

        # Compute penalties
        ema = self.ema_counts.clamp(min=0.0, max=self.max_clamp)
        penalty_factors = self.penalty**ema
        if self._ignore_idx is not None and self._ignore_idx.numel():
            penalty_factors[self._ignore_idx] = 1.0

        # Apply penalties
        pf = penalty_factors[None, :]
        scores = torch.where(scores < 0, scores * pf, scores / pf)
        return scores


class WindowedNoRepeatNGram(LogitsProcessor):
    """Block n-gram repetition within a recent token window, ACROSS generate calls.

    Streaming generation calls `generate` once per frame with only that frame's message as
    input_ids; the past lives in the KV cache, which logits processors never see. A stateless
    blocker therefore only guards within a single frame (a couple of tokens) and long-period
    loops sail through. This keeps its own rolling history of every token it has seen so the
    window genuinely spans frames.

    :param ngram_size: Size of n-grams to block
    :param window: Number of recent tokens to scan (O(window) per step, not O(sequence))
    :param ignore_tokens: ids never banned. Stop/frame-boundary tokens must be exempt: silent
        frames produce legitimate runs of the boundary token, and banning it after a few of
        them would force the model to speak on every frame.
    """

    def __init__(self, ngram_size: int, window: int = 60, ignore_tokens: list[int] | None = None):
        self.ngram_size = ngram_size
        self.window = window
        self.ignore_tokens = set(ignore_tokens or [])
        self.history: list[int] = []
        self._last_ids: list[int] = []

    def reset(self):
        """Clear history between independent streams."""
        self.history = []
        self._last_ids = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ids = input_ids[0].tolist()
        # Same generate call, one step later → append the newly sampled delta; anything else is a
        # fresh per-frame call → append its whole input.
        if len(ids) > len(self._last_ids) and ids[: len(self._last_ids)] == self._last_ids:
            self.history.extend(ids[len(self._last_ids) :])
        else:
            self.history.extend(ids)
        self._last_ids = ids
        self.history = self.history[-self.window :]

        ids = self.history
        prefix = tuple(ids[-(self.ngram_size - 1) :])
        banned = set()
        for i in range(len(ids) - self.ngram_size + 1):
            if tuple(ids[i : i + self.ngram_size - 1]) == prefix:
                banned.add(ids[i + self.ngram_size - 1])
        for tok in banned - self.ignore_tokens:
            scores[0, tok] = -float("inf")
        return scores


class TokenBiasProcessor(LogitsProcessor):
    """Add multiplicative bias to up- or down-weigh specific tokens

    :param token_ids: List of token ids to up- or down-weight
    :param bias: Multiplicative weight
    """

    def __init__(self, token_ids: list[int], bias: float) -> None:
        self.token_ids = token_ids
        self.bias = bias
        self._idx: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self.bias != 1.0 and self.token_ids:
            # Vectorized on device: `if scores[:, tok] < 0` forces a host sync per token per step
            if self._idx is None or self._idx.device != scores.device:
                self._idx = torch.tensor(self.token_ids, dtype=torch.long, device=scores.device)
            vals = scores[:, self._idx]
            scores[:, self._idx] = torch.where(vals < 0, vals / self.bias, vals * self.bias)
        return scores


USER_PROMPT = "Please describe what you see in the video in real-time. Stick to the visual facts. "
QWEN_SYSTEM_PROMPT = (
    "You are an expert video commentator providing real-time factual and neutral "
    "commentary on visual content."
)


def resolve_stop_tokens(eos_token_id: int | list[int]) -> list[int]:
    """Normalize a HF generation_config.eos_token_id into a list

    The Qwen LiveCC checkpoint has 5 end-of-turn tokens: several of its tokenizer's
    tokens decode to text containing "KeyEvent" (its special-token placeholder) and
    are not registered as HF special tokens, so any of them must be treated as a
    stop condition everywhere generation can end, or "KeyEvent" leaks into the text.
    """
    return eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]


def build_logits_processors(
    stop_tokens: list[int],
    repetition_penalty: float,
    eos_bias: float,
    ngram_size: int,
    ngram_window: int,
    repetition_penalty_decay: float = 0.99,
    repetition_penalty_max_count: float = 5.0,
) -> list[LogitsProcessor]:
    """Build the logits processors used for streaming caption generation

    :param stop_tokens: All end-of-generation token ids (exempt from repetition penalty)
    :param repetition_penalty: EMA repetition penalty strength
    :param eos_bias: Multiplicative logit bias on the stop tokens
    :param ngram_size: N-gram size for windowed repetition blocking
    :param ngram_window: Token window to scan for n-gram repeats
    :param repetition_penalty_decay: EMA decay of the repetition penalty (in (0, 1))
    :param repetition_penalty_max_count: Clamp on the EMA count before penalizing
    """
    return [
        EMARepetitionPenalty(
            penalty=repetition_penalty,
            decay=repetition_penalty_decay,
            ignore_tokens=stop_tokens,
            max_clamp=repetition_penalty_max_count,
        ),
        TokenBiasProcessor(stop_tokens, eos_bias),
        WindowedNoRepeatNGram(
            ngram_size=ngram_size, window=ngram_window, ignore_tokens=stop_tokens
        ),
    ]


def tokenize_frame_messages(
    processor: Any, frame_idx: int, prompt: str, system_prompt: str | None, is_qwen: bool
) -> dict:
    """Build and tokenize the messages for one streaming frame

    The first frame carries the system/user prompts and the assistant prefix; later
    frames are a bare image continuation of the same (never-ended) assistant turn.

    :param processor: Model processor (asst_start_tokens is cleared after the first frame)
    :param frame_idx: Index of the current frame in the stream
    :param prompt: Initial assistant prompt prefix
    :param system_prompt: Optional system prompt (first frame only)
    :param is_qwen: Whether the model is a QwenVL variant (controls BoS stripping)
    """
    if frame_idx == 0:
        messages = [
            {"role": "user", "content": [{"text": USER_PROMPT, "type": "text"}]},
            {
                "role": "assistant",
                "content": [{"image": None, "type": "image"}, {"type": "text", "text": prompt}],
            },
        ]
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": [{"text": system_prompt, "type": "text"}]}
            ] + messages
    else:
        processor.asst_start_tokens = []
        messages = [{"role": "assistant", "content": [{"image": None, "type": "image"}]}]
    inputs = processor.tokenize_messages(messages)
    assert inputs is not None, "Tokenization failed!"
    # For Helium, remove the BoS token on continuation frames
    if frame_idx > 0 and not is_qwen:
        inputs["input_ids"] = inputs["input_ids"][:, 1:]
    return inputs


def __format_srt_time__(seconds: float, ms: bool = True, full: bool = False) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if full:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    if ms:
        return f"{millis:03d} ms"
    return f"{secs:02d},{millis:03d} s"


#: A word that closes a sentence, allowing for a trailing quote or bracket. Abbreviations
#: ("Dr.", "e.g.") are deliberately not special-cased: an early cut only shortens a
#: narration unit, which is far less audible than a wrong sentence boundary would be.
SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")

#: Linear fade applied when a sentence is cut off by the next one, to avoid a click.
TTS_FADE_SECONDS = 0.01


def __group_into_sentences__(
    subtitles: list[tuple[float, float, str]],
) -> list[tuple[float, str]]:
    """Regroup per-frame caption chunks into sentences timestamped by their first word

    A caption chunk is whatever the model emitted for one frame, which is a few words
    rather than a sentence. Every word inherits the start timestamp of the chunk it came
    from, and a sentence is timestamped by its first word, so the narration starts when
    the caption started rather than when it finished.

    Sentences whose first word falls in the same chunk are merged into one unit: they are
    contiguous text spoken at the same instant, and scheduling them at an identical
    timestamp would leave the earlier one no room at all.

    :param subtitles: List of (start, end, text) caption entries, in order
    :return: List of (start, sentence text) entries with strictly increasing starts
    """
    sentences: list[tuple[float, str]] = []
    current: list[str] = []
    start = 0.0
    for chunk_start, _, text in subtitles:
        for word in text.split():
            if not current:
                start = chunk_start
            current.append(word)
            if SENTENCE_END.search(word):
                if sentences and sentences[-1][0] == start:
                    sentences[-1] = (start, f"{sentences[-1][1]} {' '.join(current)}")
                else:
                    sentences.append((start, " ".join(current)))
                current = []
    if current:
        sentences.append((start, " ".join(current)))
    return sentences


def __synthesize_tts_track__(
    subtitles: list[tuple[float, float, str]], wav_path: str | Path, voice: str = "alba"
) -> None:
    """Synthesize a narration track for the captions with pocket-tts

    Synthesis is per sentence, not per caption chunk: a chunk is a handful of words, and
    speaking each one separately gives choppy narration with no sentence prosody.

    Each sentence starts at the timestamp of its first word. When the previous sentence is
    still playing by then it is cut off (with a short fade) instead of pushing the new one
    back, so the narration never drifts behind the video no matter how slow the voice is
    relative to the caption rate.

    :param subtitles: List of (start, end, text) caption entries
    :param wav_path: Output .wav file path
    :param voice: pocket-tts voice name or audio prompt path
    """
    import wave

    import numpy as np
    from pocket_tts import TTSModel

    tts = TTSModel.load_model()
    voice_state = tts.get_state_for_audio_prompt(voice)
    sample_rate = tts.sample_rate
    sentences = __group_into_sentences__(subtitles)
    chunks: list[tuple[int, Any]] = []
    for start, text in tqdm(sentences, desc="TTS"):
        chunks.append((int(start * sample_rate), tts.generate_audio(voice_state, text).numpy()))
    if not chunks:
        return

    fade = np.linspace(1.0, 0.0, int(TTS_FADE_SECONDS * sample_rate), dtype=np.float32)
    track = np.zeros(chunks[-1][0] + len(chunks[-1][1]), dtype=np.float32)
    cut, dropped = 0, 0.0
    for i, (offset, audio) in enumerate(chunks):
        # The last sentence has the rest of the track to itself.
        budget = chunks[i + 1][0] - offset if i + 1 < len(chunks) else len(audio)
        if len(audio) > budget:
            cut += 1
            dropped += (len(audio) - budget) / sample_rate
            audio = audio[:budget].copy()
            n = min(len(fade), len(audio))
            audio[len(audio) - n :] *= fade[len(fade) - n :]
        track[offset : offset + len(audio)] = audio
    rich.print(
        f"TTS: {len(chunks)} sentences from {len(subtitles)} caption chunks; "
        f"{cut} cut short to stay in sync ({dropped:.1f}s of speech dropped)"
    )

    pcm = (np.clip(track, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def load_video(video_path: str | Path, preprocessor: Callable, tgt_fps: int) -> torch.Tensor:
    """Load video and return as a list of PIL Images sampled at 2 fps"""
    from torchcodec.decoders._video_decoder import VideoDecoder

    video = VideoDecoder(video_path, device="cpu")
    start = video.metadata.begin_stream_seconds
    max_duration = video.metadata.end_stream_seconds - start
    seconds = [start + x / tgt_fps for x in range(int(tgt_fps * max_duration))]
    frames = video.get_frames_played_at(seconds).data
    return preprocessor(frames)


def load_model(model_id: str, image_size: int = 448):
    """Load a released model and its processor from the Hugging Face hub.

    :param model_id: released model name (e.g. "CASA-Qwen2_5-VL-3B-LiveCC"), full hub repo
        id, or a path to a local checkpoint directory
    :param image_size: resolution the processor resizes frames to
    """
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.processing_auto import AutoProcessor

    repo = resolve_repo(model_id)
    token = os.getenv("HUGGINGFACE_TOKEN", None)
    # `torch_dtype` below only covers what from_pretrained loads. The Helium1 remote code
    # builds its vision tower with `_from_config(config.vision_config)` and no dtype, so
    # transformers falls back to `vision_config.torch_dtype` (unset in the released configs):
    # the tower is built in fp32 and logs "attempting to use Flash Attention 2.0 without
    # specifying a torch dtype". Pinning the dtype on the config covers those submodules.
    config = AutoConfig.from_pretrained(repo, trust_remote_code=True, token=token)
    config.torch_dtype = torch.bfloat16
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.torch_dtype = torch.bfloat16
    model = AutoModel.from_pretrained(
        repo,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        token=token,
    )
    processor = AutoProcessor.from_pretrained(
        repo, image_size=image_size, trust_remote_code=True, token=token
    )
    # device_map="cuda" already placed the weights; initializing on CPU first would also
    # trip the flash-attention "model not initialized on GPU" warning
    return model.eval(), processor  # type: ignore


def gen_subtitles(
    sample_path: str,
    model_id: str = "CASA-Qwen2_5-VL-3B-LiveCC",
    max_new_tokens: int = 20,
    fps: int = 2,
    repetition_penalty: float = 1.15,
    prompt: str = "In this video",
    temp: float = 0.4,
    top_k: int = 256,
    eos_bias: float = 0.95,
    output_dir: str = "./livecc_samples",
    srt: bool = False,
    tts_voice: str | None = "alba",
    ngram_size: int = 4,
    ngram_window: int = 60,
    repetition_penalty_decay: float = 0.99,
    repetition_penalty_max_count: float = 5.0,
    image_size: int = 448,
    compile_vision: bool = True,
    fold_stop_into_prefill: bool = True,
):
    """Live Captioning

    :param sample_path: Path to video
    :param model_id: Model to use
    :param max_new_token: Max number of tokens to generate per frame
    :param fps: Fps for video frame extraction
    :param repetition_penalty: Repetition penalty
    :param prompt: Initial prompt
    :param temp: Sampling temperature
    :param top_k: Sampling top_k
    :param eos_bias: Reweigh the end of generation per frame
    :param output_dir: Where to save captioning outputs"
    :param srt: Whether to generate the new video with embedded subtitle. If
        False, will only generate subtitles in a json file
    :param tts_voice: pocket-tts voice used to narrate the captions in the output
        video (replaces the original audio track). Only used with srt; set to
        None to keep the original audio
    :param ngram_size: N-gram size for windowed repetition blocking
    :param ngram_window: Token window to scan for n-gram repeats (O(window) per step)
    :param repetition_penalty_decay: EMA decay of the repetition penalty (in (0, 1))
    :param repetition_penalty_max_count: Clamp on the EMA count before penalizing
    :param image_size: Frame resolution fed to the vision tower. Match the checkpoint's
        training resolution: the livecc runs use video_image_size=448, and the 896 default costs
        4x the vision compute and 4x the cross-attention image KV at a resolution the model
        never saw on video
    :param compile_vision: torch.compile the vision tower. Every frame has the same shape, so
        the tower is a static-shape module that compiles cleanly; expect warmup on the first
        frame(s), then a faster steady state
    :param fold_stop_into_prefill: Record the per-frame stop token in the KV cache by prepending
        it to the next frame's prefill instead of spending a dedicated 1-token forward per frame
    """

    model, processor = load_model(model_id, image_size=image_size)
    if compile_vision:
        # Qwen's vision attention does `max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()`,
        # a scalar readout dynamo cannot trace: it splits the tower into subgraphs there and logs a
        # graph-break warning per layer. The split is benign (subgraphs are compiled and cached
        # after the first frame), so just silence the one-time warning wall. Do NOT "fix" it with
        # torch._dynamo.config.capture_scalar_outputs: on this torch version the unbacked symbol
        # flows into repeat_interleave inside the tower and compilation dies with
        # PendingUnbackedSymbolNotFound.
        logging.getLogger("torch._dynamo.variables.tensor").setLevel(logging.ERROR)
        # The tower sits at image_prefix.visual on the Qwen release and image_prefix.enc.visual
        # on the Helium one.
        tower_owner = (
            model.image_prefix
            if hasattr(model.image_prefix, "visual")
            else getattr(model.image_prefix, "enc", None)
        )
        if tower_owner is not None and hasattr(tower_owner, "visual"):
            tower_owner.visual = torch.compile(tower_owner.visual)
        else:
            rich.print("[yellow]compile_vision: no vision tower found, skipping[/yellow]")
    is_qwenvl_model = "qwen" in model_id.lower()
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Load a video file
    video_path = Path(sample_path)
    video = load_video(
        video_path.resolve(),
        preprocessor=processor._image_processor.process_images,
        tgt_fps=fps,
    ).to("cuda")
    movie_name, movie_ext = os.path.basename(video_path).rsplit(".", 1)
    model_name = model_id.rstrip("/").split("/")[-1]
    movie_name = f"{movie_name}_{model_name}"
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{movie_name}_fps={fps}_rp={repetition_penalty}_t={temp}_e={eos_bias}_subtitled",
    )
    final_video_path = f"{output_path}.{movie_ext}"
    output_data_path = f"{output_path}_data.jsonl"
    with open(output_data_path, "w"):
        pass

    kv_cache = DynamicCache()
    stop_tokens = resolve_stop_tokens(model.generation_config.eos_token_id)
    stop_token = stop_tokens[0]
    logits_processor = build_logits_processors(
        stop_tokens=stop_tokens,
        repetition_penalty=repetition_penalty,
        eos_bias=eos_bias,
        ngram_size=ngram_size,
        ngram_window=ngram_window,
        repetition_penalty_decay=repetition_penalty_decay,
        repetition_penalty_max_count=repetition_penalty_max_count,
    )
    system_prompt = QWEN_SYSTEM_PROMPT if is_qwenvl_model else None

    # Let's start generating
    query_idx = 0
    total_time = 0.0
    total_tokens_generated = 0
    subtitles: list[tuple[float, float, str]] = []
    subtitles_delay: list[tuple[float, float, str]] = []
    # Generation is represented as one long `assistant turn` so we do not end it
    processor.asst_end_tokens = []
    full_caption = ""

    timing_hook = TimingHook()
    handle = model.register_forward_hook(timing_hook)
    pending_stop_token: int | None = None

    for query_idx in tqdm(range(len(video))):
        start_timestamp = query_idx / fps
        timing_hook.reset()

        # Tokenize the frame messages
        inputs = tokenize_frame_messages(
            processor, query_idx, prompt, system_prompt, is_qwenvl_model
        )
        if pending_stop_token is not None:
            # The previous frame's stop token was sampled but never forwarded (generation ends
            # before its forward), so its KV state is still missing from the cache. Prepending it
            # here records it during this frame's prefill for free, replacing the dedicated
            # 1-token forward per frame. The image insertion points are indices into input_ids
            # and must shift with it.
            inputs["input_ids"] = torch.cat(
                [
                    torch.tensor([[pending_stop_token]], dtype=inputs["input_ids"].dtype),
                    inputs["input_ids"],
                ],
                dim=1,
            )
            points = inputs.get("image_embeds_insertion_points")
            if points is not None:
                inputs["image_embeds_insertion_points"] = (
                    [p + 1 for p in points] if isinstance(points, list) else points + 1
                )
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].cuda()
        inputs.pop("attention_mask", None)
        inputs.pop("pixel_values", None)

        # Generate call
        out = model.generate_from_image(
            **inputs,
            pixel_values=list(video[query_idx : (query_idx + 1)].cuda().to(torch.bfloat16)),
            reset_streaming=False,
            max_new_tokens=max_new_tokens,
            do_sample=temp > 0,
            temperature=temp,
            top_k=top_k,
            past_key_values=kv_cache,
            logits_processor=logits_processor,
            attention_mask=torch.ones((1, inputs["input_ids"].shape[1]), device="cuda"),
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        num_tokens_generated = out.shape[1] - inputs["input_ids"].shape[1]

        # Track which stop token the model actually emitted. Feeding a DIFFERENT one into the KV
        # cache below poisons the stream: a frame-boundary model separates windows with its minted
        # cadence token, and replacing it with <|endoftext|> (a document separator) makes the
        # context look like concatenated web documents — after a few dozen frames the model drops
        # the commentary cadence entirely and free-runs into max_new_tokens every frame.
        last_token = int(out[0, -1].item())
        ended_on_stop = last_token in stop_tokens
        if ended_on_stop:
            stop_token = last_token

        # Decode; the last token is only dropped when it is a stop token (some baselines' stop ids
        # decode to visible text). On a max_new_tokens truncation the last token is a real word.
        pred_s = type_cast(
            str,
            processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] : out.shape[1] - int(ended_on_stop)],
                skip_special_tokens=True,
            ),
        ).strip()

        # The stop token was sampled but never forwarded, so its KV state is not in the cache yet.
        # Either defer it to the next frame's prefill (free) or run a dedicated 1-token pass.
        if fold_stop_into_prefill:
            pending_stop_token = stop_token
        else:
            with torch.no_grad():
                model.forward(
                    torch.tensor([stop_token])[None, :].cuda(),
                    pixel_values=None,
                    past_key_values=kv_cache,
                    use_cache=True,
                    attention_mask=torch.ones((1, 1), device="cuda"),
                    position_ids=torch.ones(
                        (3, 1, 1) if is_qwenvl_model else (1, 1), dtype=torch.long
                    ).cuda()
                    * kv_cache._seen_tokens,
                    __is_first_gen_call__=False,
                )
        model.reset_ca_streaming_states()

        # Add prompt for first generation
        if query_idx == 0 and prompt is not None:
            pred_s = prompt + " " + pred_s

        # Use time to first token to display first subtitle
        ttft = timing_hook.timings[0]
        total_time += sum(timing_hook.timings)
        memory_so_far = max(timing_hook.mems)
        end_timestamp = (query_idx + 1) / fps
        subtitles.append(
            (
                start_timestamp + (ttft if query_idx > 0 else 0),
                end_timestamp,
                pred_s.replace("\n", ""),
            )
        )
        total_tokens_generated += out.shape[1] + 1
        num_toks = kv_cache._seen_tokens

        subtitles_delay.append(
            (
                start_timestamp,
                end_timestamp,
                f"[avg. ttftok: {__format_srt_time__(sum(timing_hook.timings) / len(timing_hook.timings), ms=True)}]\n"
                f"[avg. tok/s: {__format_srt_time__(total_time / total_tokens_generated, ms=True)}]\n"
                f"[KV cache: {num_toks} toks]\n"
                f"[mem: {memory_so_far:05.2f} GB]",
            )
        )

        with open(output_data_path, "a") as wf:
            wf.write(
                json.dumps(
                    dict(
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        ttft=timing_hook.timings[0],
                        num_tokens_generated=num_tokens_generated,
                        memory_so_far=memory_so_far,
                        subtitle=pred_s,
                    )
                )
                + "\n"
            )

        # Display current Gen
        if not srt:
            full_caption += f"[grey]{__format_srt_time__(start_timestamp, full=True)}[/grey] (mem: {memory_so_far:.2f} GB) [bold green]{pred_s.strip()}[/bold green]\n"
    handle.remove()
    if not srt:
        rich.print(full_caption)
        rich.print(f"\nGenerated captions output in [yellow]{output_data_path}[/yellow]")

    # Write subtitles file
    if srt:
        subtitles_file = ["output_subtitles_1.srt", "output_subtitles_2.srt"]
        subtitles_file = [Path(x) for x in subtitles_file]
        for sbt, sbt_file in zip([subtitles, subtitles_delay], subtitles_file):
            with open(sbt_file.resolve(), "w", encoding="utf-8") as f:
                for i, (start, end, text) in enumerate(sbt, 1):
                    f.write(f"{i}\n")
                    f.write(
                        f"{__format_srt_time__(start, full=True)} --> {__format_srt_time__(end, full=True)}\n"
                    )
                    f.write(f"{text}\n\n")

        __convert_to_progressive_subtitles__(
            subtitles_file[0].resolve(), subtitles_file[0].resolve()
        )

        # Embed subtitles into the video
        # Setup:
        # mkdir ~/fonts
        # cd ~/fonts
        # wget https://github.com/googlefonts/roboto-3-classic/releases/download/v3.012/Roboto_v3.012.zip
        # unzip Roboto_v3.012.zip
        #  mv unhinted/static/Roboto*.ttf .
        # rm -rf android/ chromeos/ hinted/ unhinted/ web/ __MACOSX/ Roboto_v3.012.zip
        # Main subtitle
        subtitle_format = """fontsdir=$HOME/fonts:force_style='FontName=Roboto,FontSize=28,PrimaryColour=&H0039F2AE,OutlineColour=&H80000000,Outline=3,Bold=1,Alignment=1,MarginV=20'"""
        movie_name, ext = os.path.basename(video_path).rsplit(".", 1)
        output_video_path = Path(f"{movie_name}_subtitle_temp.{ext}")
        cmd = f'ffmpeg -y -i {video_path.resolve()} -vf subtitles="{subtitles_file[0].resolve()}:{subtitle_format}" {output_video_path.resolve()}'
        subprocess.run(cmd, shell=True)

        # Narrate the captions with pocket-tts, replacing the original audio track
        tts_wav_path = Path(f"{output_path}_tts.wav")
        audio_flags = ""
        if tts_voice is not None:
            __synthesize_tts_track__(subtitles, tts_wav_path, voice=tts_voice)
            # apad + -shortest keeps the video duration: narration overrunning the end
            # is cut, and shorter narration is padded with silence
            audio_flags = (
                f'-i "{tts_wav_path.resolve()}" -map 0:v -map 1:a -c:a aac -af apad -shortest'
            )

        # Delay subtitles
        subtitle_format = """fontsdir=$HOME/fonts:force_style='FontName=Roboto,FontSize=16,PrimaryColour=&H0039F2AE,OutlineColour=&H80000000,Outline=1,Bold=0,Alignment=4,MarginR=20'"""
        cmd = f'ffmpeg -y -i {output_video_path} {audio_flags} -vf subtitles="{subtitles_file[1]}:{subtitle_format}" {final_video_path}'
        subprocess.run(cmd, shell=True)
        rich.print(f"Final video output in [yellow]{final_video_path}[/yellow]")
        rich.print(f"Generated captions output in [yellow]{output_data_path}[/yellow]")

        # Cleanup
        for x in subtitles_file + [output_video_path, tts_wav_path]:
            if x.exists():
                x.unlink()


if __name__ == "__main__":
    Fire(gen_subtitles)
