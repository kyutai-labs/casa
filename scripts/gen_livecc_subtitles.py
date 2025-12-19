# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "rich>=12.6.0",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchcodec==0.4.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Live Captioning Inference

uv run gen_livecc_subtitles --help
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from typing import cast as type_cast

import rich
import torch
from fire import Fire
from torchcodec.decoders._video_decoder import VideoDecoder
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessor
from transformers.models.auto.modeling_auto import AutoModel


class TimingHook:
    """Hook to measure token time generation"""

    def __init__(self):
        self.timings = []
        self.mems = []
        self.current_start = None

    def reset(self):
        self.timings = []
        self.mems = []
        self.current_start = time.perf_counter()

    def __call__(self, module: torch.nn.Module, input: Any, output: Any):
        torch.cuda.synchronize()
        if self.current_start is not None:
            elapsed = time.perf_counter() - self.current_start
            self.timings.append(elapsed)
            self.mems.append(torch.cuda.memory.max_memory_allocated() / (1024**3))
        self.current_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()


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
        processor = EMARepetitionPenalty(penalty=1.2, vocab_size=50257, decay=0.99)
        ...
        while generating:
            logits = processor(input_ids, logits)
            next_token = sample(logits)
            processor.update(next_token)
    """

    def __init__(
        self,
        penalty: float,
        vocab_size: int,
        decay: float = 0.99,
        ignore_tokens: list[int] | None = None,
        max_clamp: float = 5.0,
    ):
        super().__init__()
        if not (0 < decay < 1):
            raise ValueError("decay must be in (0,1).")

        self.penalty = float(penalty)
        self.vocab_size = vocab_size
        self.decay = float(decay)
        self.ignore_tokens = [] if ignore_tokens is None else ignore_tokens
        self.ema_counts = torch.zeros(vocab_size, dtype=torch.float32)
        self.max_clamp = max_clamp

    # ----------------------------------------------------------------------
    # 1) EXPLICIT UPDATE CALL
    # ----------------------------------------------------------------------
    def update(self, token_id: int | torch.Tensor):
        """
        Tell the processor which token was actually sampled.
        This updates the EMA state, but does NOT modify logits.
        """
        if isinstance(token_id, torch.Tensor):
            token_id = int(token_id.item())

        # Decay all old counts
        self.ema_counts.mul_(self.decay)

        # Add current token
        self.ema_counts[token_id] += 1

    def reset(self):
        """Clear EMA state between independent generations."""
        self.ema_counts.zero_()

    # ----------------------------------------------------------------------
    # 2) PURE PENALTY APPLICATION
    # ----------------------------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):  # type: ignore[override]
        """
        Apply EMA-based repetition penalty to logits.
        Does NOT update EMA; that must be done via update().
        """
        if self.penalty == 1.0:
            return scores

        if self.ema_counts.device != scores.device:
            self.ema_counts = self.ema_counts.to(scores.device)

        # Update
        assert input_ids.shape[0] == 1, "EMARepetitionPeanlty expects batch size 1"
        latest_token_id = int(input_ids[0, -1].item())
        self.ema_counts.mul_(self.decay)
        self.ema_counts[latest_token_id] += 1

        # Compute penalties
        ema = self.ema_counts.clamp(min=0.0, max=self.max_clamp)
        penalty_factors = self.penalty**ema
        for tok in self.ignore_tokens:
            penalty_factors[tok] = 1.0

        # Apply penalties
        pf = penalty_factors[None, :]
        scores = torch.where(scores < 0, scores * pf, scores / pf)
        return scores


class TokenBiasProcessor(LogitsProcessor):
    """Add multiplicative bias to up- or down-weigh specific tokens

    :param token_ids: List of token ids to up- or down-weight
    :param bias: Multiplicative weight
    """

    def __init__(self, token_ids: list[int], bias: float) -> None:
        self.token_ids = token_ids
        self.bias = bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self.bias != 1.0:
            # scores shape: (batch_size, vocab_size)
            for token_id in self.token_ids:
                if scores[:, token_id] < 0:
                    scores[:, token_id] /= self.bias
                else:
                    scores[:, token_id] *= self.bias
        return scores


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


def load_video(video_path: str | Path, preprocessor: Callable, tgt_fps: int) -> torch.Tensor:
    """Load video and extract frames at 2 fps"""
    video = VideoDecoder(video_path, device="cpu")
    start = video.metadata.begin_stream_seconds
    max_duration = video.metadata.end_stream_seconds - start
    seconds = [start + x / tgt_fps for x in range(int(tgt_fps * max_duration))]
    frames = video.get_frames_played_at(seconds).data
    return preprocessor(frames)


def gen_subtitles(
    sample_path: str,
    max_new_tokens: int = 20,
    fps: int = 2,
    repetition_penalty: float = 1.15,
    prompt: str = "This video shows",
    temp: float = 0.4,
    top_k: int = 256,
    eos_bias: float = 1.0,
    image_size: int = 448,
    output_dir: str = "./livecc_samples",
    srt: bool = True,
):
    """Live Captioning

    :param sample_path: Path to video
    :param max_new_token: Max number of tokens to generate per frame
    :param fps: Fps for video frame extraction
    :param repetition_penalty: Repetition penalty
    :param prompt: Initial prompt
    :param temp: Sampling temperature
    :param top_k: Sampling top_k
    :param eos_bias: Reweigh the end of generation per frame
    :param image_size: Input image size
    :param output_dir: Where to save captioning outputs"
    :param srt: Whether to generate the new video with embedded subtitle. If
        False, will only generate subtitles in a json file
    """
    # Init model and processor
    model_id = "kyutai/CASA-Qwen-2_5-VL-3B-LiveCC"
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(
        model_id, image_size=image_size, trust_remote_code=True
    )
    is_qwenvl_model = True

    # Load a video file
    video_path = Path(sample_path)
    video = load_video(
        video_path.resolve(),
        preprocessor=processor._image_processor.process_images,
        tgt_fps=fps,
    ).to("cuda")
    movie_name, movie_ext = os.path.basename(video_path).rsplit(".", 1)
    movie_name = f"{movie_name}_{model_id.replace('/', '_')}"
    output_dir = os.path.join(output_dir, model_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{movie_name}_fps={fps}_rp={repetition_penalty}_t={temp}_e={eos_bias}_subtitled",
    )
    final_video_path = f"{output_path}.{movie_ext}"
    output_data_path = f"{output_path}_data.jsonl"
    with open(output_data_path, "w"):
        pass

    logits_processor = []
    system_prompt: str | None
    kv_cache = DynamicCache()
    stop_token = model.generation_config.eos_token_id
    user_prompt = "Explain what is happening in the video in great details."
    if is_qwenvl_model:
        logits_processor = [
            EMARepetitionPenalty(
                penalty=repetition_penalty,
                vocab_size=151936,
                # weird edge case where 715 linejump token
                ignore_tokens=[220, stop_token],
            ),
            TokenBiasProcessor([stop_token], eos_bias),
        ]

        system_prompt = (
            "You are an expert video commentator providing real-time,"
            " insightful, and engaging commentary on visual content."
        )
    else:
        logits_processor = [
            EMARepetitionPenalty(
                penalty=repetition_penalty,
                vocab_size=len(processor.tokenizer),
                ignore_tokens=[stop_token],
            ),
            TokenBiasProcessor([stop_token], eos_bias),
        ]
        system_prompt = None

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

    for query_idx in tqdm(range(len(video))):
        start_timestamp = query_idx / fps
        timing_hook.reset()

        # Input messages
        if query_idx == 0:
            messages = [
                {
                    "role": "user",
                    "content": [{"text": user_prompt, "type": "text"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"image": None, "type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": [{"text": system_prompt, "type": "text"}]}
                ] + messages
        else:
            processor.asst_start_tokens = []
            messages = [{"role": "assistant", "content": [{"image": None, "type": "image"}]}]

        # Tokenize
        inputs = processor.tokenize_messages(messages)
        # fi Helium, remove BoS token
        if query_idx > 0 and not is_qwenvl_model:
            inputs["input_ids"] = inputs["input_ids"][:, 1:]
        assert inputs is not None, "Tokenization failed!"
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
            pre_image_tokens=processor.pre_image_tokens,
            post_image_tokens=processor.post_image_tokens,
            attention_mask=torch.ones((1, inputs["input_ids"].shape[1]), device="cuda"),
            eos_token_id=stop_token,
        )
        num_tokens_generated = out.shape[1] - inputs["input_ids"].shape[1]

        # Decode
        pred_s = type_cast(
            str,
            processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ),
        ).strip()

        # Prepare for next iter with one fake pass with the eos_token_id
        # since we stopped generation before it could be recorded inside the KV cache
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
                reinit_casa_handler=False,
            )
            model.reset_casa_streaming_states()

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

        # Main subtitle
        subtitle_format = """fonts:force_style='FontSize=28,PrimaryColour=&H0039F2AE,OutlineColour=&H80000000,Outline=3,Bold=1,Alignment=1,MarginV=20'"""
        movie_name, ext = os.path.basename(video_path).rsplit(".", 1)
        output_video_path = Path(f"{movie_name}_subtitle_temp.{ext}")
        cmd = f'ffmpeg -y -i {video_path.resolve()} -vf subtitles="{subtitles_file[0].resolve()}:{subtitle_format}" {output_video_path.resolve()}'
        subprocess.run(cmd, shell=True)

        # Delay subtitles
        subtitle_format = """fonts:force_style='FontSize=16,PrimaryColour=&H0039F2AE,OutlineColour=&H80000000,Outline=1,Bold=0,Alignment=4,MarginR=20'"""
        cmd = f'ffmpeg -y -i {output_video_path} -vf subtitles="{subtitles_file[1]}:{subtitle_format}" {final_video_path}'
        subprocess.run(cmd, shell=True)
        rich.print(f"Final video output in [yellow]{final_video_path}[/yellow]")
        rich.print(f"Generated captions output in [yellow]{output_data_path}[/yellow]")

        # Cleanup
        for x in subtitles_file + [output_video_path]:
            if x.exists():
                x.unlink()


if __name__ == "__main__":
    Fire(gen_subtitles)
