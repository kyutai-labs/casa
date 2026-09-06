# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "numpy",
#     "pillow",
#     "einops>=0.8.1",
#     "accelerate",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Streaming multi-turn inference benchmark for the release checkpoints.

Simulates the live captioning / conversational setting (same loop as
scripts/live_captioner.py): a persistent KV cache to which every turn appends one
new image and a fixed number of generated text tokens. For insertion models the
image tokens accumulate in the KV cache; for CA models only the text does, with
cross-attention reading the latest image.

Per turn, we record time-to-first-token (image encoding + prefill of the new
frame), decode throughput, KV cache length, and peak GPU memory, so the scaling
with the number of turns is directly visible. Batch size is 1, matching the
live streaming use case.

Example usage:
    uv run scripts/speed_bench_streaming.py CASA-Helium1-VL-2B --num_turns 64
    uv run scripts/speed_bench_streaming.py Helium1-VL-2B --num_turns 256 --tokens_per_turn 32
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

PROMPT = "Please describe what you see in the video in real-time. Stick to the visual facts."
SYSTEM_PROMPT = (
    "You are an expert video commentator providing real-time factual and neutral "
    "commentary on visual content."
)

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


def load_model(model_id: str, image_size: int, device: str = "cuda") -> tuple[Any, Any]:
    """Load a released model and its processor from the Hugging Face hub.

    :param model_id: released model name (e.g. "CASA-Qwen2_5-VL-3B"), full hub repo id,
        or a path to a local checkpoint directory
    :param image_size: resolution the processor resizes frames to
    :param device: where to place the weights
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
        device_map=device,
        token=token,
    )
    processor = AutoProcessor.from_pretrained(
        repo, image_size=image_size, trust_remote_code=True, token=token
    )
    # device_map already placed the weights; initializing on CPU and moving afterwards
    # trips the flash-attention "model not initialized on GPU" warning
    return model.eval(), processor  # type: ignore


class ForwardTimer:
    """Forward hook recording the time elapsed between consecutive model calls.

    After a generate call, timings[0] is the prefill time of the new inputs
    (time to first token) and timings[1:] are the per-token decoding times.
    """

    def __init__(self):
        self.timings: list[float] = []
        self.current_start: float | None = None

    def reset(self) -> None:
        self.timings = []
        self.current_start = time.perf_counter()

    def __call__(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        torch.cuda.synchronize()
        if self.current_start is not None:
            self.timings.append(time.perf_counter() - self.current_start)
        self.current_start = time.perf_counter()


def make_frame(rng: np.random.Generator, image_size: int) -> Image.Image:
    return Image.fromarray(
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8), mode="RGB"
    )


def run_stream(
    model: Any,
    processor: Any,
    model_id: str,
    image_size: int,
    num_turns: int,
    tokens_per_turn: int,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Run one full streaming session and return per-turn statistics."""
    is_qwen = "qwen" in model_id.lower()
    rng = np.random.default_rng(seed=0)
    eos_token_id = model.generation_config.eos_token_id
    stop_token = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

    # Generation is represented as one long assistant turn, so we do not end it
    processor.asst_end_tokens = []
    kv_cache = DynamicCache()
    timer = ForwardTimer()
    handle = model.register_forward_hook(timer)
    per_turn: list[dict[str, float]] = []

    try:
        for turn in range(num_turns):
            frame = make_frame(rng, image_size)
            if turn == 0:
                messages = (
                    [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
                    if is_qwen
                    else []
                ) + [
                    {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
                    {"role": "assistant", "content": [{"type": "image", "image": frame}]},
                ]
            else:
                processor.asst_start_tokens = []
                messages = [{"role": "assistant", "content": [{"type": "image", "image": frame}]}]

            # No BoS token on continuation turns. Suppressing it at tokenization
            # (rather than slicing input_ids) keeps image_embeds_insertion_points
            # consistent with the token positions.
            inputs = processor.tokenize_messages(messages, suppress_bos_token=turn > 0)
            inputs = inputs.to(model.device)
            inputs.pop("attention_mask", None)
            # The image processor returns fp32; the model runs in bf16. Depending on the
            # checkpoint, pixel_values is either a tensor or a list of per-image tensors.
            pixel_values = inputs.get("pixel_values")
            if isinstance(pixel_values, list):
                inputs["pixel_values"] = [p.to(torch.bfloat16) for p in pixel_values]
            elif pixel_values is not None:
                inputs["pixel_values"] = pixel_values.to(torch.bfloat16)

            torch.cuda.reset_peak_memory_stats()
            timer.reset()
            out = model.generate_from_image(
                **inputs,
                reset_streaming=False,
                do_sample=False,
                min_new_tokens=tokens_per_turn,
                max_new_tokens=tokens_per_turn,
                past_key_values=kv_cache,
                attention_mask=torch.ones((1, inputs["input_ids"].shape[1]), device=model.device),
                eos_token_id=eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            ttft = timer.timings[0]
            decode_s = sum(timer.timings[1:])
            num_generated = out.shape[1] - inputs["input_ids"].shape[1]

            # Record a stop token in the KV cache since generation was cut before
            # it could be emitted, then reset the CA streaming state for the next frame
            model.forward(
                torch.tensor([stop_token])[None, :].to(model.device),
                pixel_values=None,
                past_key_values=kv_cache,
                use_cache=True,
                attention_mask=torch.ones((1, 1), device=model.device),
                position_ids=torch.ones(
                    (3, 1, 1) if is_qwen else (1, 1), dtype=torch.long, device=model.device
                )
                * kv_cache._seen_tokens,
                __is_first_gen_call__=False,
            )
            model.reset_ca_streaming_states()

            per_turn.append(
                {
                    "turn": turn,
                    "ttft_s": ttft,
                    "decode_s": decode_s,
                    "decode_tok_per_s": (num_generated - 1) / decode_s if decode_s > 0 else 0.0,
                    "kv_tokens": int(kv_cache._seen_tokens),
                    "peak_mem_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )
            if verbose:
                text = processor.tokenizer.decode(
                    out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                print(f"[turn {turn}] {text.strip()}")
    except torch.OutOfMemoryError:
        # Expected for insertion models at long horizons: the KV cache grows with
        # every frame. Keep the turns completed so far; the summary records the OOM.
        per_turn.append({"turn": len(per_turn), "oom": True})
    finally:
        handle.remove()
        model.reset_ca_streaming_states()

    return per_turn


@torch.inference_mode()
def main(
    model_id: str,
    num_turns: int = 64,
    tokens_per_turn: int = 16,
    image_size: int = 896,
    num_warmup_turns: int = 2,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Benchmark streaming multi-turn inference speed and memory.

    :param model_id: released model name (see RELEASED_MODELS), full hub repo id, or a
        path to a local checkpoint directory
    :param num_turns: number of image+text turns in the streaming session
    :param tokens_per_turn: text tokens generated per turn (forced length, greedy)
    :param image_size: input image resolution
    :param num_warmup_turns: turns of an initial discarded session (kernel warmup)
    :param overwrite: recompute settings already present in the result file
    :param verbose: print the generated text at every turn
    """
    # A hub repo id or a local path would otherwise turn the filename into nested dirs
    model_name = model_id.rstrip("/").split("/")[-1]
    result_path = Path("result_logs") / f"speed_streaming_{model_name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if result_path.exists():
        with open(result_path, "r") as f:
            results = json.load(f)

    key = f"turns{num_turns}_toks{tokens_per_turn}"
    if key in results and not overwrite:
        print(f"Skipping {key} (already computed)")
        return

    model, processor = load_model(model_id, image_size)
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    results.setdefault("meta", {}).update(
        {"gpu": torch.cuda.get_device_name(), "image_size": image_size}
    )

    run_stream(model, processor, model_id, image_size, num_warmup_turns, tokens_per_turn)
    per_turn = run_stream(
        model, processor, model_id, image_size, num_turns, tokens_per_turn, verbose=verbose
    )

    completed = [t for t in per_turn if "oom" not in t]
    oom = len(completed) < num_turns
    tail = completed[-max(1, len(completed) // 10) :]
    head = completed[: max(1, len(completed) // 10)]
    summary = {
        "completed_turns": len(completed),
        "oom": oom,
        "total_time_s": sum(t["ttft_s"] + t["decode_s"] for t in completed),
        "ttft_s_first_turns": sum(t["ttft_s"] for t in head) / len(head),
        "ttft_s_last_turns": sum(t["ttft_s"] for t in tail) / len(tail),
        "decode_tok_per_s_first_turns": sum(t["decode_tok_per_s"] for t in head) / len(head),
        "decode_tok_per_s_last_turns": sum(t["decode_tok_per_s"] for t in tail) / len(tail),
        "final_kv_tokens": completed[-1]["kv_tokens"],
        "peak_mem_gib": max(t["peak_mem_gib"] for t in completed),
    }
    results[key] = {"summary": summary, "per_turn": per_turn}
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[{model_id} | {key}]" + (" OOM!" if oom else ""))
    print(
        f"  ttft first/last turns: {summary['ttft_s_first_turns'] * 1000:.1f} / "
        f"{summary['ttft_s_last_turns'] * 1000:.1f} ms"
    )
    print(
        f"  decode first/last turns: {summary['decode_tok_per_s_first_turns']:.1f} / "
        f"{summary['decode_tok_per_s_last_turns']:.1f} tok/s"
    )
    print(
        f"  final KV cache: {summary['final_kv_tokens']} tokens | "
        f"peak mem {summary['peak_mem_gib']:.1f} GiB | "
        f"total {summary['total_time_s']:.1f} s"
    )


if __name__ == "__main__":
    fire.Fire(main)
