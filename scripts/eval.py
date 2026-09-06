# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "rich>=12.6.0",
#     "icecream",
#     "anls>=0.0.2",
#     "anls-star>=0.0.12",
#     "lmms-eval @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git",
#     "datasets>=3.4.1",
#     "einops>=0.8.1",
#     "accelerate",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Inference + evaluation with lmms lab"""

import json
import os
import re
import string
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.distributed as dist
from anls import anls_score
from anls_star import anls_score as anls_star_score
from datasets import Dataset, load_dataset
from fire import Fire
from icecream import ic
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

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


def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group when launched under torchrun.

    Falls back to single-process mode (rank 0, world size 1) otherwise, so the
    script still runs with a plain `uv run scripts/eval.py`.

    :return: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            # Generous timeout: ranks reach the final all_gather barrier at slightly
            # different times depending on shard contents, and the first collective
            # also lazily sets up the NCCL communicator (600s default is too tight).
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def load_eval_fn(name: str) -> tuple[Dataset, Callable]:
    if name == "chartqa":
        return get_eval_chartqa()
    if name == "realworldqa":
        return get_eval_realworldqa()
    if name == "textvqa":
        return get_eval_textvqa()
    if name == "ai2d":
        return get_eval_ai2d()
    if name == "mme":
        return get_eval_mme()
    if name == "ocrbench":
        return get_eval_ocrbench()
    if name == "docvqa":
        return get_eval_docvqa()
    if name == "infographic_vqa":
        return get_eval_infographic_vqa()
    if name == "gqa":
        return get_eval_gqa()
    raise ValueError(f"Unsupported benchmark: {name}")


def anls_evaluate(
    predictions: dict[str, str],
    eval_dataset: Dataset,
) -> dict[str, float]:
    def __check_for_match__(pred: str, gt: str | list[str]) -> list[str]:
        """Check whether pred and gt are matching
        :param pred: A prediction string
        :param gt: A ground-truth string or list of strings

        :return: List of matching answers
        """
        pred = pred.lower().strip()
        prefix = "^"
        suffix = r"(\b|\.)"
        if isinstance(gt, str):
            gt_answers = [gt.lower().strip()]
        else:
            gt_answers = [ans.lower().strip() for ans in gt]
        patterns = [
            re.compile(f"{prefix}{re.escape(ans)}{suffix}", re.MULTILINE) for ans in gt_answers
        ]
        matching_answers = [
            gt[idx] for idx, pat in enumerate(patterns) if (pat.search(pred) is not None)
        ]
        return matching_answers

    accs: dict[str, list[float]] = {
        "acc": [],
        "anls": [],
        "anls_star": [],
    }
    for idx, item in enumerate(eval_dataset):
        question_id = str(idx)
        if question_id not in predictions:
            continue
        gt_answers = item["answers"]
        # Our models tend to add a full stop at the end of the answer,
        # which can significantly reduce the ANLS scores.
        pred = predictions[question_id].rstrip(".").strip()

        matching_answers = __check_for_match__(pred=pred, gt=gt_answers)
        accs["acc"].append(len(matching_answers) > 0)
        accs["anls"].append(
            anls_score(
                gold_labels=gt_answers,
                prediction=pred,
            )
        )
        accs["anls_star"].append(anls_star_score(gt=gt_answers, pred=pred))

    return {
        "acc": sum(accs["acc"]) / len(accs["acc"]) * 100,
        "anls": sum(accs["anls"]) / len(accs["anls"]) * 100,
        "anls_star": sum(accs["anls_star"]) / len(accs["anls_star"]) * 100,
    }


def lmms_evaluate(
    predictions: dict[str, str],
    eval_dataset: Dataset,
    process_fn: Callable[[dict[str, Any], list[str]], dict[str, Any]],
    aggregate_fn: Callable[[list[dict[str, Any]]], float],
) -> float:
    res = []
    for idx, item in enumerate(eval_dataset):
        if str(idx) in predictions:
            res.append(process_fn(item, [predictions[str(idx)]]))
    res = aggregate_fn(res)
    return res


def get_eval_textvqa():
    eval_dataset = load_dataset("lmms-lab/textvqa").select_columns(
        ["answers", "question_id", "image", "question"]
    )["validation"]

    def eval_textvqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from lmms_eval.tasks.textvqa.utils import textvqa_process_results  # type: ignore

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            return 100 * sum([x["exact_match"] for x in results]) / len(results)

        return lmms_evaluate(predictions, eval_dataset, textvqa_process_results, aggregate_results)

    return eval_dataset, eval_textvqa


def get_eval_realworldqa():
    eval_dataset = load_dataset("lmms-lab/RealWorldQA").select_columns(
        ["answer", "question", "image", "question"]
    )["test"]

    def eval_realworldqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from lmms_eval.tasks.realworldqa.utils import realworldqa_process_results

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            predictions = [predictions[0].rstrip(".")]
            return realworldqa_process_results(doc, predictions)

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            return 100 * sum([x["exact_match"] for x in results]) / len(results)

        return lmms_evaluate(predictions, eval_dataset, process_results, aggregate_results)

    return eval_dataset, eval_realworldqa


def get_eval_chartqa():
    eval_dataset = load_dataset("lmms-lab/ChartQA").select_columns(
        ["question", "answer", "type", "image", "question"]
    )["test"]

    def eval_chartqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from lmms_eval.tasks.chartqa.utils import chartqa_process_results

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            predictions = [predictions[0].rstrip(".")]
            return chartqa_process_results(doc, predictions)

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            return 100 * sum([x["relaxed_overall"] for x in results]) / len(results)

        return lmms_evaluate(predictions, eval_dataset, process_results, aggregate_results)

    return eval_dataset, eval_chartqa


def get_eval_mme():
    eval_dataset = load_dataset("lmms-lab/MME").select_columns(
        ["answer", "category", "question_id", "image", "question"]
    )["test"]

    def eval_mme(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from lmms_eval.tasks.mme.utils import mme_aggregate_results, mme_process_results

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            score = mme_process_results(doc, predictions)
            if "mme_perception_score" in score:
                return score["mme_perception_score"]
            return score["mme_cognition_score"]

        return lmms_evaluate(predictions, eval_dataset, process_results, mme_aggregate_results)

    return eval_dataset, eval_mme


def get_eval_ocrbench():
    eval_dataset = load_dataset("echo840/OCRBench").select_columns(
        ["answer", "dataset", "question_type", "image", "question"]
    )["test"]

    def eval_ocrbench(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from types import SimpleNamespace

        from lmms_eval.tasks.ocrbench.utils import (
            OCRBench_score,
            ocrbench_aggregate_accuracy,
            ocrbench_process_results,
        )

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            predictions = [predictions[0].rstrip(".")]
            return ocrbench_process_results(doc, predictions)["ocrbench_accuracy"]

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            args = SimpleNamespace(output_path="tmp.out")
            # reset OCRBench_score as it is stored as a global
            # variable in lmms_eval
            for key in OCRBench_score:
                OCRBench_score[key] = 0
            return ocrbench_aggregate_accuracy(results, args) * 100

        return lmms_evaluate(predictions, eval_dataset, process_results, aggregate_results)

    return eval_dataset, eval_ocrbench


def get_eval_ai2d():
    eval_dataset = load_dataset("lmms-lab/ai2d").select_columns(
        ["answer", "options", "image", "question"]
    )["test"]

    def eval_ai2d(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        from lmms_eval.tasks.ai2d.utils import MultiChoiceRegexFilter, ai2d_doc_to_target

        ai2d_filter = MultiChoiceRegexFilter(
            group_select=0,
            ignore_case=True,
            ignore_punctuation=True,
            regex_pattern="([A-Z])\\.",
        )

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            predictions = [re.sub("^Answer: ", "", predictions[0])]
            target = ai2d_doc_to_target(doc, "mcq")  # "qa" for qwen-vl
            pred = ai2d_filter.apply(predictions, [doc])
            return {"exact_match": 1.0 if target == pred[0] else 0.0}

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            return 100 * sum([r["exact_match"] for r in results]) / len(results)

        return lmms_evaluate(predictions, eval_dataset, process_results, aggregate_results)

    return eval_dataset, eval_ai2d


def get_eval_docvqa():
    eval_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA").select_columns(
        ["answers", "questionId", "image", "question", "question"]
    )["validation"]

    def eval_docvqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        return anls_evaluate(predictions, eval_dataset)["anls_star"]

    return eval_dataset, eval_docvqa


def get_eval_infographic_vqa():
    eval_dataset = load_dataset("lmms-lab/DocVQA", "InfographicVQA").select_columns(
        ["answers", "questionId", "image", "question"]
    )["validation"]

    def eval_infographic_vqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset
        return anls_evaluate(predictions, eval_dataset)["anls_star"]

    return eval_dataset, eval_infographic_vqa


def get_eval_gqa():
    # GQA ships questions and images in separate configs; join them by image id.
    images = load_dataset("lmms-lab/GQA", "testdev_balanced_images")["testdev"]
    id2image = {row["id"]: row["image"] for row in images}
    eval_dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions")["testdev"]
    eval_dataset = eval_dataset.map(lambda doc: {"image": id2image[doc["imageId"]]}).select_columns(
        ["answer", "question", "image"]
    )

    def eval_gqa(predictions: dict[str, str]) -> float:
        nonlocal eval_dataset

        def normalize(text: str) -> str:
            # Matches lmms-eval's exact_match (ignore_case + ignore_punctuation).
            return text.strip().lower().translate(str.maketrans("", "", string.punctuation))

        def process_results(doc: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
            match = normalize(predictions[0]) == normalize(doc["answer"])
            return {"exact_match": 1.0 if match else 0.0}

        def aggregate_results(results: list[dict[str, Any]]) -> float:
            return 100 * sum([r["exact_match"] for r in results]) / len(results)

        return lmms_evaluate(predictions, eval_dataset, process_results, aggregate_results)

    return eval_dataset, eval_gqa


def format_question(
    dataset_name: str,
    question: str,
    options: list[str] | None = None,
) -> str:
    """Build the prompt for a benchmark question.

    The wording is part of each benchmark's protocol: the answer parsers in
    :func:`load_eval_fn` only read a letter because the multiple-choice prompt lettered the
    options. These are the exact strings the reported results were produced under, so
    changing one invalidates that benchmark's numbers; ``tests/test_eval_prompts.py`` pins
    them.
    """
    # AI2D
    if dataset_name == "ai2d":
        assert options is not None
        if question.strip()[-1] not in {"?", "."}:
            question += "?"
        question = f"Question: {question}\nChoices:\n"

        opts = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        suffix = "\nAnswer with the letter."
        question = f"{question}{opts}{suffix}"
    # MME's questions already end in their own "Please answer yes or no."; the generic
    # answer-format instruction is appended on top of it.
    if dataset_name == "mme":
        question = f"{question.strip()}\nAnswer the question using a single word or phrase."
    # Other datasets
    if dataset_name in [
        "chartqa",
        "docvqa",
        "infographic_vqa",
        "textvqa",
        "realworldqa",
        "ocrbench",
        "gqa",
    ]:
        pattern = (
            "Answer the question using a single word or phrase."
            if dataset_name != "ocrbench"
            else "Please directly answer the question."
        )
        question = f"{question}\n{pattern}"
    return question


def load_model(model_id: str, image_size: int = 896, device: str = "cuda"):
    """Load a released model and its processor from the Hugging Face hub.

    :param model_id: released model name (e.g. "CASA-Qwen2_5-VL-3B"), full hub repo id
        or a path to a local checkpoint directory
    :param image_size: resolution the processor resizes images to
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


class _EvalDataset(TorchDataset):
    """Wraps an HF eval dataset; image preprocessing runs in DataLoader workers."""

    def __init__(
        self, hf_dataset: Dataset, dataset_name: str, image_processor: Any, image_size: int
    ):
        self.ds = hf_dataset
        self.dataset_name = dataset_name
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        elt = self.ds[idx]
        # Resize + patchify: CPU-bound, parallelised across DataLoader workers
        pixel_value = self.image_processor.process_images(
            elt["image"].convert("RGB"), img_size=self.image_size
        ).squeeze(0)  # (h_patches, w_patches, 588)
        return {
            "idx": idx,
            "pixel_value": pixel_value,
            "question": format_question(
                self.dataset_name,
                elt["question"],
                elt.get("options", elt.get("choices")),
            ),
            "answer": elt.get("answer", ""),
        }


def _collate_fn(items: list[dict[str, Any]], processor: Any, add_system_prompt: bool) -> dict:
    """Tokenise text in the main process; images are already preprocessed."""
    conversations = [
        (
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                }
            ]
            if add_system_prompt
            else []
        )
        + [
            {
                "role": "user",
                "content": [
                    # image=None: skip image processing here, pixel_values injected below
                    {"type": "image", "image": None},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        for item in items
    ]
    inputs = processor.tokenize_messages(messages=conversations)
    # Replace the None placeholder with the tensors already processed by workers. The image
    # processor returns fp32, while the model runs in bf16, so cast here (in the workers)
    # rather than feeding a dtype the vision tower does not expect.
    inputs["pixel_values"] = [item["pixel_value"].to(torch.bfloat16) for item in items]
    inputs["indices"] = [item["idx"] for item in items]
    inputs["answers"] = [item["answer"] for item in items]
    return inputs


@torch.inference_mode()
def infer(
    model_id: str,
    dataset_name: Literal[
        "chartqa",
        "textvqa",
        "realworldqa",
        "ai2d",
        "mme",
        "ocrbench",
        "docvqa",
        "infographic_vqa",
        "gqa",
    ] = "chartqa",
    overwrite: bool = False,
    image_size: int = 896,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    num_workers: int = 8,
    verbose: bool = False,
):
    """Example usage:

    Single GPU:  uv run scripts/eval.py CASA-Helium1-VL-2B chartqa --batch_size 16
    Multi GPU:   uv run torchrun --nproc_per_node=8 scripts/eval.py CASA-Helium1-VL-2B chartqa --batch_size 16

    :param model_id: released model name (see RELEASED_MODELS), full hub repo id, or a
        path to a local checkpoint directory
    """

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    # Each rank loads its own model copy; silence HF loading logs off the main rank.
    if not is_main:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

    result_path = Path("result_logs") / f"{model_id.rstrip('/').split('/')[-1]}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if result_path.exists():
        with open(result_path, "r") as f:
            results = json.load(f)

    if dataset_name in results and not overwrite:
        if is_main:
            print(f"Overall accuracy (already computed): {results[dataset_name]:.2f}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    model, processor = load_model(model_id, image_size, device=device)
    if is_main:
        print(f"Evaluating {model_id} on dataset {dataset_name} across {world_size} GPU(s).")
    dataset, eval_fn = load_eval_fn(dataset_name)
    if is_main:
        print(f"Dataset has {len(dataset)} samples.")

    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    add_system_prompt = "qwen" in model_id.lower()
    eval_ds = _EvalDataset(dataset, dataset_name, processor.image_processor, image_size)
    # Each rank processes a distinct shard of the dataset. drop_last=False pads the
    # last shard with a few duplicate samples; harmless since predictions are keyed
    # by sample index and merged into a dict before evaluation.
    sampler = (
        DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    data_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(_collate_fn, processor=processor, add_system_prompt=add_system_prompt),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    predictions: dict[str, str] = {}
    pbar = tqdm(data_loader, ncols=80, disable=not is_main)
    prev_milestone = 0

    for batch in pbar:
        indices = batch.pop("indices")
        answers = batch.pop("answers")
        inputs = batch.to(model.device)
        input_len = inputs["input_ids"].shape[1]
        output_ids = model.generate_from_image(**inputs, max_new_tokens=max_new_tokens)[
            :, input_len:
        ]
        responses = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for i, bidx in enumerate(indices):
            predictions[str(bidx)] = responses[i]
            if verbose and is_main:
                ic(responses[i], answers[i])

        # Running accuracy is a single-GPU progress indicator only. In distributed
        # mode the periodic full-dataset eval_fn() makes rank 0 far slower than the
        # others, skewing arrival at the all_gather barrier — so skip it entirely.
        milestone = len(predictions) // 100
        if world_size == 1 and milestone > prev_milestone and dataset_name != "mme":
            acc = eval_fn(predictions)
            pbar.set_postfix(last_running_acc=f"{acc:.2f}")
            prev_milestone = milestone

    # Gather every rank's predictions onto rank 0 for the final evaluation.
    if world_size > 1:
        gathered: list[dict[str, str] | None] = [None] * world_size
        dist.all_gather_object(gathered, predictions)
        if is_main:
            predictions = {}
            for shard in gathered:
                predictions.update(shard)  # type: ignore[arg-type]
        dist.destroy_process_group()

    if not is_main:
        return

    acc = eval_fn(predictions)
    print(f"Overall accuracy: {acc:.2f}")

    results = {}
    if result_path.exists():
        with open(result_path, "r") as f:
            results = json.load(f)

    if dataset_name not in results or overwrite:
        results[dataset_name] = acc
        results = dict(sorted(results.items()))
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    Fire(infer)
