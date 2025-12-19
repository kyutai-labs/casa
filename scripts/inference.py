# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "rich>=12.6.0",
#     "anls>=0.0.2",
#     "anls-star>=0.0.12",
#     "lmms-eval @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git",
#     "datasets>=3.4.1",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Inference + evaluation with lmms lab"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Literal

import rich
import torch
from anls import anls_score
from anls_star import anls_score as anls_star_score
from datasets import Dataset, load_dataset
from fire import Fire
from tqdm.auto import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.processing_auto import AutoProcessor


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
        from lmms_eval.tasks.textvqa.utils import (
            textvqa_process_results,  # type: ignore
        )

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
        from lmms_eval.tasks.ai2d.utils import (
            MultiChoiceRegexFilter,
            ai2d_doc_to_target,
        )

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


def format_question(
    dataset_name: str,
    question: str,
    options: list[str] | None = None,
) -> str:
    # AI2D
    if dataset_name == "ai2d":
        assert options is not None
        if question.strip()[-1] not in {"?", "."}:
            question += "?"
        question = f"Question: {question}\nChoices:\n"

        opts = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        suffix = "\nAnswer with the letter."
        question = f"{question}{opts}{suffix}"
    # Other datasets
    if dataset_name in [
        "chartqa",
        "docvqa",
        "infographic_vqa",
        "textvqa",
        "realworldqa",
        "ocrbench",
    ]:
        pattern = (
            "Answer the question using a single word or phrase."
            if dataset_name != "ocrbench"
            else "Please directly answer the question."
        )
        question = f"{question}\n{pattern}"
    return question


def infer(
    model_id: Literal["CASA-Qwen-2_5-VL-3B", "CASA-Helium1-VL-2B", "Helium1-VL-2B"],
    dataset_name: Literal[
        "chartqa",
        "textvqa",
        "realworldqa",
        "ai2d",
        "mme",
        "ocrbench",
        "docvqa",
        "infographic_vqa",
    ] = "chartqa",
    save: bool = True,
    overwrite: bool = True,
    img_size: int = 896,
    max_new_tokens: int = 128,
    verbose: bool = False,
):
    """Example usage:

    :param model_id: Model to evaluate
    :param dataset_name: Dataset to evaluate
    :param save: If True, will save evaluation results in a json file
    :param overwrite: If False, skip eval for existing saved datasets
    :param device: Device to use
    :param img_size: Input image size
    :param max_new_tokens: Maximum number of token to generate
    :param verbose: Add verbosity for debugging
    """

    result_path = Path("logs") / f"{model_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if result_path.exists() and not overwrite:
        rich.print(
            "[yellow]Abort:[/yellow] Saved file already exists for"
            f" this dataset in {result_path}. If you want to continue,"
            " run with `--overwrite True`"
        )

    model_id = f"kyutai/{model_id}"
    print(f"Evaluating {model_id} on dataset {dataset_name}.")
    dataset, eval_fn = load_eval_fn(dataset_name)
    print(f"Dataset has {len(dataset)} samples.")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id, img_size=img_size, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    predictions = {}
    pbar = tqdm(enumerate(dataset), total=len(dataset), ncols=80)
    for idx, elt in pbar:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": elt["image"],
                    },
                    {
                        "type": "text",
                        "text": format_question(
                            dataset_name,
                            elt["question"],
                            elt.get("options", elt.get("choices", None)),
                        ),
                    },
                ],
            },
        ]
        if "qwen" in model_id.lower():
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                }
            ] + conversation
        inputs = processor.tokenize_messages(messages=conversation)
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]
        output_ids = model.generate_from_image(
            **inputs,
            max_new_tokens=max_new_tokens,
            pre_image_tokens=processor.pre_image_tokens,
            post_image_tokens=processor.post_image_tokens,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
        )[0, input_len:]
        response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        predictions[str(idx)] = response
        if idx % 100 == 0 and dataset_name != "mme":
            acc = eval_fn(predictions)
            pbar.set_postfix(running_acc=f"{acc:.2f}")
        if verbose:
            rich.print(f"[cyan]Prediction:[/cyan] {response}")

    acc = eval_fn(predictions)
    print(f"Overall accuracy: {acc:.2f}")

    if save:
        result_path = Path("logs") / f"{model_id}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

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
