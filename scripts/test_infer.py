# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "pillow",
#     "einops>=0.8.1",
#     "transformers==4.51.3",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ]
# ///
"""Simple model test"""

import io
import os
import urllib.request

import fire
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main(repo: str) -> None:
    model = AutoModel.from_pretrained(
        repo,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        token=os.getenv("HUGGINGFACE_TOKEN", None),
    ).cuda()
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

    url = "https://ameroyer.github.io/images/thumbs/pub/moshivis.png"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Describe this image.",
                },
            ],
        },
    ]
    inputs = processor.tokenize_messages(messages=conversation)
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate_from_image(
        **inputs,
        max_new_tokens=512,
        pre_image_tokens=processor.pre_image_tokens,
        post_image_tokens=processor.post_image_tokens,
        eos_token_id=model.generation_config.eos_token_id,
    )[0, input_len:]
    response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
