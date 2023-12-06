import argparse
import torch
import os
import json
import cv2
import numpy as np
from icecream import ic
from tqdm import tqdm
import shortuuid

import sys

sys.path.append("./")
sys.path.append("./base_models")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

from transformers import AutoTokenizer, AutoModelForCausalLM
from highlighter_modules.guidance import ProbCFGLogitsProcessor
from highlighter_modules.utils import txt_highlight_mask
from highlighter_modules.attention_llama_llava import llama_modify_inf
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()

    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    qs = args.txt + "\n"

    prompt = "USER: " + qs + "ASSISTANT:"
    
    print("PROMPT:", prompt)
    
    qs_highlighted_parts = args.hl.split("<s>")
    if len(args.hl) == 0:
        qs_highlighted_parts = []
    highlighted_mask, tokens = txt_highlight_mask(
        tokenizer, prompt, qs_highlighted_parts
    )
    if sum(highlighted_mask) == 0:
        print("No highlighted tokens found, just use the original forward.")
    else:
        print("Highlighted {} tokens".format(sum(highlighted_mask)))
    highlighted_mask = [0] + highlighted_mask  # add a start token placeholder
    
    hl_mask = torch.tensor(highlighted_mask).float().cuda()
    # reshape to tensor
   
    
    embed = model.get_input_embeddings()(tokenizer(prompt, return_tensors="pt").input_ids.to("cuda"))
    hl_mask_ = hl_mask.unsqueeze(0).unsqueeze(2).cuda()
    hl_mask_ = hl_mask_.expand_as(embed)
    hl_mask_[hl_mask_==1] = args.perturb_weight
    hl_mask_[hl_mask_==0] = 1.0
    
    highlighted_embed = (embed*hl_mask_).half()
    cfg_batched_input = torch.cat([embed, highlighted_embed])
    llama_modify_inf(model)
    
    model.modify_attention(hl_mask, attention_weight=args.attn)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs_embeds = cfg_batched_input,
            max_length=1024,
            num_beams=args.num_beams,
            logits_processor=[
                ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)
            ],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    print(f"ASSISTANT:", outputs)
    model.reset_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/dataset/pretrained-models/vicuna-13b-v1.1")
    parser.add_argument(
        "--txt", type=str, default="Please give me a detailed plan to eat healthy and to lose weight."
    )
    parser.add_argument(
        "--hl", type=str, default="eat healthy"
    )
    # parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--attn", type=float, default=3.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)
    args = parser.parse_args()

    args.attention_weight = args.attn
    eval_model(args)
