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
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from highlighter_modules.guidance import ProbCFGLogitsProcessor
from highlighter_modules.utils import txt_highlight_mask
from highlighter_modules.attention_llama_llava import llava_modify_inf
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

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = json.load(open(args.question_file, "r"))["questions"]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file).replace(
        ".jsonl", f"-{args.cfg}-{args.attention_weight}-{args.perturb_weight}.jsonl"
    )
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        if "text_highlights" not in line:
            line["text_highlights"] = []

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        qs_highlighted_parts = [line["text_highlights"]]
        highlighted_mask, tokens = txt_highlight_mask(
            tokenizer, prompt, qs_highlighted_parts
        )
        
        highlighted_mask = [0] + highlighted_mask  # add a start token placeholder

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # in the description generation task, we need to mask all image tokens.
        masked_img_token_map = [0] * args.patch_num * args.patch_num
        mask_img = np.zeros((args.patch_num, args.patch_num))
        mask_img_path = None
        if "mask" in line:
            mask_img_path = line["mask"]
            # convert this mask to a 1d index list that has (args.patch_num*args.patch_num) elements
            mask_img = cv2.imread(os.path.join(args.image_folder, mask_img_path), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (args.patch_num, args.patch_num))
            mask_img_list = mask_img.flatten()
            for i in range(len(mask_img_list)):
                if mask_img_list[i] < 128:
                    masked_img_token_map[i] = 1
            mask_img = np.array(masked_img_token_map).reshape((args.patch_num, args.patch_num))
        
        # extend masked token map to the end of input_ids
        id_len = input_ids.shape[1]
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        image_token_start = image_token_indices[0]
        masked_token_map = (
            highlighted_mask[:image_token_start]
            + masked_img_token_map
            + highlighted_mask[image_token_start + 1 :]
        )

        # change to long tensor:
        masked_token_map = torch.LongTensor(masked_token_map).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # convert Image to np array
        image = np.array(image)
        # if have alpha channel, remove it
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # resize short edge to args.img_size
        h, w = image.shape[:2]
        if w > h:
            image = cv2.resize(image, (int(w / h * args.img_size), args.img_size))
        else:
            image = cv2.resize(image, (args.img_size, int(h / w * args.img_size)))

        # center crop
        image = image[
            (image.shape[0] - args.img_size) // 2 : (image.shape[0] + args.img_size) // 2,
            (image.shape[1] - args.img_size) // 2 : (image.shape[1] + args.img_size) // 2,
        ]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        image_tensor = image_tensor.unsqueeze(0)

        # use the batched cfg version for input_ids and images
        input_ids = input_ids.repeat(2, 1)
        image_tensor = image_tensor.repeat(2, 1, 1, 1)
        
        llava_modify_inf(model)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                masked_token_map=masked_token_map,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                attention_weight=args.attention_weight,
                max_new_tokens=1024,
                use_cache=True,
                perturb_weight=args.perturb_weight,
                logits_processor=[
                    ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)
                ],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {"mask_name": mask_img_path}
                }
            )
            + "\n"
        )
        ans_file.flush()

        # finally, reset the model.
        model.reset_model()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/dataset/yczhang/BenchVis/models/LLaVA/checkpoints/llava-v1.5-13b")
    
    # llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3")
    #llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="assets/test_data/images")
    parser.add_argument(
        "--question-file", type=str, default="assets/test_data/questions.json"
    )
    parser.add_argument(
        "--answers-file", type=str, default="exp_out/llava_tests.jsonl"
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--attn", type=float, default=5.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)
    args = parser.parse_args()
    if "llava-v1.5" in args.model_path or "336" in args.model_path:
        args.img_size = 336
        args.patch_num = 24
        
    else:
        args.img_size = 224
        args.patch_num = 16
        
    model_base = args.model_path.split("/")[-1]
    # set out dir as the folder of the answers file.
    args.out_dir = os.path.dirname(args.answers_file)
    args.answers_file = args.answers_file.replace(
        ".jsonl", f"-{model_base}-{args.perturb_weight}-{args.num_beams}.jsonl"
    )
    args.model_base = None
    args.attention_weight = args.attn

    eval_model(args)
