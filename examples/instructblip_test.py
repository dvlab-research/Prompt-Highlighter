# JULIAN: this file is to construct a direct enhancement in Q-Former attention, lets go!
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

from highlighter_modules.guidance import ProbCFGLogitsProcessor
from highlighter_modules.attention_qformer import blip_modify_inf
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer
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

    base_model_path = "blip2_vicuna_instruct"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    with torch.device("cuda"):
        model, vis_processors, _ = load_model_and_preprocess(name=base_model_path, model_type=args.model_name, is_eval=True, device=device)
        
    # initialize llama tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    questions = json.load(open(args.question_file, "r"))['questions']
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
        cur_prompt = qs
        
        # TODO: hybrid highlighting.
        '''
        if "text_highlights" not in line:
            qs_highlighted_parts = []
        else:
            qs_highlighted_parts = [line["text_highlights"]]
        highlighted_mask, tokens  = txt_highlight_mask(tokenizer, cur_prompt, qs_highlighted_parts)
        highlighted_mask = [0] + highlighted_mask # add a start token placeholder
        '''
        
        mask_img_path = ""
        masked_img_token_map = [0] * 257
        mask_img = np.zeros((16, 16))
        if "mask" in line:
            mask_img_path = line["mask"]
            # convert this mask to a 1d index list that has 576(24x24) elements
            mask_img = cv2.imread(os.path.join(args.image_folder, mask_img_path), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (16, 16))
            mask_img_list = mask_img.flatten()
            for i in range(len(mask_img_list)):
                if mask_img_list[i] < 128:
                    masked_img_token_map[i+1] = 1
            mask_img = np.array(masked_img_token_map[1:]).reshape((16, 16))
        
        
        # change to long tensor:
        masked_img_token_map = torch.LongTensor(masked_img_token_map).cuda()
        
        image = np.array(Image.open(os.path.join(args.image_folder, image_file)))
        # change to RGB if grayscale or alpha channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        image = Image.fromarray(image)
        image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
        
        # convert Image to np array
        image = np.array(image)
        # if have alpha channel, remove it
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        # resize short edge to 224
        h, w = image.shape[:2]
        if w > h:
            image = cv2.resize(image, (int(w/h*224), 224))
        else:
            image = cv2.resize(image, (224, int(h/w*224)))
        
        # center crop
        image = image[(image.shape[0]-224)//2:(image.shape[0]+224)//2, (image.shape[1]-224)//2:(image.shape[1]+224)//2]
        
        #find BertSelfAttention and change the attention weight with cfg embeddings
        blip_modify_inf(model)
           
        image_tensor = image_tensor.repeat(2, 1, 1, 1)
            
        with torch.inference_mode():
            output = model.generate(
                {"image": image_tensor, "prompt": cur_prompt}, num_beams=args.num_beams, 
                logits_processor = [ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)], 
                perturb_weight=args.perturb_weight,
                attention_weight=args.attention_weight,
                masked_img_token_map=masked_img_token_map)
            
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": [output[0]],
                                   "answer_id": ans_id,
                                   "model_id": base_model_path,
                                   "metadata": {"mask_name": mask_img_path}}) + "\n")
        ans_file.flush()
        
        # finally, reset the model.
        model.reset_qformer_model()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/dataset/pretrained-models/vicuna-13b-v1.1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="assets/test_data/images")
    parser.add_argument(
        "--question-file", type=str, default="assets/test_data/questions.json"
    )
    parser.add_argument(
        "--answers-file", type=str, default="exp_out/instructblip_tests.jsonl"
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.3)
    parser.add_argument("--attn-weight", type=float, default=20.0)
    parser.add_argument("--model-name", type=str, default="vicuna13b")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--perturb_weight", type=float, default=0.01)
    args = parser.parse_args()
    args.attention_weight = args.attn_weight
    import sys
    # visible the last device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.device_count() - 1)
    eval_model(args)