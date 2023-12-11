#!/bin/bash

cfg=1.3
attn=3.0
perturb_weight=0.01
exp_name=llava-v1.5-13b-hl-$cfg-$attn-$perturb_weight

SPLIT="mmbench_dev_20230712"

python examples/eval_scripts/llava_vqa_mmbench_hl.py \
    --model-path /dataset/julianzhang/BenchVis/models/LLaVA/checkpoints/llava-v1.5-13b  \
    --question-file base_models/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file base_models/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/$exp_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --cfg $cfg \
    --attn $attn \
    --perturb_weight $perturb_weight \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python base_models/LLaVA/scripts/convert_mmbench_for_submission.py \
    --annotation-file base_models/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir base_models/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir base_models/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $exp_name

python examples/eval_scripts/eval_mmbench.py --filename base_models/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/$exp_name.jsonl
