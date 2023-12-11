#!/bin/bash

cfg=1.3
attn=2.0
perturb_weight=0.01
exp_name=llava-v1.5-13b-hl-$cfg-$attn-$perturb_weight

python examples/eval_scripts/llava_vqa_loader_hl.py \
    --model-path /dataset/julianzhang/BenchVis/models/LLaVA/checkpoints/llava-v1.5-13b \
    --question-file base_models/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder base_models/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file base_models/LLaVA/playground/data/eval/MME/answers/$exp_name.jsonl \
    --temperature 0 \
    --cfg $cfg \
    --attn $attn \
    --perturb_weight $perturb_weight \
    --conv-mode vicuna_v1

cd base_models/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $exp_name

cd eval_tool

python calculation.py --results_dir answers/$exp_name > answers/$exp_name/eval.log

