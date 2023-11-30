# Prompt Highlighter: Interactive Control for Multi-Modal LLMs

![logo](assets/logo.png)

Control text generation by highlighting our prompt! Prompt Highlighter is a training-free inference pipeline, which facilitates token-level user interactions for customized generation. Our method is compatible for both LLMs and VLMs.

![teaser](assets/teaser.png)

## Overview

- [Prompt Highlighter: Interactive Control for Multi-Modal LLMs](#prompt-highlighter-interactive-control-for-multi-modal-llms)
  - [Overview](#overview)
  - [MileStones](#milestones)
  - [Data Preparation](#data-preparation)
    - [Customized Inference](#customized-inference)
    - [Test Benchmarks](#test-benchmarks)
  - [Quick Start](#quick-start)
    - [Vicuna (LLaMA-based LLMs)](#vicuna-llama-based-llms)
    - [LLaVA](#llava)
    - [InstructBLIP](#instructblip)
    - [InternLMVLComposer](#internlmvlcomposer)
  - [Method](#method)
  - [Cite Prompt Highlighter](#cite-prompt-highlighter)
  - [Acknowledgement](#acknowledgement)

## MileStones

## Data Preparation

### Customized Inference

### Test Benchmarks

## Quick Start

```bash
conda create -n highlighter python=3.10 -y
conda activate highlighter
# you may also use your installed llava if you have installed.
cd base_models
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### Vicuna (LLaMA-based LLMs)

### LLaVA

### InstructBLIP

### InternLMVLComposer

TBD

## Method

<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/pipeline.png">
    <!-- /pypi-strip -->
    <img alt="pipeline" src="assets/pipeline.png" width="100%">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

An abstract pipeline of Prompt Highlighter. Users can control the focus of generation by marking out specific image regions or text spans. Then a token-level mask $\mathbf{m}$ is created to guide the language model's inference. Motivated by the classifier-free diffusion guidance, we form regular and unconditional context pairs based on highlighted tokens, demonstrating that the autoregressive generation in models can be guided in a classifier-free way. Notably, we find that, during inference, guiding the models with highlighted tokens through the attention weights leads to more desired outputs.

## Cite Prompt Highlighter

## Acknowledgement
