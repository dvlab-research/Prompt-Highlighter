# file for highlight guidance.
# REF: CFG-LLM: Stay on topic with Classifier-Free Guidance
# https://arxiv.org/abs/2306.17806
from transformers.generation.logits_process import LogitsProcessor
import torch


class ProbCFGLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        use_log: bool = False,  # whether to use log softmax.
    ):
        self.guidance_scale = guidance_scale
        self.use_log = use_log

    def __call__(self, input_ids, scores):
        if self.use_log:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)

        bs = input_ids.shape[0] // 2
        cond_logits, uncond_logits = scores[:bs], scores[bs:]
        cond_logits = (
            self.guidance_scale * (cond_logits - uncond_logits) + uncond_logits
        )

        # directly copy two.
        logits = torch.cat([cond_logits, cond_logits], dim=0)
        return logits
