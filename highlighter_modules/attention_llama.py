# add import packages
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from torch.nn import CrossEntropyLoss

import types
    

# this file is for partial highlight helper functions.
def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ):

    # just a copy of the original forward
    return self.forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )

def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    attention_weight: float = 7.0,
    ):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # print(query_states.shape, key_states.shape, use_cache)
    if not hasattr(self, "attention_weight"):
        self.attention_weight = math.log(attention_weight)
        print("Set attention_weight to", self.attention_weight)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    ###############################################
    # ATTENTION ACTIVATION:
    if self.hl_mask is not None:
        attn_mask_cur = torch.ones_like(attn_weights).to(attn_weights.device)
        hl_mask = self.hl_mask.unsqueeze(0).unsqueeze(2).expand_as(attn_mask_cur)
        attn_mask_cur[hl_mask==1] += self.attention_weight
        bs = hl_mask.shape[0]
        # masked the last half of the sequence.
        if bs > 1:
            attn_mask_cur[bs//2:][hl_mask[bs//2:]==1] *= -1
        
        attn_weights += attn_mask_cur
        self.hl_mask = torch.cat((self.hl_mask, torch.zeros((self.num_heads, 1)).cuda()), dim=-1)
    ###############################################

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def set_highlight_mask(self, highlight_mask=None):
    if highlight_mask is None:
        self.hl_mask = None
    else:
        self.hl_mask = highlight_mask.unsqueeze(0).repeat(self.num_heads, 1)

def modify_llama_attention(self, highlight_mask, attention_weight=None):
    count = 0
    self.model.hl_mask = highlight_mask
    for module in self.model.modules():
        if isinstance(module, LlamaAttention):
            count += 1
            if count > 0:
                if attention_weight is not None:
                    module.attention_weight = math.log(attention_weight)
                # use a placeholder function for the original forward.
                module.ori_forward = types.MethodType(llama_attn_forward, module)
                module.forward = types.MethodType(llama_new_forward, module)
                module.set_highlight_mask = types.MethodType(set_highlight_mask, module)
                module.set_highlight_mask(highlight_mask)
                
def reset_llama_model(self):
    # delete the attribute hl_mask in the model.
    if hasattr(self.model, "hl_mask"):
        del self.model.hl_mask
    for module in self.model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = module.ori_forward
            
# inference modification functions.
def prepare_hl_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
            "masked_token_map": kwargs.get("masked_token_map", None),
            "attention_weight": kwargs.get("attention_weight", 1.0),
            "perturb_weight": kwargs.get("perturb_weight", 0.01),
        }
    )
    return model_inputs

def llama_hl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    return_dict: Optional[bool] = None,
    masked_token_map: Optional[torch.LongTensor] = None,
    attention_weight: float = .0,
    perturb_weight: float = 0.01,
):
    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images
        )
        
    if masked_token_map is not None and not hasattr(self.model, "hl_mask"):
    
        if torch.sum(masked_token_map) > 0:
            seq_len = inputs_embeds.shape[-2]
            masked_token_map = masked_token_map[:seq_len]
            
            self.modify_attention(masked_token_map, attention_weight)
            
            if inputs_embeds.shape[0] >= 1:
                inputs_embeds[-1][masked_token_map == 1] = inputs_embeds[-1][masked_token_map == 1]*perturb_weight

    return super(LlavaLlamaForCausalLM, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )


def llama_modify_inf(model):
    model.forward = types.MethodType(llama_hl_forward, model)
    model.prepare_inputs_for_generation = types.MethodType(prepare_hl_inputs_for_generation, model)
    model.modify_attention = types.MethodType(modify_llama_attention, model)
    model.reset_model = types.MethodType(reset_llama_model, model)