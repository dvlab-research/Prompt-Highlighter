# this file is for partial highlight helper functions.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
from typing import Optional, Tuple, List
import types
import matplotlib.pyplot as plt
import torch.nn as nn

# import attention modules.
from transformers.models.llama.modeling_llama import LlamaAttention

# if found lavis, then import
from lavis.models.blip2_models.Qformer import BertSelfAttention

def modify_qformer_attention(model, highlight_mask, attention_weight=None):
    count = 0
    model.hl_mask = highlight_mask
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            count += 1
            if attention_weight is not None:
                module.attention_weight = math.log(attention_weight)
            # use a placeholder function for the original forward.
            module.ori_forward = types.MethodType(qformer_attn_forward, module)
            module.forward = types.MethodType(qformer_new_forward, module)
            module.cross_atten_vis = False

            if count <= 3:
                module.cross_atten_vis = True

            module.index = count
            module.set_highlight_mask = types.MethodType(
                qformer_set_highlight_mask, module
            )
            module.set_highlight_mask(highlight_mask)
    # print("Number of BertSelfAttention in the model:", count)


def reset_qformer_model(model):
    # delete the attribute hl_mask in the model.
    if hasattr(model, "hl_mask"):
        del model.hl_mask
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            module.forward = types.MethodType(qformer_attn_forward, module)


def qformer_set_highlight_mask(self, highlight_mask=None):
    if highlight_mask is None:
        self.hl_mask = None
    else:
        self.hl_mask = highlight_mask.float()

def qformer_attn_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    mixed_query_layer = self.query(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)

    past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    if (
        self.position_embedding_type == "relative_key"
        or self.position_embedding_type == "relative_key_query"
    ):
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(-1, 1)
        position_ids_r = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings - 1
        )
        positional_embedding = positional_embedding.to(
            dtype=query_layer.dtype
        )  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            relative_position_scores_key = torch.einsum(
                "bhrd,lrd->bhlr", key_layer, positional_embedding
            )
            attention_scores = (
                attention_scores
                + relative_position_scores_query
                + relative_position_scores_key
            )

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    if is_cross_attention and self.save_attention:
        self.save_attention_map(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs_dropped = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs_dropped = attention_probs_dropped * head_mask

    context_layer = torch.matmul(attention_probs_dropped, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (
        (context_layer, attention_probs) if output_attentions else (context_layer,)
    )

    outputs = outputs + (past_key_value,)
    return outputs


def qformer_new_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    attention_weight: float = 7.0,
):
    # The new forward function for q_former attention.
    is_cross_attention = encoder_hidden_states is not None
    if not hasattr(self, "attention_weight"):
        self.attention_weight = math.log(attention_weight)
        print(
            "No attention weight is given, set attention_weight to",
            self.attention_weight,
        )

    if is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    mixed_query_layer = self.query(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)

    past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    if (
        self.position_embedding_type == "relative_key"
        or self.position_embedding_type == "relative_key_query"
    ):
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(-1, 1)
        position_ids_r = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings - 1
        )
        positional_embedding = positional_embedding.to(
            dtype=query_layer.dtype
        )  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum(
                "bhld,lrd->bhlr", query_layer, positional_embedding
            )
            relative_position_scores_key = torch.einsum(
                "bhrd,lrd->bhlr", key_layer, positional_embedding
            )
            attention_scores = (
                attention_scores
                + relative_position_scores_query
                + relative_position_scores_key
            )

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    ###############################################
    # ATTENTION ACTIVATION:
    if is_cross_attention:
        # A relatively faster implementation.
        # change hl_mask to the same shape as attn_weights, change type as the same as attn_weights.
        hl_mask_pos = self.hl_mask*self.attention_weight
        hl_mask_neg = -1*self.hl_mask*(2+self.attention_weight)
        bs = attention_scores.shape[0]
        hl_mask_pos = hl_mask_pos.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand_as(attention_scores[:bs//2])
        hl_mask_neg = hl_mask_neg.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand_as(attention_scores[bs//2:])
        hl_mask = torch.cat((hl_mask_pos, hl_mask_neg), dim=0)
        
        attention_scores += hl_mask
        self.hl_mask = torch.cat((self.hl_mask, torch.zeros(1).cuda()), dim=-1)
    ###############################################

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    if is_cross_attention and self.save_attention:
        self.save_attention_map(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs_dropped = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs_dropped = attention_probs_dropped * head_mask

    context_layer = torch.matmul(attention_probs_dropped, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (
        (context_layer, attention_probs) if output_attentions else (context_layer,)
    )

    outputs = outputs + (past_key_value,)
    return outputs

def blip_hl_generate(
    self,
    samples,
    use_nucleus_sampling=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    length_penalty=1,
    num_captions=1,
    temperature=1,
    logits_processor=None,
    masked_img_token_map: Optional[torch.LongTensor] = None,
    attention_weight: float = 1.0,
    perturb_weight: float = 0.01,
):
    self.modify_qformer_attention(masked_img_token_map, attention_weight)
    self.llm_tokenizer.padding_side = "left"
    if "prompt" in samples.keys():
        prompt = samples["prompt"]
    else:
        prompt = self.prompt

    image = samples["image"]

    bs = image.size(0)

    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

    # For TextCaps
    if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
        prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

    query_tokens = self.query_tokens.expand(bs, -1, -1)
    if self.qformer_text_input:
        # remove ocr tokens in q_former (for eval textvqa)
        # qformer_prompt = prompt
        # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    image_embeds[-1][masked_img_token_map == 1]*= perturb_weight

    if self.qformer_text_input:
        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    else:
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

    inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
    atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

    llm_tokens = self.llm_tokenizer(
        prompt,
        padding="longest",
        return_tensors="pt"
    ).to(image.device)

    with self.maybe_autocast():
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            logits_processor=logits_processor
        )

    outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
    output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]

    return output_text

def blip_modify_inf(model):
    model.generate = types.MethodType(blip_hl_generate, model)
    model.modify_qformer_attention = types.MethodType(modify_qformer_attention, model)
    model.reset_qformer_model = types.MethodType(reset_qformer_model, model)
    