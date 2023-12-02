# this file is for partial highlight helper functions.
# TODOs:
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import types
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

# import attention modules.
from transformers.models.llama.modeling_llama import LlamaAttention

# if found lavis, then import
from lavis.models.blip2_models.Qformer import BertSelfAttention


def qformer_modify_attention(model, highlight_mask, attention_weight=None):
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


def qformer_reset_model(model):
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
        self.hl_mask = highlight_mask.unsqueeze(0)

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
    # just a copy of the original forward
    return self.forward(
        hidden_states,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
    )


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
        # attn_weight rescale.
        if self.hl_mask is not None:
            hl_mask = self.hl_mask.unsqueeze(0).unsqueeze(2).expand_as(attention_scores).to(attention_scores.dtype)
            hl_mask *= self.attention_weight
            bs = hl_mask.shape[0]
            # deactivate the last half of the sequence.
            if bs > 1:
                hl_mask[bs//2:] *= -1
            # single forward, no need to update hl_mask here.
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
