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
try:
    from lavis.models.blip2_models.Qformer import BertSelfAttention
except:
    pass

def txt_highlight_mask_prev(tokenizer, txt_prompt, highlighted_idx_range):
    # Convert text to tokens
    tokens = tokenizer.tokenize(txt_prompt)

    # Initialize the mask
    mask = [0] * len(tokens)

    # Convert highlighted_idx_range to integer ranges
    ranges = []
    for idx_range in highlighted_idx_range:
        if isinstance(idx_range, str):
            # Add a space before the string to avoid partial matches
            if idx_range[0] != " ":
                idx_range = " " + idx_range
            start_idx = txt_prompt.find(idx_range)
            if start_idx == -1:
                continue  # Skip if the string is not found
            end_idx = start_idx + len(idx_range)
            ranges.append((start_idx, end_idx))
        elif isinstance(idx_range, list):
            ranges.append((idx_range[0], idx_range[1]))
    
    # Mark the highlighted ranges in the mask
    for start_idx, end_idx in ranges:
        start_token_idx = len(tokenizer.tokenize(txt_prompt[:start_idx]))
        end_token_idx = len(tokenizer.tokenize(txt_prompt[:end_idx]))

        # TODO[MAYBE] Include the start and end tokens that partially overlap with the highlighted area

        for i in range(start_token_idx, end_token_idx):
            mask[i] = 1
    print(tokens)
    return mask

def txt_highlight_mask(tokenizer, txt_prompt, highlighted_idx_range):
    # Convert text to tokens
    tokens = tokenizer.tokenize(txt_prompt)

    # Initialize the mask
    mask = [0] * len(tokens)

    # Convert highlighted_idx_range to integer ranges
    ranges = []
    for idx_range in highlighted_idx_range:
        if isinstance(idx_range, str):
            # Add a space before the string to avoid partial matches
            if idx_range[0] != " ":
                idx_range = " " + idx_range
            start_idx = txt_prompt.find(idx_range)
            if start_idx == -1:
                continue  # Skip if the string is not found
            end_idx = start_idx + len(idx_range)
            ranges.append((start_idx, end_idx))
        elif isinstance(idx_range, list) and len(idx_range) == 2:
            ranges.append((idx_range[0], idx_range[1]))

    # Mark the highlighted ranges in the mask
    for start_idx, end_idx in ranges:
        start_token_idx = len(tokenizer.tokenize(txt_prompt[:start_idx]))
        end_token_idx = len(tokenizer.tokenize(txt_prompt[:end_idx]))

        # TODO[MAYBE] Include the start and end tokens that partially overlap with the highlighted area

        for i in range(start_token_idx, end_token_idx):
            mask[i] = 1

    return mask, tokens

def qformer_modify_attention(model, highlight_mask, attention_weight=None):
    count = 0
    model.hl_mask = highlight_mask
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            count += 1
            if attention_weight is not None:
                module.attention_weight = math.log(attention_weight)
            # use a placeholder function for the original forward.
            module.ori_forward = types.MethodType(qformer_ori_forward, module)
            module.forward = types.MethodType(qformer_new_forward, module)
            module.cross_atten_vis = False
            
            if count <= 3:
                module.cross_atten_vis = True
                
            module.index = count
            module.set_highlight_mask = types.MethodType(qformer_set_highlight_mask, module)
            module.set_highlight_mask(highlight_mask)
    # print("Number of BertSelfAttention in the model:", count)
    
def qformer_reset_model(model):
    # delete the attribute hl_mask in the model.
    if hasattr(model, "hl_mask"):
        del model.hl_mask
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            module.forward = types.MethodType(qformer_ori_forward, module)
    # print("Reset model to normal")
    
def qformer_set_highlight_mask(self, highlight_mask=None):
    if highlight_mask is None:
        self.hl_mask = None
    else:
        self.hl_mask = highlight_mask.unsqueeze(0)

def qformer_ori_forward(self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
    # just a copy of the original forward
    return self.forward(hidden_states,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
    )

def qformer_new_forward(self,
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
        print("Set attention_weight to", self.attention_weight)

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
        
    if is_cross_attention:
        # print("attention_scores.shape:", attention_scores.shape, self.attention_head_size)
        # attn_weight rescale.
        if self.hl_mask is not None:
            attn_mask_cur = torch.ones_like(attention_scores).to(attention_scores.device)
            hl_mask = self.hl_mask.unsqueeze(0).unsqueeze(2).expand_as(attn_mask_cur)
            attn_mask_cur[hl_mask==1] *= self.attention_weight
            print("attn_mask_cur:", attn_mask_cur.shape, attn_mask_cur.sum())
            attention_scores[hl_mask==1] += self.attention_weight# attn_mask_cur
            # no need to update hl_mask here.
            
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


if __name__ == "__main__":
    txt_prompt = "Please help me to refine the following, just fix the grammar and keep the LaTeX format:\n\caption{[aMYr, 5k69] The availability of RIVAL with ControlNet. Two examples are given for each modality of the control condition~(Canny edge, segmentation map, pose annotations, and depth map). Exemplars are shown on the left of each image pair.}"
    txt_highlight_index = ["fix the grammar", "keep the LaTeX format"]
    tokenizer = AutoTokenizer.from_pretrained("models/LLaVA/pretrained-models/Llama-2-7b-chat-hf")
    highlighted_mask = txt_highlight_mask(tokenizer, txt_prompt, txt_highlight_index)
    print(highlighted_mask, sum(highlighted_mask))
    
    # load a llama model.
    model = AutoModelForCausalLM.from_pretrained("models/LLaVA/pretrained-models/Llama-2-7b-chat-hf")
    # inference with this model
    output = model.generate(tokenizer(txt_prompt, return_tensors="pt").input_ids.to("cuda"))
    