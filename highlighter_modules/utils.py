# utilization functions for the highlighter.

# return the mask and tokens for the highlighted text prompt.
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
                start_idx = txt_prompt.find(
                    idx_range[1:]
                )  # remove the space and try again
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
        # TODO: Include the start and end tokens that partially overlap with the highlighted area
        for i in range(start_token_idx, end_token_idx):
            mask[i] = 1

    return mask, tokens
