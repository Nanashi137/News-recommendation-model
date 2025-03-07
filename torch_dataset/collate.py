import torch 


def CustomCollator(batch, tokenizer, max_title_length:int=30):

    clicked_news_texts = [item["clicked_news"] for item in batch]  # List of clicked news lists
    candidate_news_texts = [item["candidate_news"] for item in batch]  # List of candidate news lists
    history_masks = torch.stack([item["history_mask"] for item in batch], dim=0)  # Keep history masks
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)  # Labels tensor

    # Batch dimensions
    batch_size = len(clicked_news_texts)
    max_clicked = len(clicked_news_texts[0])  # Assuming consistent padding length
    num_candidates = len(candidate_news_texts[0])  # Assuming fixed num_candidates

    # Tokenizing Clicked News
    flattened_clicked_news = sum(clicked_news_texts, [])  # Flatten to (batch_size * max_clicked)
    clicked_news_tokens = tokenizer(
        flattened_clicked_news,
        padding="max_length",
        truncation=True,
        max_length=max_title_length,
        return_tensors="pt"
    )
    clicked_news_input_ids = clicked_news_tokens["input_ids"].view(batch_size, max_clicked, -1)
    clicked_news_attention_mask = clicked_news_tokens["attention_mask"].view(batch_size, max_clicked, -1)

    # Tokenizing Candidate News
    flattened_candidate_news = sum(candidate_news_texts, [])  # Flatten to (batch_size * num_candidates)
    candidate_news_tokens = tokenizer(
        flattened_candidate_news,
        padding="max_length",
        truncation=True,
        max_length=max_title_length,
        return_tensors="pt"
    )
    candidate_news_input_ids = candidate_news_tokens["input_ids"].view(batch_size, num_candidates, -1)
    candidate_news_attention_mask = candidate_news_tokens["attention_mask"].view(batch_size, num_candidates, -1)

    return {
        "clicked_news_input_ids": clicked_news_input_ids,  # (B, max_clicked_news, max_title_length)
        "clicked_news_attention_mask": clicked_news_attention_mask,  # (B, max_clicked_news, max_title_length)
        "history_mask": history_masks,  # (B, max_clicked_news)
        "candidate_news_input_ids": candidate_news_input_ids,  # (B, num_candidates, max_title_length)
        "candidate_news_attention_mask": candidate_news_attention_mask,  # (B, num_candidates, max_title_length)
        "labels": labels  # (B,)
    }