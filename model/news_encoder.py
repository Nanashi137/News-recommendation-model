import os 

import torch
import torch.nn as nn

from .additive_attention import AdditiveAttention

from transformers import AutoTokenizer, AutoModel, AutoConfig

class NewsEncoder(nn.Module):
    def __init__(self, encoder, encoder_cfg,embedding_dim: int, additive_dim: int) -> None:
        super(NewsEncoder, self).__init__()

        # Base encoder layer
        self.base_encoder = encoder
        self.base_encoder_config = encoder_cfg

        self.base_encoder_hidden_dim = self.base_encoder_config.hidden_size
        self.embedding_dim = embedding_dim

        self.projection = nn.Linear(in_features=self.base_encoder_hidden_dim, out_features=self.embedding_dim)
        # Additive attention block
        
        self.additive_attention_dim = additive_dim
        self.additive_attention = AdditiveAttention(in_features=self.embedding_dim, additive_dim=self.additive_attention_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        batch_size, history_length, title_length = input_ids.shape 

        input_ids = input_ids.view(-1, title_length)
        attention_mask = attention_mask.view(-1, title_length)
        
        embeddings = self.projection(self.base_encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state)

        pooled_embeddings = self.additive_attention(embeddings, attention_mask)

        return pooled_embeddings.view(batch_size, history_length, -1)    


# Debug
if __name__ == "__main__":
    # Load model
    model_root = "./plm"
    model_save_dir = "minilm_l12_h384_uncased"

    model_dir = os.path.join(model_root, model_save_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    encoder_cfg = AutoConfig.from_pretrained(model_dir)

    batch_size = 5
    max_title_length = 30
    embedding_dim=256
    additive_dim=200
    max_history_length = 40

    test_size = (1, 5, 10)
    clicked_news = torch.randint(low=0, high=100, size=test_size)
    attention_mask = torch.randint(low=0, high=1, size=test_size)
    news_encoder = NewsEncoder(encoder=encoder, encoder_cfg=encoder_cfg, embedding_dim=embedding_dim, additive_dim=additive_dim)

    print(news_encoder(input_ids=clicked_news, attention_mask=attention_mask).shape)

