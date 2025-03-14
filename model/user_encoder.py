import os 

import torch
import torch.nn as nn

from .fastformer import MultiheadFastformerAttention
from .additive_attention import AdditiveAttention

class UserEncoder(nn.Module):
    def __init__(self, max_history_length: int=512, *, embedding_dim: int, additive_dim:int, num_heads: int=16, num_layers: int=2, dropout: float=0.3) -> None:
        super(UserEncoder, self).__init__()

        self.max_history_length = max_history_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Fastfomer
        self.fastformer_layers = nn.ModuleList([
            nn.Sequential(MultiheadFastformerAttention(embedding_dim=self.embedding_dim,
                                                       num_heads=self.num_heads)
                        ) for _ in range(self.num_layers)])
        
        self.dropout = nn.Dropout(p=dropout)

        # Additive attention 
        self.additive_attention_dim = additive_dim
        self.additive_attention = AdditiveAttention(in_features=self.embedding_dim, additive_dim=self.additive_attention_dim)


    def forward(self, user_history: torch.Tensor, history_mask: torch.Tensor=None): 
        batch_size, history_length, embed_dim = user_history.shape
        history_length = min(history_length, self.max_history_length)

        x = user_history[:,:history_length,:]

        for layer in self.fastformer_layers: 
            x = layer(x)
            x = self.dropout(x)


        user_embedding = self.additive_attention(embeddings=x, mask=history_mask)

        return user_embedding

# Debug
if __name__ == "__main__":

    model_root = "./plm-nr"
    model_save_dir = "minilm_l12_h384_uncased"

    model_dir = os.path.join(model_root, model_save_dir)

    batch_size = 1
    embedding_dim = 256
    num_heads = 16
    length = 20
    num_layers = 2
    additive_dim=200
    max_history_length = 30

    user_history = torch.randn(batch_size, 5, embedding_dim)
    # history_mask = torch.ones(batch_size, 5)

    user_encoder = UserEncoder(embedding_dim=embedding_dim, additive_dim=additive_dim, num_heads=num_heads, num_layers=num_layers)

    print(user_encoder(user_history=user_history).shape)

