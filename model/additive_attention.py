import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, in_features: int, additive_dim:int) -> None:
        super(AdditiveAttention, self).__init__()
        
        self.in_features = in_features
        self.additive_dim = additive_dim
        self.projection = nn.Linear(in_features=in_features, out_features=additive_dim, bias=True)
        self.w_q = nn.Parameter(torch.randn(self.additive_dim))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor=None): 
        projected_matrix = self.projection(embeddings)
        scores = torch.matmul(torch.tanh(projected_matrix), self.w_q)

        MASKING_VALUE = -1e+30 if scores.dtype == torch.float32 else -1e+4
        scores = scores.masked_fill(mask==0, MASKING_VALUE) if mask is not None else scores
        attention_score  = torch.softmax(scores, dim=1)
        weighted_outputs = torch.einsum("b i, b i j -> b j", attention_score, embeddings)

        return weighted_outputs
    
# Debug
if __name__ == "__main__":
    projection_dim = 10
    embedding_dim = 3
    x = torch.randn(2, 5, embedding_dim)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    aa = AdditiveAttention(in_features=embedding_dim, additive_dim=projection_dim)

    attened_embedding = aa(x, mask)

    print(attened_embedding)



