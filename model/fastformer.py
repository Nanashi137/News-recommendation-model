import torch 
import torch.nn as nn     

class MultiheadFastformerAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int=8) -> None:
        super(MultiheadFastformerAttention, self).__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads 
        self.head_dim = embedding_dim // num_heads


        self.query_transformation = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)
        self.key_transformation = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)
        self.value_transformation = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)
        self.final_transformation = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True)

        self.w_q = nn.Parameter(torch.randn(self.embedding_dim))
        self.w_k = nn.Parameter(torch.randn(self.embedding_dim))

        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, news_embeddings: torch.Tensor):
        batch_size, history_length, embedding_dim = news_embeddings.shape
        news_embeddings = self.layernorm(news_embeddings)

        # Query - Key - Value transformations
        query = self.query_transformation(news_embeddings).view(batch_size, history_length, self.num_heads, self.head_dim).transpose(1,2)
        key = self.key_transformation(news_embeddings).view(batch_size, history_length, self.num_heads, self.head_dim).transpose(1,2)
        value = self.value_transformation(news_embeddings).view(batch_size, history_length, self.num_heads, self.head_dim).transpose(1,2)

        # Query flow 
        query_weighted = torch.einsum("bijk,ik -> bij", query, self.w_q.view(self.num_heads, self.head_dim))
        alpha = torch.softmax(query_weighted,dim=2).unsqueeze(dim=-1)
        global_query = torch.sum(alpha*query, dim=2)

        # Key flow
        p = global_query.unsqueeze(dim=2)*key
        key_weighted = torch.einsum("bijk,ik -> bij", p, self.w_k.view(self.num_heads, self.head_dim))
        beta = torch.softmax(key_weighted, dim=2).unsqueeze(dim=-1)
        global_key = torch.sum(beta*p, dim=2)

        # Value flow
        u = (global_key.unsqueeze(dim=2)*value).contiguous().view(batch_size, history_length, self.embedding_dim)
        r = self.final_transformation(u)

        output = r + query.contiguous().view(batch_size, history_length, self.embedding_dim) # Mimicking residual connection 

        return output

# Debug
if __name__ == "__main__":
    batch_size = 10
    embed_dim = 256
    num_heads = 16
    length = 20

    news_embeddings = torch.randn(batch_size,length,embed_dim)

    fastformer = MultiheadFastformerAttention(embedding_dim=embed_dim, num_heads=num_heads)

    print(fastformer(news_embeddings).shape)