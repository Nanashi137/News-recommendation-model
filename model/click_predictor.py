import torch 
import torch.nn as nn 

class ClickPredictor(nn.Module):
    def __init__(self) -> None:
        super(ClickPredictor, self).__init__()

    def forward(self,user_embedding: torch.Tensor, candidate_news_embeddings: torch.Tensor):

        logits = torch.einsum("b j, b i j -> b i", user_embedding, candidate_news_embeddings)

        return logits

# Debug
if __name__ == "__main__":

    batch_size = 2
    embedding_dim = 256
    n_candidates = 4

    user_embedding = torch.randn(batch_size, embedding_dim)
    candidate_news_embeddings = torch.randn(batch_size, n_candidates, embedding_dim)
    target = torch.randint(low=0, high=n_candidates-1, size=(batch_size, ))

    click_predictor = ClickPredictor()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(click_predictor(user_embedding=user_embedding, candidate_news_embeddings=candidate_news_embeddings),target)
    print(sum(p.numel() for p in click_predictor.parameters() if p.requires_grad))
    # print(target)
    # print(loss)
    