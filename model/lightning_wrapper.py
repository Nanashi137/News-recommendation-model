import pytorch_lightning as pl 

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np

from sklearn.metrics import roc_auc_score, ndcg_score

from pytorch_lightning.utilities.grads import grad_norm


class NewsRecommender(pl.LightningModule): 
    def __init__(self, 
                 news_encoder: nn.Module, user_encoder: nn.Module, click_predictor: nn.Module, 
                 pretrained_paths=None,
                 news_encoder_lr: float=3e-6, user_encoder_lr: float=3e-4):
        super().__init__()


        # News encoder
        self.news_encoder = news_encoder
        self.lr_news = float(news_encoder_lr)

        # User encoder
        self.user_encoder = user_encoder
        self.lr_user = float(user_encoder_lr)

        # Click predictor 
        self.click_predictor = click_predictor 

        if pretrained_paths: 
            self._load_pretrained_weights(paths=pretrained_paths)

        # Loss function 
        self.loss_fn = torch.nn.CrossEntropyLoss(reduce="mean")


    def _load_pretrained_weights(self, paths):
            if "news_encoder" in paths:
                self.news_encoder.load_state_dict(torch.load(paths["news_encoder"]))
            if "user_encoder" in paths:
                self.user_encoder.load_state_dict(torch.load(paths["user_encoder"]))
            if "click_predictor" in paths:
                self.click_predictor.load_state_dict(torch.load(paths["click_predictor"]))


    def forward(self, clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask,):
        clicked_news_embeddings = self.news_encoder(input_ids=clicked_news_input_ids, attention_mask=clicked_news_attention_mask)
        user_embedding = self.user_encoder(user_history=clicked_news_embeddings, history_mask=history_mask)
        candidate_news_embeddings = self.news_encoder(input_ids=candidate_news_input_ids, attention_mask=candidate_news_attention_mask)

        return self.click_predictor(user_embedding, candidate_news_embeddings)
    
    def training_step(self, batch, batch_idx):
        clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask, labels = batch.values()
        logits = self.forward(clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch=batch, step="val")
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch=batch, step="test")
        self.log_dict({"test_loss": loss})
    
    def _compute_mrr(self, y_pred, y_true):
        # Convert to NumPy
        y_pred = y_pred.cpu().numpy()  # Shape: (batch_size, num_candidates)
        y_true = y_true.cpu().numpy()  # Shape: (batch_size,)

        ranks = []
        for i in range(y_pred.shape[0]):  
            sorted_indices = np.argsort(y_pred[i])[::-1]  
            rank = np.where(sorted_indices == y_true[i])[0][0] + 1  
            ranks.append(1.0 / rank)

        return np.mean(ranks)

    def _compute_metrics(self, y_pred, y_true, step):
        # Convert tensors to NumPy arrays
        y_pred = F.softmax(y_pred, dim=1).cpu().numpy()  # Shape: (batch_size, num_candidates)
        y_true = y_true.cpu().numpy()  # Shape: (batch_size,)

        # Convert ground truth indices to one-hot encoding
        y_true_one_hot = np.zeros_like(y_pred)  # Shape: (batch_size, num_candidates)
        y_true_one_hot[np.arange(y_true.shape[0]), y_true] = 1 

        # Compute AUC
        auc = roc_auc_score(y_true_one_hot.flatten(), y_pred.flatten())

        # Compute NDCG@5
        ndcg_5 = ndcg_score(y_true_one_hot, y_pred, k=5)

        # Compute MRR
        mrr = self._compute_mrr(torch.tensor(y_pred), torch.tensor(y_true))

        # Log the metrics
        self.log(f"{step}_auc", auc, prog_bar=True, logger=True)
        self.log(f"{step}_ndcg@5", ndcg_5, prog_bar=True, logger=True)
        self.log(f"{step}_mrr", mrr, prog_bar=True, logger=True)

    def _shared_eval(self, batch, step):
        clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask, labels = batch.values()
        logits = self.forward(clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)

        # Compute and log metrics
        self._compute_metrics(y_pred=logits, y_true=labels, step=step)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.news_encoder.parameters(), "lr": self.lr_news},  # Lower LR for news encoder
            {"params": self.user_encoder.parameters(), "lr": self.lr_user},  # Higher LR for user encoder
        ])

        scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.8)

        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


