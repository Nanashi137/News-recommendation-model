import pytorch_lightning as pl 

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

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
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch=batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}
    
    def test_step(self, batch):
        loss = self._shared_eval(batch=batch)

        self.log_dict({"test_loss": loss})
    
    def _shared_eval(self, batch):
        clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask, labels = batch.values()
        logits = self.forward(clicked_news_input_ids, clicked_news_attention_mask, history_mask, candidate_news_input_ids, candidate_news_attention_mask)
        loss = self.loss_fn(logits, labels)

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


