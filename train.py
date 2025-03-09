from model.news_encoder import NewsEncoder
from model.user_encoder import UserEncoder 
from model.click_predictor import ClickPredictor
from model.lightning_wrapper import NewsRecommender

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging
)
import pytorch_lightning as pl

import torch
from torch.utils.data import random_split, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch_dataset.dataset import NewsRecommenderDataset
from torch_dataset.collate import CustomCollator

import os 
import argparse
import gc

from configs.config import load_config

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Model training arguments"
    )

    parser.add_argument(
        "--model_dir", type=str, nargs="?", help="Path to pre-trained language model folder", default="./plm/microsoft_MiniLM-L12-H384-uncased"
    )

    parser.add_argument(
        "--data_dir", type=str, nargs="?", help="Path MIND dataset root folder", default="./data"
    )

    parser.add_argument(
        "--config_path", type=str, nargs="?", help="Path training config file", default="./configs/train_config.yml"
    )

    return parser.parse_args()

def main(args): 
    # Training configs
    training_cfg_path = args.config_path
    training_cfg = load_config(conf_path=training_cfg_path)

    news_encoder_cfg = training_cfg['news_encoder']
    user_encoder_cfg = training_cfg['user_encoder']
    general_cfg = training_cfg['general']

    # Reproducibility
    seed = general_cfg['seed']

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    pl.seed_everything(seed, verbose=False)

    
    # Loading pre-trained language model
    model_dir = args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    encoder.train()
    encoder_cfg = AutoConfig.from_pretrained(model_dir)

    # Data paths
    root = args.data_dir
    train_root = os.path.join(root, "train")
    dev_root = os.path.join(root, "dev")

    # Load data
    print("Loading data...")
    full_train_dataset = NewsRecommenderDataset(behaviors_path=os.path.join(train_root, "behaviors.tsv"), 
                                        news_path=os.path.join(train_root, "news.tsv"), 
                                        max_history_length=general_cfg["max_history_length"], 
                                        candidate_k=general_cfg["candidate_k"])

    split_ratio = 0.1
    val_size = int(0.1*len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = NewsRecommenderDataset(behaviors_path=os.path.join(dev_root, "behaviors.tsv"), 
                                        news_path=os.path.join(dev_root, "news.tsv"), 
                                        max_history_length=general_cfg["max_history_length"], 
                                        candidate_k=general_cfg["candidate_k"])
    print("Data loaded!")

    # Dataloader 
    num_workers = general_cfg["num_workers"]
    batch_size = general_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn= lambda b:CustomCollator(batch=b, tokenizer=tokenizer), num_workers=num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn= lambda b:CustomCollator(batch=b, tokenizer=tokenizer), num_workers=num_workers, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn= lambda b:CustomCollator(batch=b, tokenizer=tokenizer), num_workers=num_workers, persistent_workers=False)

    # Load news recommendation model's modules
    news_encoder = NewsEncoder(encoder=encoder, encoder_cfg=encoder_cfg, embedding_dim=general_cfg['embedding_dim'], additive_dim=news_encoder_cfg['projection_dim'],)
    user_encoder = UserEncoder(embedding_dim=general_cfg["embedding_dim"], additive_dim=user_encoder_cfg["projection_dim"],
                            max_history_length=general_cfg['max_history_length'], dropout=0.2, num_layers=user_encoder_cfg["n_layers"])
    click_predictor = ClickPredictor()

    # News recommendation model
    news_recommender = NewsRecommender(news_encoder=news_encoder, user_encoder=user_encoder, click_predictor=click_predictor,
                                   news_encoder_lr=news_encoder_cfg["learning_rate"], user_encoder_lr=user_encoder_cfg["learning_rate"])
    
    # Initialize trainer 
    training_callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=training_cfg['general']['early_stopping']),
            StochasticWeightAveraging(swa_lrs=1e-2),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=training_cfg['checkpoint']['save_dir'],
                save_top_k=training_cfg['checkpoint']['topk_model'],
                monitor="val_loss",
                filename="{epoch:02d}-{val_loss:.4f}",
                save_last=True,
                mode="min",
            ),
            ModelSummary(-1)    
        ]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=training_cfg['checkpoint']['save_dir'])
    trainer = pl.Trainer(max_epochs=training_cfg['general']['max_epochs'],
                        callbacks=training_callbacks,
                        log_every_n_steps=training_cfg['logging']['step'], 
                        logger=tb_logger,
                        accelerator="gpu",
                        devices=1)

    # Training
    print("Trainning...")
    torch.cuda.empty_cache() 
    gc.collect()
    trainer.fit(
        model=news_recommender,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path= None)
    print("Training completed!")

    # Testing
    print("Testing...")
    trainer.test(model=news_recommender, 
            dataloaders=test_loader, 
            ckpt_path=None)
    print("Testing completed!")

    print("Completed!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)