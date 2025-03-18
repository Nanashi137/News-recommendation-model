import torch
from torch.utils.data import Dataset, IterableDataset
import random
import polars as pl

class BaseRecommenderDataset:
    def __init__(self, behaviors_path, news_path, max_history_length: int=10, candidate_k: int=5) -> None:
        self.max_history_length = max_history_length
        self.candidate_k = candidate_k

        # Load news
        self.news_dict = self._load_news(news_path=news_path)

        # Load behaviors
        self.behaviors = self._load_behaviors(behaviors_path=behaviors_path)

        # Convert history to dictionary {ImpressionID: [clicked_news]}
        self.impresison_history = self._build_impression_history()

        # Parse impressions
        self.impressions = self._parse_impressions()

    def _load_news(self, news_path):
        news_df = pl.read_csv(
            source=news_path, separator="\t", has_header=False,
            new_columns=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"],
            schema_overrides={"NewsID": pl.String, "Category": pl.String, "Category": pl.String, "Title": pl.String, "Abstract": pl.String,
                              "URL": pl.String,"TitleEntities": pl.String, "AbstractEntities": pl.String},
            infer_schema=False
        )
        return dict(zip(news_df["NewsID"], news_df["Title"]))

    def _load_behaviors(self, behaviors_path):
        behaviors = pl.read_csv(
            source=behaviors_path, separator="\t", has_header=False,
            new_columns=["ImpressionID", "UserID", "Time", "History", "ImpLog"],
            schema_overrides={"ImpressionID": str, "UserID": str, "Time": str, "History": str, "ImpLog": str},
            infer_schema=False
        )

        return behaviors.filter(behaviors["History"].is_not_null() & behaviors["ImpLog"].is_not_null())

    def _build_impression_history(self):
        id_col = self.behaviors["ImpressionID"]
        history_col = self.behaviors["History"]
        impresison_history = {}
        for row in range(len(id_col)):
            imp_id = id_col[row]
            history = history_col[row]
            impresison_history[imp_id] = history.split()
        return impresison_history

    def _parse_impressions(self):
        parsed_impressions = []
        id_col = self.behaviors["ImpressionID"]
        implog_col = self.behaviors["ImpLog"]
        for row in range(len(id_col)):
            imp_id  = id_col[row]
            imp_log = implog_col[row]

            news_clicks = [pair.split("-") for pair in imp_log.split()]
            news_ids, click_labels = zip(*news_clicks)
            click_labels = list(map(int, click_labels))

            if 1 not in click_labels:
                continue  

            # Select 1 positive
            pos_indices = [i for i, label in enumerate(click_labels) if label == 1]
            pos_idx = random.choice(pos_indices)  
            positive_news = news_ids[pos_idx]

            # Select k-1 negatives
            negative_news = [news_ids[i] for i in range(len(news_ids)) if click_labels[i] == 0]
            if len(negative_news) < self.candidate_k - 1:
                continue 

            negative_news = random.sample(negative_news, self.candidate_k - 1)

            # Create shuffled candidate list
            candidate_news = negative_news + [positive_news]
            random.shuffle(candidate_news)
            label_index = candidate_news.index(positive_news)  # Get index of clicked news

            parsed_impressions.append((imp_id, candidate_news, label_index))

        return parsed_impressions

class NewsRecommenderDataset(Dataset, BaseRecommenderDataset):
    def __init__(self, behaviors_path, news_path, max_history_length: int=10, candidate_k: int=5) -> None:
        super().__init__(behaviors_path, news_path, max_history_length, candidate_k)

    def __len__(self):
        return len(self.impressions)

    def __getitem__(self, idx):
        imp_id, candidate_news, label_index = self.impressions[idx]

        # Get user's clicked history
        history = self.impresison_history[imp_id][-self.max_history_length:]  # Take last L clicks
        history_mask = torch.zeros(self.max_history_length)
        history_mask[:len(history)] = 1
        # Padding
        history = history + ["PAD"]*(self.max_history_length - len(history))
        

        # Convert NewsIDs to Titles
        clicked_news_titles = [self.news_dict.get(nid, "UNKNOWN") for nid in history]
        candidate_news_titles = [self.news_dict.get(nid, "UNKNOWN") for nid in candidate_news]

        return {
            "clicked_news": clicked_news_titles,   
            "history_mask": history_mask,
            "candidate_news": candidate_news_titles,  
            "labels": label_index  
        }
