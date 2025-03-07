import torch
from torch.utils.data import Dataset
import random
import pandas as pd

class NewsRecommenderDataset(Dataset):
    def __init__(self, behaviors_path, news_path, max_history_length=10, candidate_k=5):
        """
          behaviors_path: Path to behaviors.tsv file.
          news_path: Path to news.tsv file (contains NewsID → Title mapping).
          max_history_length: Fixed length of clicked news history.
          candidate_k: Total number of candidate news (k = 1 positive + (k-1) negatives).
        """
        self.max_history_length = max_history_length
        self.candidate_k = candidate_k

        # Load news.tsv 
        self.news_dict = self._load_news(news_path)

        # Load behaviors.tsv
        self.behaviors = pd.read_csv(
            behaviors_path, sep="\t", header=None,
            names=["ImpressionID", "UserID", "Time", "History", "ImpLog"],
            dtype={"ImpressionID": str, "UserID": str, "Time": str, "History": str, "ImpLog": str}
        )

        # Convert history to dictionary {UserID: [clicked_news]}
        self.user_histories = self._build_user_history()

        # Parse impressions
        self.impressions = self._parse_impressions()

    def _load_news(self, news_path):
        """Loads news.tsv and returns a dictionary {NewsID: Title}."""
        news_df = pd.read_csv(
            news_path, sep="\t", header=None,
            names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"]
        )
        return dict(zip(news_df["NewsID"], news_df["Title"]))

    def _build_user_history(self):
        """Builds user history dictionary {UserID: [clicked_news]}."""
        user_histories = {}
        for _, row in self.behaviors.iterrows():
            user_id = row["UserID"]
            history = row["History"]
            if isinstance(history, str):
                user_histories.setdefault(user_id, []).extend(history.split())
        return user_histories

    def _parse_impressions(self):
        """Parses impressions and returns a list of (user_id, candidates, label_index)."""
        parsed_impressions = []
        for _, row in self.behaviors.iterrows():
            user_id = row["UserID"]
            imp_log = row["ImpLog"]

            # Parse ImpLog
            if not isinstance(imp_log, str):
                continue

            news_clicks = [pair.split("-") for pair in imp_log.split()]
            news_ids, click_labels = zip(*news_clicks)
            click_labels = list(map(int, click_labels))

            # Select a positive sample (random clicked news)
            if 1 not in click_labels:
                continue  # Skip if no clicked news

            pos_indices = [i for i, label in enumerate(click_labels) if label == 1]
            pos_idx = random.choice(pos_indices)  # Pick one clicked news
            positive_news = news_ids[pos_idx]

            # Select k-1 negatives
            negative_news = [news_ids[i] for i in range(len(news_ids)) if click_labels[i] == 0]
            if len(negative_news) < self.candidate_k - 1:
                continue  # Skip if not enough negatives

            negative_news = random.sample(negative_news, self.candidate_k - 1)

            # Create shuffled candidate list
            candidate_news = negative_news + [positive_news]
            random.shuffle(candidate_news)
            label_index = candidate_news.index(positive_news)  # Get index of clicked news

            parsed_impressions.append((user_id, candidate_news, label_index))

        return parsed_impressions

    def __len__(self):
        return len(self.impressions)

    def __getitem__(self, idx):
        """Returns a dictionary containing:
          - clicked_news_titles (fixed-length history of titles)
          - candidate_news_titles (shuffled list of candidate titles)
          - label (index of clicked news)
        """
        user_id, candidate_news, label_index = self.impressions[idx]

        # Get user's clicked history
        history = self.user_histories.get(user_id, [])[-self.max_history_length:]  # Take last L clicks
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