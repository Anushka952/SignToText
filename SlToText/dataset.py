import torch
import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader

class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, landmarks_dir, json_map_path, max_landmarks=150):
        self.data = pd.read_csv(csv_path)
        self.landmarks_dir = landmarks_dir
        self.max_landmarks = max_landmarks
        with open(json_map_path, "r") as f:
            self.label_map = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.landmarks_dir, row["path"])

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Landmark file not found: {file_path}")

        df = pd.read_parquet(file_path).fillna(0)
        points = df[['x', 'y', 'z']].values[:self.max_landmarks]
        points = (points - points.min()) / (points.max() - points.min() + 1e-6)

        seq_features = torch.tensor(points, dtype=torch.float32)
        label = torch.tensor(self.label_map.get(row["sign"], -1), dtype=torch.long)

        # Ensure valid labels
        if label.item() == -1:
            raise ValueError(f"Label {row['sign']} not found in JSON map!")

        return seq_features, label

def get_dataloader(csv_path, landmarks_dir, json_map_path, batch_size=32, num_workers=2):
    dataset = SignLanguageDataset(csv_path, landmarks_dir, json_map_path)

    # Set num_workers=0 for Windows to avoid multiprocessing issues
    if os.name == 'nt':
        num_workers = 0

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
