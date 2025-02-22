import torch
import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader

class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, landmarks_dir, json_map_path, transform=None, max_landmarks=100):
        """
        Args:
            csv_path (str): Path to train.csv
            landmarks_dir (str): Directory containing all .parquet files
            json_map_path (str): Path to sign_to_prediction_index_map.json
            transform: Optional transforms for data augmentation
            max_landmarks (int): Max number of landmarks to keep for memory optimization
        """
        self.data = pd.read_csv(csv_path)
        self.landmarks_dir = landmarks_dir
        self.transform = transform
        self.max_landmarks = max_landmarks  # Limit number of landmarks for efficiency

        # Load label mapping
        with open(json_map_path, "r") as f:
            self.label_map = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.abspath(os.path.join(self.landmarks_dir, row["path"]))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_parquet(file_path)
        df.fillna(0, inplace=True)  

        # Extract X, Y, Z coordinates and limit landmarks
        points = df[['x', 'y', 'z']].iloc[:self.max_landmarks].values
        points = np.nan_to_num(points)
        points = (points - points.min()) / (points.max() - points.min() + 1e-6)  # Min-Max Scaling

        # Ensure at least 21 landmarks exist
        if points.shape[0] < 21:
            padded = np.zeros((21, 3))
            padded[: points.shape[0], :] = points
            points = padded

        # CNN Input: Shape (1, max_landmarks, 3) to match grayscale format
        cnn_input = torch.tensor(points.T, dtype=torch.float32).unsqueeze(0)

        # Transformer sequence features
        seq_features = self.extract_transformer_features(points)

        # Get label
        label = torch.tensor(self.label_map[row["sign"]], dtype=torch.long)

        return cnn_input, seq_features, label

    def extract_transformer_features(self, points):
        """
        Extracts features for transformer input:
        - Motion (Future & History)
        - Pairwise Distances
        """
        motion_future = np.diff(points, axis=0, prepend=points[0:1])
        motion_history = np.diff(points, axis=0, append=points[-1:])
        pairwise_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

        # Ensure all features are non-negative
        motion_future = np.abs(motion_future)
        motion_history = np.abs(motion_history)
        pairwise_distances = np.abs(pairwise_distances)

        seq_features = np.concatenate([
            motion_future.flatten(),
            motion_history.flatten(),
            pairwise_distances.flatten()
        ])

        return torch.tensor(seq_features, dtype=torch.float32)

def get_dataloader(csv_path, landmarks_dir, json_map_path, batch_size=32, shuffle=True):
    dataset = SignLanguageDataset(csv_path, landmarks_dir, json_map_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
