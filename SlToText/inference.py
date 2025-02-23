import torch
import pandas as pd
import numpy as np
import json
from model import SignLanguageModel

# Paths to necessary files
MODEL_PATH = "sign_language_model_full_weights.pth"  # Ensure it's the latest trained model
JSON_MAP_PATH = "D:/Extra/sign_to_prediction_index_map.json"
SAMPLE_LANDMARK_FILE = "D:/Extra/train_landmark_files/49445/1000397667.parquet"  # Change to test sample

# Load label mapping
with open(JSON_MAP_PATH, "r") as f:
    label_map = json.load(f)
index_to_sign = {v: k for k, v in label_map.items()}

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(num_classes=len(index_to_sign)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# Function to preprocess landmarks
def preprocess_landmarks(landmark_file, max_landmarks=150):
    df = pd.read_parquet(landmark_file).fillna(0)  # Handle missing values
    if df.empty:
        print("⚠️ Warning: Empty input file.")
        return None

    points = df[['x', 'y', 'z']].values[:max_landmarks]
    
    if points.shape[0] < max_landmarks:
        padding = np.zeros((max_landmarks - points.shape[0], 3))  # Zero-padding
        points = np.vstack((points, padding))
    
    # Normalize using mean & std
    mean, std = points.mean(axis=0), points.std(axis=0)
    points = (points - mean) / (std + 1e-6)  # Avoid division by zero
    
    seq_features = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return seq_features

# Function to predict sign (Top 5 Predictions)
def predict_sign(landmark_file):
    seq_features = preprocess_landmarks(landmark_file)
    if seq_features is None:
        return "Unknown Sign (Empty Input)"

    seq_features = seq_features.to(device)
    
    with torch.no_grad():
        output = model(seq_features)
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get top 5 predictions
        top5_probabilities, top5_indices = torch.topk(probabilities, 20, dim=1)
        top5_probabilities = top5_probabilities.squeeze().tolist()
        top5_indices = top5_indices.squeeze().tolist()

    # print(f"🔢 Raw model output: {output.cpu().numpy()}")
    # print(f"📊 Predicted Probabilities: {probabilities.cpu().numpy()}\n")
    
    # Map indices to sign labels
    top5_predictions = [
        (index_to_sign.get(idx, f"Unknown-{idx}"), prob)
        for idx, prob in zip(top5_indices, top5_probabilities)
    ]

    # Display results
    print("🔝 Top 5 Predictions:")
    for rank, (sign, prob) in enumerate(top5_predictions, 1):
        print(f"{rank}. {sign}: {prob:.4f}")

    return top5_predictions[0][0]  # Return the top predicted sign

# Run prediction on a sample
if __name__ == "__main__":
    predicted_sign = predict_sign(SAMPLE_LANDMARK_FILE)
    print(f"\n🔮 Predicted Sign: {predicted_sign}")



    
