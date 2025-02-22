import torch
import pandas as pd
import numpy as np
import os
from model import SignLanguageModel  # Import trained model class
from dataset import SignLanguageDataset  # Import preprocessing logic

# Load label map
import json
with open("D:/Extra/sign_to_prediction_index_map.json", "r") as f:
    label_map = json.load(f)
    index_to_sign = {v: k for k, v in label_map.items()}  # Reverse mapping

# **1️⃣ Load the Trained Model**
def load_model(model_path="sign_language_model.pth", num_classes=250):
    """Loads the trained model onto CPU."""
    device = torch.device("cpu")  # Force model to run on CPU

    model = SignLanguageModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to CPU
    model.eval()  # Set model to evaluation mode
    
    return model, device


# **2️⃣ Prepare Input Data**
def preprocess_input(file_path, device):
    """Prepares the input data for CNN and Transformer."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")

    df = pd.read_parquet(file_path)
    df.fillna(0, inplace=True)  # Handle missing values

    # Extract X, Y, Z coordinates
    points = df[['x', 'y', 'z']].values
    points = np.nan_to_num(points)  # Replace NaNs
    points = (points - points.mean()) / (points.std() + 1e-8)  # Normalize

    # **Fix CNN Input (Ensure 1-channel grayscale for EfficientNet)**
    cnn_input = torch.tensor(points.T, dtype=torch.float32)  # (3, num_landmarks)
    cnn_input = cnn_input.unsqueeze(0)  # (1, 3, num_landmarks)
    cnn_input = cnn_input.unsqueeze(0)  # (1, 1, 3, num_landmarks) ✅ Convert to grayscale

    # Transformer-friendly sequence features
    seq_features = extract_transformer_features(points)

    # Ensure all tensors are on the correct device
    cnn_input = cnn_input.to(device)
    seq_features = seq_features.to(device)
    print("\n📊 **Feature Statistics Before Normalization:**")
    print(f"Min: {np.min(points)}, Max: {np.max(points)}, Mean: {np.mean(points)}, Std: {np.std(points)}")

    return cnn_input, seq_features.unsqueeze(0)


# **3️⃣ Extract Transformer Features**
def extract_transformer_features(points):
    """Extracts motion history, distances, and angles for Transformer input."""
    motion_future = np.diff(points, axis=0, prepend=points[0:1])
    motion_history = np.diff(points, axis=0, append=points[-1:])
    pairwise_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    # **Reduce feature size to avoid memory issues**
    motion_future = motion_future[:100].flatten()  # Limit to 100 points
    motion_history = motion_history[:100].flatten()
    pairwise_distances = pairwise_distances[:10, :10].flatten()  # Reduce matrix size

    seq_features = np.concatenate([motion_future, motion_history, pairwise_distances])

    # Ensure at least 768 features (Transformer requirement)
    if len(seq_features) < 768:
        seq_features = np.pad(seq_features, (0, 768 - len(seq_features)), mode='constant')

    return torch.tensor(seq_features, dtype=torch.float32)


# **4️⃣ Perform Inference**
def predict_sign(model, file_path, device):
    """Runs inference and predicts the sign."""
    cnn_input, seq_features = preprocess_input(file_path, device)

    # Debugging: Ensure correct shapes before passing to the model
    print(f"CNN Input Shape: {cnn_input.shape}")  # Should be (1, 1, 3, num_landmarks)
    print(f"Sequence Features Shape: {seq_features.shape}")  # Should be (1, 768)

    # 
    with torch.no_grad():
        output = model(cnn_input, seq_features)
    
    # Apply softmax to get class probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    top5_probs, top5_classes = torch.topk(probs, 5, dim=1)

    print("\n🔍 **Top 5 Predictions:**")
    for i, (idx, prob) in enumerate(zip(top5_classes[0], top5_probs[0])):
        print(f"{i+1}. {index_to_sign[idx.item()]} (Confidence: {prob.item()*100:.2f}%)")
    
    predicted_class = top5_classes[0][0].item()
    predicted_sign = index_to_sign.get(predicted_class, "Unknown")
    
    return predicted_sign



# **5️⃣ Run the Model on a Sample File**
if __name__ == "__main__":
    model, device = load_model()

    # Sample test file from train_landmark_files/
    test_file = "D:/Extra/train_landmark_files/18796/1002643353.parquet"  # Replace with real file

    prediction = predict_sign(model, test_file, device)
    print(f"Predicted Sign: {prediction}")
