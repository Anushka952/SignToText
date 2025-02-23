import torch
import torch.optim as optim
import torch.nn as nn
from model import SignLanguageModel
from dataset import get_dataloader

def train_model(model, train_loader, epochs=10, lr=0.001, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for seq_features, labels in train_loader:
            seq_features, labels = seq_features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seq_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

def main():
    # Update paths here
    csv_path = "D:/Extra/train.csv"
    landmarks_dir = "D:/Extra/"
    json_map_path = "D:/Extra/sign_to_prediction_index_map.json"

    # Load Data
    train_loader = get_dataloader(csv_path, landmarks_dir, json_map_path, batch_size=32)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = SignLanguageModel(num_classes=250).to(device)

    # Train Model
    train_model(model, train_loader, epochs=10, device=device)

    # Save Full Model (Not just weights)
    torch.save(model, "sign_language_model_full.pth")
    print("Model saved as sign_language_model_full.pth")

if __name__ == "__main__":
    main()
