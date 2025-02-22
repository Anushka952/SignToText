import torch
import argparse
from model import SignLanguageModel, train_model
from dataset import get_dataloader

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train Sign Language Model")
    parser.add_argument("--csv_path", default="D:/Extra/train.csv", help="Path to train.csv")
    parser.add_argument("--landmarks_dir", default="D:/Extra/", help="Path to landmark .parquet files")
    parser.add_argument("--json_map_path", default="D:/Extra/sign_to_prediction_index_map.json", help="Path to JSON label mapping")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=250, help="Number of sign language classes")
    args = parser.parse_args()

    # Load data
    train_loader = get_dataloader(args.csv_path, args.landmarks_dir, args.json_map_path, batch_size=args.batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageModel(num_classes=args.num_classes).to(device)

    # Train model
    train_model(model, train_loader, epochs=args.epochs, lr=args.learning_rate, device=device)

    # Save model
    torch.save(model.state_dict(), "sign_language_model.pth")
    print("Model training complete and saved!")

# **Required for Windows multiprocessing**
if __name__ == "__main__":
    main()
