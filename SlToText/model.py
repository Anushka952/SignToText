import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from transformers import DebertaV2Model

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()

        # CNN-based EfficientNet for spatial features
        self.cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.cnn.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input for grayscale
        self.cnn.classifier[1] = nn.Linear(self.cnn.classifier[1].in_features, 256)

        # Transformer-based DeBERTa for temporal sequence modeling
        self.transformer = DebertaV2Model.from_pretrained("microsoft/deberta-v3-small")
        self.fc_transformer = nn.Linear(self.transformer.config.hidden_size, 256)

        # Fusion layer
        self.fc_final = nn.Linear(512, num_classes)

    def forward(self, cnn_features, seq_features):
        # CNN processing
        img_out = self.cnn(cnn_features)

        # Debugging Info
        #print("CNN Features shape:", cnn_features.shape)  # Expected: [batch_size, 1, 3, 100]
        #print("Sequence Features shape:", seq_features.shape)  # Expected: [batch_size, feature_dim]

        # Ensure valid indices for DeBERTa
        seq_features = seq_features.float()  # Convert to float
        batch_size, feature_dim = seq_features.shape

        # Ensure sequence length is compatible with DeBERTa (768 hidden dim)
        seq_len = feature_dim // 768  # Compute seq_len
        seq_features = seq_features[:, :seq_len * 768]  # Truncate extra
        seq_features = seq_features.view(batch_size, seq_len, 768)  # Reshape to [batch, seq_len, 768]

        # Pass through transformer
        seq_out = self.transformer(inputs_embeds=seq_features).last_hidden_state[:, 0, :]
        seq_out = self.fc_transformer(seq_out)

        # Combine CNN & Transformer outputs
        combined = torch.cat((img_out, seq_out), dim=1)
        output = self.fc_final(combined)
        return output

def train_model(model, train_loader, epochs=10, lr=0.001, device="cuda"):
    """
    Function to train the Sign Language Model
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for cnn_features, seq_features, labels in train_loader:
            cnn_features, seq_features, labels = (
                cnn_features.to(device, non_blocking=True),
                seq_features.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            outputs = model(cnn_features, seq_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    return model
