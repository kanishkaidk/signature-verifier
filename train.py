# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.model.model_utils import SiameseNetwork
from backend.model.data_utils import UnifiedSignatureDataset, SignaturePairDataset, transform


# Config
DATA_ROOT = "data/"  # path to dataset root
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
CHECKPOINT_PATH = "saved_models/siamese_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
signature_dataset = UnifiedSignatureDataset(DATA_ROOT, transform=transform)
pair_dataset = SignaturePairDataset(signature_dataset, transform=transform)
train_loader = DataLoader(pair_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = SiameseNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
print("🛠️ Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"📉 Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"✅ Model saved to {CHECKPOINT_PATH}")
