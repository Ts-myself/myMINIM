import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import os

from model import *
from utils import MedicalImageDataset

# parameters
T = 1000
beta_min = 1e-4
beta_max = 0.03
hidden_dim = 256
num_attention_heads = 8

# def train():
#     # load data
#     train_data_path = ''
#     train_dataset = OCTADataset()

#     model = MINIM(beta_min, beta_max, T, hidden_dim, num_attention_heads)
    

# Training script
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Load dataset
    train_dataset = MedicalImageDataset(root="../dataset/OCTA", split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MINIM(if_embed=False).to(device)

    # Load checkpoint if available
    checkpoint_path = "./train/checkpoint_epoch_10.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
    except FileNotFoundError:
        print("Checkpoint not found, starting from scratch.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for oct_batch, octa_batch in progress_bar:
            oct_batch = oct_batch.to(device)
            octa_batch = octa_batch.to(device)
            b = octa_batch.size(0)
            t = torch.randint(0, T, (b,), device=device).long()
            pred = model(oct_batch, None, t.float())
            loss = F.mse_loss(pred, octa_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"./train/checkpoint_epoch_{epoch+1+10}.pth")


if __name__ == "__main__":
    main()
