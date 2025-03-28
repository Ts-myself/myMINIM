import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import os

from model_wyt import *
from utils import MedicalImageDataset

# Define diffusion parameters
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


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
    model = MINIM().to(device)

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
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for oct_batch, octa_batch in progress_bar:
            oct_batch = oct_batch.to(device)
            octa_batch = octa_batch.to(device)
            b = octa_batch.size(0)

            # Sample timestep t
            # t = torch.randint(0, T, (b,), device=device).long()

            # Forward pass
            pred = model(oct_batch)

            # Loss
            loss = F.mse_loss(pred, octa_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Checkpointing
        torch.save(model.state_dict(), f"./train/wyt1/checkpoint_epoch_{epoch+1+10}.pth")


if __name__ == "__main__":
    main()
