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
    train_dataset = MedicalImageDataset(root=".\dataset\OCTA", split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MINIM(if_embed=False).to(device)

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

            # Sample timestep t
            t = torch.randint(0, T, (b,), device=device).long()

            # Forward pass
            pred = model(oct_batch, None, t.float())

            # Loss
            loss = F.mse_loss(pred, octa_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Checkpointing
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
