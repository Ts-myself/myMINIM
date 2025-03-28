import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import os
import datetime

from model import *
from utils import *

log = Logger()
# parameters
T = 1000
beta_min = 1e-4
beta_max = 0.03
hidden_dim = 256
num_attention_heads = 8

def train():
    epochs = 3
    lr = 1e-4

    # load data
    train_data_path = 'dataset/OCTA500/OCTA_3mm'
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    train_dataset = OCTADataset(train_data_path, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    log.info('data loaded')

    # init model
    model = MINIM(beta_min, beta_max, T, hidden_dim, num_attention_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    log.info('model initalized')

    # craete folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join('myMINIM/train', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    log.info('save_dir created')

    # training loop
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_dataloader)
        for oct, octa, disease in progress_bar:
            oct, octa = oct.to(device), octa.to(device)
            synth = model(octa, T, oct, disease)
            loss = criterion(synth, octa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"{save_dir}/checkpoint_epoch_{epoch+1}.pth")


# # Training script
# def main():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define transform
#     transform = transforms.Compose(
#         [
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ]
#     )

#     # Load dataset
#     train_dataset = MedicalImageDataset(root="../dataset/OCTA", split="train", transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#     # Initialize model
#     model = MINIM(if_embed=False).to(device)

#     # Load checkpoint if available
#     checkpoint_path = "./train/checkpoint_epoch_10.pth"
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint)
#         print(f"Checkpoint loaded successfully from {checkpoint_path}")
#     except FileNotFoundError:
#         print("Checkpoint not found, starting from scratch.")
#     except Exception as e:
#         print(f"Error loading checkpoint: {e}")

#     # Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     # Training loop
#     num_epochs = 3
#     for epoch in range(num_epochs):
#         model.train()
#         progress_bar = tqdm(train_loader)
#         for oct_batch, octa_batch in progress_bar:
#             oct_batch = oct_batch.to(device)
#             octa_batch = octa_batch.to(device)
#             b = octa_batch.size(0)
#             t = torch.randint(0, T, (b,), device=device).long()
#             pred = model(oct_batch, None, t.float())
#             loss = F.mse_loss(pred, octa_batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#         torch.save(model.state_dict(), f"./train/checkpoint_epoch_{epoch+1+10}.pth")


if __name__ == "__main__":
    train()
