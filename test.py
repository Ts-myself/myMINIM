import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

from model import MINIM
from utils import MedicalImageDataset


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = MedicalImageDataset(root="../dataset/OCTA", split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MINIM(if_embed=False).to(device)
    checkpoint_path = "./train/checkpoint_epoch_3.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    total_loss = 0.0
    sample_id = 0

    test_result_path = "./test"
    num = len([d for d in os.listdir(test_result_path) if os.path.isdir(os.path.join(test_result_path, d))]) + 1
    output_dir = os.path.join(test_result_path, f"result_{num}", "output")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (oct_batch, octa_batch) in enumerate(tqdm(test_loader)):
            oct_batch = oct_batch.to(device)
            octa_batch = octa_batch.to(device)
            b = oct_batch.size(0)

            t = torch.zeros(b, device=device).float()
            pred_octa = model(oct_batch, None, t)

            loss = F.mse_loss(pred_octa, octa_batch).item()
            total_loss += loss

            pred_octa_clamped = torch.clamp(pred_octa, 0, 1)
            octa_batch_clamped = torch.clamp(octa_batch, 0, 1)

            # 拼接图片（左侧为真实，右侧为生成）
            combined = torch.cat((octa_batch_clamped, pred_octa_clamped), dim=3)  # 在宽度维度拼接
            save_image(combined, os.path.join(output_dir, f"test_{i:04d}.png"))

            tqdm.write(f"Sample {i}, Loss: {loss:.6f}")
            with open(os.path.join(output_dir, "test_log.txt"), "a") as f:
                f.write(f"Sample {i}, Loss: {loss:.6f}\n")

            sample_id += 1

    avg_loss = total_loss / sample_id
    tqdm.write(f"Sample {sample_id}, Loss: {loss:.6f}")
    with open(os.path.join(output_dir, "test_log.txt"), "a") as f:
        f.write(f"Average Loss: {avg_loss:.6f}\n")


if __name__ == "__main__":
    test()
