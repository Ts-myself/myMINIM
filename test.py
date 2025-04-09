import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model import *
from utils import *

font = ImageFont.load_default()


def combineImg(pred_octa_clamped, octa_batch_clamped):
    # 处理张量 -> numpy 数组，并确保它的形状是 (H, W) 或 (H, W, 3)
    true_img_array = (octa_batch_clamped.squeeze().cpu().numpy() * 255).astype(np.uint8)  # 形状 (256, 256)
    pred_img_array = (pred_octa_clamped.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 转换为 PIL 图像
    true_img = Image.fromarray(true_img_array, mode="L")  # 灰度图
    pred_img = Image.fromarray(pred_img_array, mode="L")

    true_img = true_img.convert("RGB")
    pred_img = pred_img.convert("RGB")

    width, height = true_img.size
    new_height = height + 30  # 额外添加 30px 作为标签区域

    # 创建空白画布
    combined_img = Image.new("RGB", (width * 2, new_height), (255, 255, 255))

    # 拼接图片
    combined_img.paste(true_img, (0, 50))
    combined_img.paste(pred_img, (width, 50))

    # 添加文本标签
    draw = ImageDraw.Draw(combined_img)
    text_y = 10  # 文字位置
    draw.text((width // 4, text_y), "True OCTA", fill=(0, 0, 0), font=font)
    draw.text((width + width // 4, text_y), "Synthetic OCTA", fill=(0, 0, 0), font=font)

    return combined_img


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # test_dataset = MedicalImageDataset(root="../dataset/OCTA", split="test", transform=transform)
    test_dataset = OCTADataset(root="dataset/OCTA500/OCTA_3mm", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MINIM_0(if_embed=False).to(device)
    checkpoint_path = "./myMINIM/train/SimpleUnet_weights/checkpoint_epoch_10.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    total_loss = 0.0
    sample_id = 0

    test_result_path = "./myMINIM/test/SimpleUnet"
    num = len([d for d in os.listdir(test_result_path) if os.path.isdir(os.path.join(test_result_path, d))]) + 1
    result_dir = os.path.join(test_result_path, f"result_{num}")
    output_dir = os.path.join(test_result_path, f"result_{num}", "output")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, item in enumerate(tqdm(test_loader)):
            oct_batch, octa_batch, disease, oct_path, octa_path = item['oct_image'], item['octa_image'], item['disease'], item['oct_path'], item['octa_path']
            oct_batch = oct_batch.to(device)
            octa_batch = octa_batch.to(device)
            b = oct_batch.size(0)

            t = torch.zeros(b, device=device).float()
            pred_octa = model(oct_batch, None, t)

            loss = F.mse_loss(pred_octa, octa_batch).item()
            total_loss += loss

            pred_octa_clamped = torch.clamp(pred_octa, 0, 1)
            octa_batch_clamped = torch.clamp(octa_batch, 0, 1)

            # combined_img = combineImg(pred_octa_clamped, octa_batch_clamped)
            # combined_img.save(os.path.join(output_dir, f"test_{i:04d}.png"))
            # save the predicted image and name as the octa_path
            pred_octa_clamped = pred_octa_clamped.squeeze().cpu()
            pred_octa_clamped = transforms.ToPILImage()(pred_octa_clamped)
            cur_octa_dir = os.path.join(output_dir, octa_path[0].split("/")[-2])
            os.makedirs(cur_octa_dir, exist_ok=True)
            pred_octa_clamped.save(os.path.join(cur_octa_dir, octa_path[0].split("/")[-1]))
            # save the original image and name as the oct_path

            tqdm.write(f"Sample {i}, Loss: {loss:.6f}")
            with open(os.path.join(result_dir, "test_log.txt"), "a") as f:
                f.write(f"Sample {i}, Loss: {loss:.6f}\n")

            sample_id += 1

    avg_loss = total_loss / sample_id
    tqdm.write(f"Sample {sample_id}, Loss: {loss:.6f}")
    with open(os.path.join(result_dir, "test_log.txt"), "a") as f:
        f.write(f"Average Loss: {avg_loss:.6f}\n")


if __name__ == "__main__":
    test()
