import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertTokenizer


# 时间嵌入模块，用于扩散时间步
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Linear(dim, dim)

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return self.embedding(emb)


# 交叉注意力层，结合图像和文本特征
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (h*w, b, c)
        attn_output, _ = self.attn(query=x, key=context.permute(1, 0, 2), value=context.permute(1, 0, 2))
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        return attn_output


class BERTENcoder(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, text):
        outputs = self.bert(text)
        return outputs.last_hidden_state  # (b, seq_len, hidden_dim)


# 文本编码器，使用 LSTM 替代 BERT
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, text):
        embedded = self.embedding(text)  # (b, seq_len)
        output, _ = self.lstm(embedded)  # (b, seq_len, hidden_dim)
        return output.permute(1, 0, 2)  # (seq_len, b, hidden_dim)


class LabelEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(1, hidden_dim)

    def forward(self, label):
        hidden = self.fc(label)
        return hidden


# 简化版 UNet，包含交叉注意力
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim, if_embed=True):
        super().__init__()
        self.if_embed = if_embed
        if self.if_embed:
            self.time_embed = TimeEmbedding(text_dim)
            self.time_proj = nn.Linear(text_dim, 128)  # 投影时间嵌入以匹配瓶颈通道
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.bottleneck_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bottleneck_attn = CrossAttention(dim=128)
        self.bottleneck_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1))
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, context, t):
        b = x.size(0)
        if self.if_embed:
            time_emb = self.time_embed(t)  # (b, text_dim)
            time_emb_proj = self.time_proj(time_emb)  # (b, 128)
            time_emb_exp = time_emb_proj[:, :, None, None]  # (b, 128, 1, 1)

        x1 = F.relu(self.enc1(x))  # (b, 64, H, W)
        x2 = self.down(x1)  # (b, 64, H/2, W/2)
        x3 = self.bottleneck_conv1(x2) + time_emb_exp if self.if_embed else self.bottleneck_conv1(x2)  # 添加时间嵌入
        x3 = F.relu(x3)
        x3 = self.bottleneck_attn(x3, context) if self.if_embed else x3  # 交叉注意力与文本
        x4 = (
            self.bottleneck_conv2(x3) + time_emb_exp if self.if_embed else self.bottleneck_conv2(x3)
        )  # 再次添加时间嵌入
        x4 = F.relu(x4)
        x5 = self.up(x4)  # (b, 128, H, W)
        x6 = torch.cat([x1, x5], dim=1)  # (b, 64+128=192, H, W)
        x7 = self.dec1(x6)  # (b, 64, H, W)
        output = self.out(x7)  # (b, out_channels, H, W)
        return output

# Define modified MINIM class without text embedding
class MINIM_0(nn.Module):
    def __init__(self, if_embed=True):
        super().__init__()
        self.if_embed = if_embed
        if self.if_embed:
            self.text_encoder = LSTMEncoder(vocab_size=10000, embedding_dim=128, hidden_dim=128)
        self.unet = SimpleUNet(in_channels=1, out_channels=1, text_dim=128, if_embed=if_embed)

    def forward(self, images, texts, t):
        if self.if_embed:
            text_emb = self.text_encoder(texts)  # (seq_len, b, hidden_dim)
        text_emb = None
        output = self.unet(images, text_emb, t)
        return output

class MINIM(nn.Module):
    def __init__(self, beta_min=1e-4, beta_max=0.02, T=1000, hidden_dim=256, num_attention_heads=8):
        super(MINIM, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.hidden_dim = hidden_dim

        self.oct_converter = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.vit_encoder = ViTModel.from_pretrained('myMINIM/huggingface/vit-base-patch16-224-in21k')
        
        self.tokenizer = BertTokenizer.from_pretrained('myMINIM/huggingface/bert-base-uncased')
        self.bert = BertModel.from_pretrained('myMINIM/huggingface/bert-base-uncased')
        
        # Unet
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # cross-attention: shallow for mode, deep for text
        self.cross_attn_shallow = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=True)
        self.cross_attn_deep = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=True)
        
        self.out_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1)
        
    def noise_schedule(self, t):
        return self.beta_min + t * ((self.beta_max - self.beta_min) / self.T)
    
    def forward_diffusion(self, image, t):
        beta_t = self.noise_schedule(t)
        noise = torch.randn_like(image)
        noisy_image = math.sqrt(1 - beta_t) * image + math.sqrt(beta_t) * noise
        return noisy_image, noise
    
    def cross_attention(self, x, text_emb, attn_layer):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, seq_len, C]
        # input size [B, seq_len, embed_dim]
        attn_output, _ = attn_layer(x_flat, text_emb, text_emb)
        # reshape to original
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        return attn_output

    def reverse_diffusion(self, noisy_image, t, modality_image, description_text):
        if modality_image.size(1) == 1:
            modality_image = self.oct_converter(modality_image)
        modality_emb = self.vit_encoder(pixel_values=modality_image).last_hidden_state  # [B, seq_len, hidden_dim]
        
        description_tokens = self.tokenizer(description_text, return_tensors="pt", padding=True, truncation=True)
        description_emb = self.bert(**description_tokens).last_hidden_state  # [B, seq_len, hidden_dim]
        
        x = self.conv1(noisy_image)  # [B, hidden_dim, H, W]
        
        # shallow Unet
        x = self.cross_attention(x, modality_emb, self.cross_attn_shallow)
        # deep Unet
        x = self.conv2(x)
        x = self.cross_attention(x, description_emb, self.cross_attn_deep)
        
        mu_theta = self.out_conv(x)
        
        sigma_t = self.noise_schedule(t)
        noise = torch.randn_like(mu_theta)
        x_prev = mu_theta + math.sqrt(sigma_t) * noise
        return x_prev

    def forward(self, image, t, modality_text, description_text):
        noisy_image, _ = self.forward_diffusion(image, t)
        x_prev = self.reverse_diffusion(noisy_image, t, modality_text, description_text)
        return x_prev