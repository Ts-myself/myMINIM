import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertTokenizer


# class TimeEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.embedding = nn.Linear(dim, dim)

#     def forward(self, t):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#         if self.dim % 2 == 1:
#             emb = F.pad(emb, (0, 1), mode="constant")
#         return self.embedding(emb)


# class CrossAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

#     def forward(self, x, context):
#         b, c, h, w = x.shape
#         x = x.view(b, c, h * w).permute(2, 0, 1)  # (h*w, b, c)
#         attn_output, _ = self.attn(query=x, key=context.permute(1, 0, 2), value=context.permute(1, 0, 2))
#         attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
#         return attn_output


# class BERTENcoder(nn.Module):
#     def __init__(self, bert_model_name):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)

#     def forward(self, text):
#         outputs = self.bert(text)
#         return outputs.last_hidden_state  # (b, seq_len, hidden_dim)


# class LSTMEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

#     def forward(self, text):
#         embedded = self.embedding(text)  # (b, seq_len)
#         output, _ = self.lstm(embedded)  # (b, seq_len, hidden_dim)
#         return output.permute(1, 0, 2)  # (seq_len, b, hidden_dim)


# class LabelEncoder(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.fc = nn.Linear(1, hidden_dim)

#     def forward(self, label):
#         hidden = self.fc(label)
#         return hidden


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.if_embed = if_embed
        # if self.if_embed:
        #     self.time_embed = TimeEmbedding(text_dim)
        #     self.time_proj = nn.Linear(text_dim, 128)  # 投影时间嵌入以匹配瓶颈通道
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.bottleneck_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bottleneck_attn = CrossAttention(dim=128)
        self.bottleneck_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1))
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = F.relu(self.enc1(x))  # (b, 64, H, W)
        x2 = self.down(x1)  # (b, 64, H/2, W/2)
        x3 = self.bottleneck_conv1(x2) +  self.bottleneck_conv1(x2)  # 添加时间嵌入
        x3 = F.relu(x3)
        x4 = (
            self.bottleneck_conv2(x3) 
        )  # 再次添加时间嵌入
        x4 = F.relu(x4)
        x5 = self.up(x4)  # (b, 128, H, W)
        x6 = torch.cat([x1, x5], dim=1)  # (b, 64+128=192, H, W)
        x7 = self.dec1(x6)  # (b, 64, H, W)
        output = self.out(x7)  # (b, out_channels, H, W)
        return output


# Define modified MINIM class without text embedding
class MINIM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.if_embed = if_embed
        # if self.if_embed:
        #     self.text_encoder = LSTMEncoder(vocab_size=10000, embedding_dim=128, hidden_dim=128)
        self.unet = SimpleUNet(in_channels=1, out_channels=1)

    def forward(self, images):
        output = self.unet(images)
        return output
