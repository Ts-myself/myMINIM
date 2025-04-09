import os
from PIL import Image
from torch.utils.data import Dataset
import datetime
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        初始化数据集。
        
        :param root: 数据集根目录（例如 'data/'）。
        :param split: 要加载的分割 ('train', 'test', 或 'val')，默认是 'train'。
        :param transform: 可选的图像变换。
        """
        self.root = root
        self.split = split
        self.transform = transform
        # 定义支持的图像扩展名
        # self.image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        self.image_extensions = ['.bmp']
        # 读取指定分割的 CSV 文件
        csv_file = os.path.join(root, f'{split}.csv')
        with open(csv_file, 'r') as f:
            ids = [line.strip() for line in f]
        self.pairs = []
        for id in ids:
            oct_folder = os.path.join(root, 'OCT', id)
            octa_folder = os.path.join(root, 'OCTA', id)
            # 获取 OCT 和 OCTA 的图像文件列表
            oct_images = [os.path.join(oct_folder, f) for f in os.listdir(oct_folder) 
                         if any(f.lower().endswith(ext) for ext in self.image_extensions)]
            octa_images = [os.path.join(octa_folder, f) for f in os.listdir(octa_folder) 
                          if any(f.lower().endswith(ext) for ext in self.image_extensions)]
            # 排序以确保配对一致
            oct_images = sorted(oct_images)
            octa_images = sorted(octa_images)
            # 检查图像数量是否匹配
            if len(oct_images) != len(octa_images):
                raise ValueError(f'ID {id} 的 OCT 和 OCTA 图像数量不匹配: {len(oct_images)} vs {len(octa_images)}')
            # 创建配对
            for i in range(len(oct_images)):
                self.pairs.append((oct_images[i], octa_images[i]))

    def __len__(self):
        """
        返回数据集的总样本数。
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        
        :param idx: 要检索的样本索引。
        :return: 一个元组 (oct_img, octa_img)，其中 oct_img 是输入图像，octa_img 是目标图像。
        """
        oct_path, octa_path = self.pairs[idx]
        oct_img = Image.open(oct_path)
        octa_img = Image.open(octa_path)
        if self.transform:
            oct_img = self.transform(oct_img)
            octa_img = self.transform(octa_img)
        return oct_img, octa_img

class OCTADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.oct_dir = os.path.join(root, "OCT")
        self.octa_dir = os.path.join(root, "OCTA")
        self.label_file = os.path.join(root, "Text labels.xlsx")
        self.transform = transform

        # 读取Excel文件，加载标签信息
        self.labels_df = pd.read_excel(self.label_file)
        # 构造ID与disease的映射，ID均转换为字符串格式
        self.id_to_label = {str(row['ID']): row['Disease'] for _, row in self.labels_df.iterrows()}

        # 构建所有配对（pair）的列表，每个元素包含 (oct_path, octa_path, disease)
        self.pairs = []
        # 遍历 OCT 文件夹中的每个患者文件夹
        patient_ids = os.listdir(self.oct_dir)
        for pid in patient_ids:
            # 检查该患者是否在 OCTA 中存在，并且有标签记录
            if (pid in os.listdir(self.octa_dir)) and (pid in self.id_to_label):
                oct_patient_dir = os.path.join(self.oct_dir, pid)
                octa_patient_dir = os.path.join(self.octa_dir, pid)

                # 获取当前患者所有bmp图片的文件名，并排序（假设文件名数字越小，顺序越靠前）
                oct_files = sorted([f for f in os.listdir(oct_patient_dir) if f.lower().endswith('.bmp')])
                octa_files = sorted([f for f in os.listdir(octa_patient_dir) if f.lower().endswith('.bmp')])
                
                # 以较少的图片数为准，进行配对
                n = min(len(oct_files), len(octa_files))
                for i in range(n):
                    oct_path = os.path.join(oct_patient_dir, oct_files[i])
                    octa_path = os.path.join(octa_patient_dir, octa_files[i])
                    disease = self.id_to_label[pid]
                    self.pairs.append((oct_path, octa_path, disease))
                    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        oct_path, octa_path, disease = self.pairs[idx]
        # 打开图片
        oct_img = Image.open(oct_path)
        octa_img = Image.open(octa_path)
        
        # 如果有预处理函数，则应用
        if self.transform:
            oct_img = self.transform(oct_img)
            octa_img = self.transform(octa_img)
        
        sample = {
            'oct_image': oct_img,
            'octa_image': octa_img,
            'disease': disease,
            'oct_path': oct_path,
            'octa_path': octa_path
        }
        return sample


class Logger:
    def __init__(self):
        self.levels = {
            "INFO": "\033[92m[INFO]\033[0m",   # 绿色
            "WARN": "\033[93m[WARN]\033[0m",   # 黄色
            "ERROR": "\033[91m[ERROR]\033[0m", # 红色
        }

    def log(self, level, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} {self.levels.get(level, '[LOG]')} {message}")

    def info(self, message):
        self.log("INFO", message)

    def warn(self, message):
        self.log("WARN", message)

    def error(self, message):
        self.log("ERROR", message)