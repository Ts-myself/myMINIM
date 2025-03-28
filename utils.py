import os
from PIL import Image
from torch.utils.data import Dataset

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