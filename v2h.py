import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

input_root = 'myMINIM/test/SimpleUnet/result_1/output'
output_dir = os.path.join(input_root, 'project_full')

os.makedirs(output_dir, exist_ok=True)

for folder_name in os.listdir(input_root):
    folder_path = os.path.join(input_root, folder_name)
    
    if not os.path.isdir(folder_path) or folder_path == output_dir:
        continue

    try:
        image_paths = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.bmp'))]
        image_paths.sort(key=lambda x: int(x.split('.')[0]))
        
        first_image = Image.open(os.path.join(folder_path, image_paths[0]))
        image_2D = np.zeros((len(image_paths), first_image.size[0]))
        
        for idx, filename in enumerate(image_paths):
            img = Image.open(os.path.join(folder_path, filename))
            image_2D[idx] = np.max(np.array(img), axis=0)
        
        image_2D = np.flipud(image_2D)
        output_path = os.path.join(output_dir, f"{folder_name}.bmp")
        plt.imsave(output_path, image_2D, cmap='gray')
        
        print(f"成功处理文件夹: {folder_name}")
        
    except Exception as e:
        print(f"处理文件夹 {folder_name} 时出错: {str(e)}")
        continue

print("全部处理完成！")