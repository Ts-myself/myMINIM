import os
import csv
from glob import glob
from cli import HuatuoChatbot
from tqdm import tqdm
import pandas as pd

bot = HuatuoChatbot('./HuatuoGPT-Vision/model/HuatuoGPT-Vision-7B')

# load data
# data_root = './dataset/OCTA/OCT'
data_root = './dataset/OCTA500/OCTA_3mm/OCTA'
generate_dir = './HuatuoGPT-Vision/generate'

existing_folders = [f for f in os.listdir(generate_dir) if os.path.isdir(os.path.join(generate_dir, f))]
new_folder_name = f"result{len(existing_folders) + 1}"
save_dir = os.path.join(generate_dir, new_folder_name)
os.makedirs(save_dir, exist_ok=True)

image_paths = glob(os.path.join(data_root, '*/*.bmp'))


# load label
label_file = os.path.join(data_root, '../Text labels.xlsx')
labels_df = pd.read_excel(label_file)
id_to_label = {str(row['ID']): row['Disease'] for _, row in labels_df.iterrows()}

# query = "仔细看看这张OCT图像是否有疾病特征？如果有，说出疾病名称和描述一下，如果没有，就直接说无疾病。不要加额外语句。"
# query = "Does this OCT image show any disease? If any suspicious disease, discribe its name and details in the image. Do not add any unnecessary word. Answer in chinese."

def withLabel():
    # log
    # query_normal = "该OCT图像所有者无任何疾病，请以'无疾病'开头进行描述。用中文回答，示例：'无疾病，其中...'"
    # query_disease = ["该OCT图像所有者患有", "疾病，请详细描述特点（例如‘OCT中的黄斑区轻微肿胀’等位置和程度细节），并以疾病名称开头。用中文回答，示例：'患有DR疾病，其中...'"]
    query_normal = "This B-Scan OCT image shows no disease, just simply discribe it."
    query_disease = ["This B-Scan OCT image shows the disease of ", ", discribe it with details."]
    with open(os.path.join(save_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(query_normal + '\n')
        f.write(query_disease[0] + '\n')
        f.write(query_disease[1] + '\n')
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Description"])

    with tqdm(image_paths, desc="Processing images") as pbar:
        for i, image_path in enumerate(pbar):
            if i % 30 != 0:
                continue

            image_id = image_path.split('/')[-2]
            disease = id_to_label[image_id]
            if disease == "NORMAL":
                query = query_normal
            else:
                query = query_disease[0] + disease + query_disease[1]

            output = bot.inference(query, [image_path])
            pbar.set_postfix_str(f"Processed: {image_path}")

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([image_path, output])

    print(f"Results saved in {save_dir}")


def withChoice():
    # log
    query = "Describe the B-Scan format of the OCTA image whether it's NORMAL, CNV, DR or AMD."
    with open(os.path.join(save_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(query)
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Description"])

    with tqdm(image_paths, desc="Processing images") as pbar:
        for i, image_path in enumerate(pbar):
            if i % 30 != 0:
                continue

            output = bot.inference(query, [image_path])
            pbar.set_postfix_str(f"Processed: {image_path}")

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([image_path, output])

    print(f"Results saved in {save_dir}")

withLabel()