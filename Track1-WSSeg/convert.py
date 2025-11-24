import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import csv

# ================= 配置区域 =================
# 选手的预测结果文件夹 (里面放着 16分类 的预测图)
USER_PRED_DIR = "./dataset/Chesapeake_NewYork_dataset/HR_lable_truth"      #请将grountruth修改为你保存图片的文件夹路径
OUTPUT_CSV = "./experiments/submission_gt.csv"    #submission_1.csv为程序运行后输出的csv文件保存路径

# 切片配置
TILE_SIZE = 1024

# 我们要提交给 Kaggle 的最终只有这 4 个基础类
TARGET_CLASS_IDS = [1, 2, 3, 4]

# -------------------------------------------------------------------------
# 标准映射表 (NLCD 16类 -> 比赛用 4类)
# 确保选手的 16 种细分地物能正确归类到 Water, Tree, LowVeg, Built-up
# -------------------------------------------------------------------------
CLASS_MAPPING = {
    1: 1,  # water 水体 → Water
    2: 2,  # tree canopy 树冠 → Tree canopy
    3: 3,  # low vegetation 低植被 → Low vegetation
    4: 3,  # barren 荒地 → Low vegetation
    5: 4,  # impervious (other) 不透水地（其他）→ Built-up
    6: 4   # impervious (road) 不透水地（道路）→ Built-up
}

# ================= 工具函数 =================

def rle_encode(mask):
    """Kaggle 标准 RLE 编码"""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def map_classes(array, mapping):
    """将预测结果映射为 4类"""
    mapped = np.zeros_like(array, dtype=np.uint8)
    for k, v in mapping.items():
        mapped[array == k] = v
    return mapped

def slice_and_process(image_id, full_mask_16_classes):
    records = []
    
    # 1. 映射: 把选手的 16类 结果转为 4类
    full_mask_4_classes = map_classes(full_mask_16_classes, CLASS_MAPPING)

    H, W = full_mask_4_classes.shape
    
    # 2. 边缘填充 (Padding)
    pad_h = (TILE_SIZE - H % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - W % TILE_SIZE) % TILE_SIZE
    
    if pad_h > 0 or pad_w > 0:
        full_mask_4_classes = np.pad(full_mask_4_classes, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    H_pad, W_pad = full_mask_4_classes.shape
    
    # 3. 循环切片
    for y in range(0, H_pad, TILE_SIZE):
        for x in range(0, W_pad, TILE_SIZE):
            tile = full_mask_4_classes[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            tile_base_id = f"{image_id.replace('predictions-new', 'lc')}_{y}_{x}"
            # 4. 提取 4 个类别的 RLE
            for class_id in TARGET_CLASS_IDS:
                binary_tile = (tile == class_id).astype(np.uint8)
                
                if np.sum(binary_tile) == 0:
                    rle = ""
                else:
                    rle = rle_encode(binary_tile)
                
                records.append({
                    "id": f"{tile_base_id}_{class_id}",
                    "rle_mask": rle
                })
    return records

# ================= 主执行逻辑 =================

def generate_submission():
    if not os.path.exists(USER_PRED_DIR):
        print(f"❌ 错误: 找不到文件夹 {USER_PRED_DIR}")
        return

    pred_files = sorted([f for f in os.listdir(USER_PRED_DIR) 
                     if f.endswith(".tif") and not f.startswith(".")])
    all_records = []
    
    print(f"🚀 开始处理 {len(pred_files)} 张预测图...")
    print("ℹ️  已启用 NLCD 映射: 将把 16类 预测结果转换为 4类 提交格式...")

    for fname in tqdm(pred_files, desc="Converting"):
        image_id = os.path.splitext(fname)[0]
        file_path = os.path.join(USER_PRED_DIR, fname)
        
        # 读取选手的预测结果 (预期是 1-16 的值)
        with rasterio.open(file_path) as src:
            pred_mask_16_classes = src.read(1)

        # 转换并切片
        file_records = slice_and_process(image_id, pred_mask_16_classes)
        all_records.extend(file_records)

    # 保存
    df = pd.DataFrame(all_records)
    df = df[['id', 'rle_mask']]
    df['rle_mask'] = df['rle_mask'].fillna("")
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n✅ 转换完成！已生成: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_submission()