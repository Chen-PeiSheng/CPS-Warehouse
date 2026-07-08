import pandas as pd
import torch
import json

# ----------------------
# 核心修复：超强兼容的embedding解析函数
# ----------------------
def clean_embedding(s):
    # 1. 处理空值
    if pd.isna(s):
        return None
    # 2. 转成字符串并去空格
    s = str(s).strip()
    # 3. 兼容带[]和不带[]的格式
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]  # 去掉括号
    # 4. 按逗号分割，转成数字列表
    try:
        emb_list = [float(x.strip()) for x in s.split(',')]
        # 5. 确保是768维（你的数据是768维）
        return emb_list if len(emb_list) == 768 else None
    except:
        # 解析失败返回None
        return None

# ----------------------
# 读取数据函数
# ----------------------
def load_embeddings(csv_path, data_name):
    print(f"\n📥 正在读取 {data_name}...")
    try:
        # 读取CSV（兼容中文和特殊字符）
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"   ✅ 成功打开文件，共 {len(df)} 行")
        print(f"   ✅ 列名：{df.columns.tolist()}")

        img_list = []
        txt_list = []

        # 遍历每一行
        for idx, row in df.iterrows():
            # 用下划线列名（你的文件是这个！）
            img_emb = clean_embedding(row['image_embedding'])
            txt_emb = clean_embedding(row['text_embedding'])

            # 只保留有效数据
            if img_emb is not None and txt_emb is not None:
                img_tensor = torch.tensor(img_emb, dtype=torch.float32)
                txt_tensor = torch.tensor(txt_emb, dtype=torch.float32)
                img_list.append(img_tensor)
                txt_list.append(txt_tensor)

        print(f"   ✅ 有效数据：图像 {len(img_list)} 个，文本 {len(txt_list)} 个")
        return img_list, txt_list

    except Exception as e:
        print(f"   ❌ 读取失败：{str(e)}")
        return [], []

# ----------------------
# 你的文件路径（改成桌面路径，确保能找到）
# ----------------------
correct_csv = r"C:\Users\33136\Desktop\correct_embeddings.csv"  # 正确对
halluc_csv  = r"C:\Users\33136\Desktop\halluc_embeddings.csv"   # 幻觉对
save_path   = r"C:\Users\33136\Desktop\final_data.pt"           # 保存到桌面

# ----------------------
# 执行读取
# ----------------------
print("="*60)
print("开始读取数据")
print("="*60)

# 读取正确对
correct_img, correct_txt = load_embeddings(correct_csv, "正确对CSV")

# 读取幻觉对（重点修复！）
halluc_img, halluc_txt = load_embeddings(halluc_csv, "幻觉对CSV")

# ----------------------
# 保存成模型能用的PT文件
# ----------------------
if len(correct_img) > 0 and len(halluc_img) > 0:
    torch.save({
        "correct_img": correct_img,
        "correct_txt": correct_txt,
        "wrong_img": halluc_img,  # 对应模型需要的wrong_img
        "wrong_txt": halluc_txt   # 对应模型需要的wrong_txt
    }, save_path)

    print("\n" + "="*60)
    print("🎉 全部成功！")
    print(f"✅ 正确对：{len(correct_img)} 对")
    print(f"✅ 幻觉对：{len(halluc_img)} 对")
    print(f"✅ PT文件保存到：{save_path}")
    print("✅ 现在可以直接运行训练代码了！")
    print("="*60)
else:
    print("\n" + "="*60)
    print("❌ 数据读取不完整，无法保存")
    print(f"   正确对有效数：{len(correct_img)}")
    print(f"   幻觉对有效数：{len(halluc_img)}")
    print("="*60)