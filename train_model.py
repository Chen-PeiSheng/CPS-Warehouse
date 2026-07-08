import torch
import numpy as np
import pandas as pd

# ==============================================
# 你的原始参数
# ==============================================
EMBED_DIM = 2
EPS = 0.01
SINKHORN_ITER = 50

correct_path = r"D:\PycharmProjects\PythonProject\跑分\幻觉\data\correct_embeddings.csv"
halluc_path = r"D:\PycharmProjects\PythonProject\跑分\幻觉\data\halluc_embeddings.csv"

# ==============================================
# 你的原始薛定谔桥代码
# ==============================================
def clean_emb(s):
    s = str(s).strip()
    if s.startswith('['):
        s = s[1:-1]
    return [float(x.strip()) for x in s.split(',')]

def schrodinger_bridge_align(img_list, txt_list):
    X = img_list.numpy()
    Y = txt_list.numpy()

    xx = np.sum(X ** 2, axis=1, keepdims=True)
    yy = np.sum(Y ** 2, axis=1, keepdims=True)
    xy = X @ Y.T
    C = xx + yy.T - 2 * xy

    n, m = C.shape
    mu = np.ones(n) / n
    nu = np.ones(m) / m
    K = np.exp(-C / EPS)
    u, v = np.ones(n), np.ones(m)

    for _ in range(SINKHORN_ITER):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.T @ u + 1e-8)

    P = np.diag(u) @ K @ np.diag(v)
    M = X.T @ P @ Y
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return torch.from_numpy(U[:, :EMBED_DIM]).float()

# ==============================================
# 加载原始数据（一字未改）
# ==============================================
df_correct = pd.read_csv(correct_path)
df_halluc = pd.read_csv(halluc_path)

correct_text = torch.tensor([clean_emb(x) for x in df_correct['text_embedding']])
correct_img = torch.tensor([clean_emb(x) for x in df_correct['image_embedding']])
halluc_text = torch.tensor([clean_emb(x) for x in df_halluc['text_embedding']])
halluc_img = torch.tensor([clean_emb(x) for x in df_halluc['image_embedding']])

all_text = torch.cat([correct_text, halluc_text])
all_img = torch.cat([correct_img, halluc_img])

# ==============================================
# 训练原始薛定谔桥（一字未改）
# ==============================================
proj = schrodinger_bridge_align(all_img, all_text)

# 保存原始模型
torch.save({
    "proj": proj,
    "embed_dim": all_text.size(1)
}, "sb_proj_original.pt")

print("✅ 原始薛定谔桥模型训练完成！")
print(f"   投影矩阵维度: {proj.shape}")
print(f"   模型已保存为: sb_proj_original.pt")