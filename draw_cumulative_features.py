import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ==================== 论文风格绘图设置 ====================
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.0,
    'figure.dpi': 300
})

STEPS = 50
ts = np.linspace(0.01, 0.99, STEPS)


# ==================== 解析 Embedding ====================
def clean_emb(s):
    s = str(s).strip()
    if s.startswith('['):
        s = s[1:-1]
    return np.array([float(x.strip()) for x in s.split(',')], dtype=np.float32)


# ==================== 🔥 论文原版公式（已修复） ====================
def compute_paper_curve_original(text_vec, img_vec):
    text = torch.tensor(text_vec, dtype=torch.float32)
    img = torch.tensor(img_vec, dtype=torch.float32)

    lambda_list = []

    for t in ts:
        xt = (1 - t) * text + t * img
        M = torch.outer(xt, xt)
        eig_vals = torch.linalg.eigvalsh(M)
        lam = eig_vals[-1].item()
        lambda_list.append(lam)

    # ==================== 🔥 论文原版：符号激活 ====================
    lambda_np = np.array(lambda_list)
    activated = np.zeros_like(lambda_np)
    active = False

    for i in range(STEPS):
        if lambda_np[i] > 0:
            active = True
        if active:
            activated[i] = lambda_np[i]

    # ==================== 🔥 论文原版：累积积分 ====================
    cum = np.zeros_like(activated)
    for i in range(STEPS):
        cum[i] = np.mean(activated[:i + 1])

    # ==================== 🔥 论文原版：导数 ====================
    der = np.gradient(cum, ts)

    return cum, der


# ==================== ✅ 你的正确文件名（已修复） ====================
df_c = pd.read_csv(r"C:\Users\33136\Desktop\correct_embeddings.csv")
df_h = pd.read_csv(r"C:\Users\33136\Desktop\halluc_embeddings.csv")


def get_mean(df):
    cs, ds = [], []
    for i in range(len(df)):
        t = clean_emb(df.iloc[i]['text_embedding'])
        im = clean_emb(df.iloc[i]['image_embedding'])
        c, d = compute_paper_curve_original(t, im)
        cs.append(c)
        ds.append(d)
    return np.mean(cs, axis=0), np.mean(ds, axis=0)


Rc, dRc = get_mean(df_c)
Rh, dRh = get_mean(df_h)

# ==================== 绘图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

ax1.plot(ts, Rc, 'b-', linewidth=2, label="Correct")
ax1.plot(ts, Rh, 'r-', linewidth=2, label="Hallucination")
ax1.set_title("Cumulative Max-Eigenvalue")
ax1.set_xlabel("t")
ax1.legend()
ax1.grid(alpha=0.2)

ax2.plot(ts, dRc, 'b-', linewidth=2, label="Correct")
ax2.plot(ts, dRh, 'r-', linewidth=2, label="Hallucination")
ax2.set_title("Time Derivative")
ax2.set_xlabel("t")
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("paper_correct_version.png", dpi=300)
plt.close()
print("✅ 论文原版曲线已保存！")