import torch
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

PROJ = torch.load("sb_proj.pt")
def project(x): return x @ PROJ

def clean_emb(s):
    s = str(s).strip()
    if s.startswith('['): s = s[1:-1]
    return torch.tensor([float(x.strip()) for x in s.split(',')], dtype=torch.float32)

df_correct = pd.read_csv(r"C:\Users\33136\Desktop\correct_embeddings.csv")
df_halluc = pd.read_csv(r"C:\Users\33136\Desktop\halluc_embeddings.csv")

correct_text_2d = [project(clean_emb(t)) for t in df_correct['text_embedding']]
correct_img_2d = [project(clean_emb(i)) for i in df_correct['image_embedding']]
halluc_text_2d = [project(clean_emb(t)) for t in df_halluc['text_embedding']]
halluc_img_2d = [project(clean_emb(i)) for i in df_halluc['image_embedding']]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter([p[0].item() for p in correct_text_2d], [p[1].item() for p in correct_text_2d], c='royalblue', s=40, alpha=0.7, label='Text')
plt.scatter([p[0].item() for p in correct_img_2d], [p[1].item() for p in correct_img_2d], c='forestgreen', s=40, alpha=0.7, label='Correct Image')
plt.title("Correct Pairs (SB)")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter([p[0].item() for p in halluc_text_2d], [p[1].item() for p in halluc_text_2d], c='royalblue', s=40, alpha=0.7, label='Text')
plt.scatter([p[0].item() for p in halluc_img_2d], [p[1].item() for p in halluc_img_2d], c='crimson', s=40, alpha=0.7, label='Hallucinated Image')
plt.title("Hallucinated Pairs (SB)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("2D_SB_Pair.png", dpi=300)
plt.close()
print("✅ 2D 图已保存：2D_SB_Pair.png")