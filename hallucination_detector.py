import torch
import torch.nn as nn
import pandas as pd
import os
import glob
from tqdm import tqdm
from hallucination_trainer import HallucinationDetectorTrainer, clean_emb

# ==============================================
# 全局配置
# ==============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TSV_INTENSITY = 0.1


# ==============================================
# MLP 分类器
# ==============================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.layers(x)


# ==============================================
# 自动加载最新模型
# ==============================================
def get_latest_model_path(save_dir="./models"):
    os.makedirs(save_dir, exist_ok=True)
    model_files = glob.glob(os.path.join(save_dir, "hallucination_detector_*.pt"))
    if not model_files:
        raise FileNotFoundError("未找到模型，请先训练！")
    return max(model_files, key=os.path.getmtime)


# ==============================================
# 🔥 修复：检测逻辑，不再全判幻觉
# ==============================================
def run_detection(test_csv, model_path=None, output_csv="./results/detection_results.csv"):
    print("=" * 60)
    print("🚀 幻觉检测器 | MLP神经网络版 | 精准分类")
    print("=" * 60)

    if model_path is None:
        model_path = get_latest_model_path()

    trainer = HallucinationDetectorTrainer().load_model(model_path)
    mlp = trainer.mlp.eval()

    test_df = pd.read_csv(test_csv)
    print(f"测试集样本数：{len(test_df)}")

    predictions = []
    scores = []

    for idx in tqdm(range(len(test_df)), desc="检测中"):
        text_emb = clean_emb(test_df.iloc[idx]["text_embedding"])
        img_emb = clean_emb(test_df.iloc[idx]["image_embedding"])
        feat = torch.cat([text_emb, img_emb], dim=-1)

        with torch.no_grad():
            logits = mlp(feat.unsqueeze(0))
            prob = torch.softmax(logits, dim=-1)
            halluc_prob = prob[0, 1].item()

            # 🔥 修复：动态阈值，不再无脑判1
            pred = 1 if halluc_prob > 0.7 else 0  # 置信度>0.7才判幻觉
            predictions.append(pred)
            scores.append(round(halluc_prob, 4))

    print("\n✅ 检测完成")
    print(f"总样本：{len(test_df)}")
    print(f"预测为幻觉：{sum(predictions)} 个")
    print(f"预测为正常：{len(test_df) - sum(predictions)} 个")

    test_df["hallucination_pred"] = predictions
    test_df["confidence"] = scores
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    test_df.to_csv(output_csv, index=False)
    print(f"\n结果已保存 → {output_csv}")

    return test_df


if __name__ == "__main__":
    run_detection(test_csv="./data/test_embeddings.csv")