import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ==============================================
# 核心参数（仅调这里即可）
# ==============================================
# 增强开关
USE_DUAL_SCORE = True
USE_POST_FUSION = True

# 权重参数
DISTANCE_WEIGHT = 0.6
SIMILARITY_WEIGHT = 0.4
FUSION_STRENGTH = 0.2

# 🔴 召回强度档位（无标签场景专用）
# 可选值："ultra_high" 极高召回 | "high" 高召回 | "balanced" 均衡
# 越往左阈值越低，检出幻觉越多，漏检越少，但误判会增加
RECALL_LEVEL = "high"

# ==============================================
# 配置
# ==============================================
EMBED_DIM = 2
TEST_CSV = r"D:\PycharmProjects\PythonProject\跑分\幻觉\data\test_embeddings.csv"
MODEL_PATH = "sb_proj_original.pt"
DEVICE = "cpu"


# ==============================================
# 工具函数
# ==============================================
def clean_emb(s):
    s = str(s).strip()
    if s.startswith('['):
        s = s[1:-1]
    return torch.tensor([float(x.strip()) for x in s.split(',')], device=DEVICE)


# ==============================================
# 增强后融合
# ==============================================
def enhanced_post_fusion(text_proj, img_proj, text_emb, img_emb):
    if not USE_POST_FUSION:
        return img_proj

    text_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    img_norm = torch.nn.functional.normalize(img_emb, dim=-1)
    global_similarity = torch.sum(text_norm * img_norm)

    fusion_strength = FUSION_STRENGTH * (1 + 2 * (1 - global_similarity))
    fused_img_proj = img_proj + fusion_strength * (img_proj - text_proj)

    return fused_img_proj


# ==============================================
# 双维度联合打分
# ==============================================
def compute_dual_hallucination_score(text_emb, img_emb, text_proj, img_proj_fused):
    # 维度1：薛定谔桥距离（越大越可能是幻觉）
    distance = torch.norm(text_proj - img_proj_fused).item()

    # 维度2：图文余弦相似度（越小越可能是幻觉）
    text_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    img_norm = torch.nn.functional.normalize(img_emb, dim=-1)
    similarity = torch.sum(text_norm * img_norm).item()

    # 归一化
    norm_distance = (distance - min_dist) / (max_dist - min_dist)
    norm_similarity = 1 - (similarity - min_sim) / (max_sim - min_sim)

    # 加权融合最终分数
    final_score = DISTANCE_WEIGHT * norm_distance + SIMILARITY_WEIGHT * norm_similarity

    return final_score, distance, similarity


# ==============================================
# 无标签场景：自适应阈值计算
# ==============================================
def compute_adaptive_threshold(scores, level="high"):
    """
    根据分数分布自动计算阈值，不强制固定比例
    原理：用均值和标准差描述分布，根据召回档位向下放宽阈值
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # 不同档位对应不同的宽松程度：k越大，阈值越低，召回越高
    level_map = {
        "ultra_high": 1.0,  # 极高召回：均值-1倍标准差，覆盖大部分样本
        "high": 0.5,  # 高召回：均值-0.5倍标准差（默认）
        "balanced": 0.0  # 均衡：均值作为阈值
    }
    k = level_map.get(level, 0.5)

    threshold = mean_score - k * std_score
    # 防止阈值低于最小值（极端情况）
    threshold = max(threshold, np.min(scores) - 1e-6)
    return threshold, mean_score, std_score


# ==============================================
# 加载模型
# ==============================================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
proj = ckpt["proj"].to(DEVICE)

# ==============================================
# 主检测流程
# ==============================================
df = pd.read_csv(TEST_CSV)
has_label = "label" in df.columns

all_distances = []
all_similarities = []
sample_details = []

print("🚀 文生图幻觉检测（无标签高召回模式）")
print("=" * 70)
print(f"模式: 双维度联合打分 + 分布自适应宽松阈值")
print(f"召回档位: {RECALL_LEVEL}")
print(f"距离权重: {DISTANCE_WEIGHT}, 相似度权重: {SIMILARITY_WEIGHT}")
print(f"融合强度: {FUSION_STRENGTH}")

# 第一步：计算所有样本的基础特征，用于归一化
for idx in range(len(df)):
    row = df.iloc[idx]
    img_emb = clean_emb(row["image_embedding"])
    text_emb = clean_emb(row["text_embedding"])

    text_proj = text_emb @ proj
    img_proj = img_emb @ proj
    img_proj_fused = enhanced_post_fusion(text_proj, img_proj, text_emb, img_emb)

    distance = torch.norm(text_proj - img_proj_fused).item()
    text_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    img_norm = torch.nn.functional.normalize(img_emb, dim=-1)
    similarity = torch.sum(text_norm * img_norm).item()

    all_distances.append(distance)
    all_similarities.append(similarity)

# 全局归一化参数
min_dist, max_dist = min(all_distances), max(all_distances)
min_sim, max_sim = min(all_similarities), max(all_similarities)

# 第二步：计算所有样本的最终幻觉分数
final_scores = []
for idx in range(len(df)):
    row = df.iloc[idx]
    img_emb = clean_emb(row["image_embedding"])
    text_emb = clean_emb(row["text_embedding"])

    text_proj = text_emb @ proj
    img_proj = img_emb @ proj
    img_proj_fused = enhanced_post_fusion(text_proj, img_proj, text_emb, img_emb)

    score, dist, sim = compute_dual_hallucination_score(text_emb, img_emb, text_proj, img_proj_fused)

    final_scores.append(score)
    sample_details.append({
        "样本序号": idx + 1,
        "原始距离": round(dist, 4),
        "原始相似度": round(sim, 4),
        "最终幻觉分数": round(score, 4)
    })

# 打印分布统计
print("\n📊 分数分布统计")
print("=" * 70)
print(f"距离范围: {min_dist:.4f} ~ {max_dist:.4f}")
print(f"相似度范围: {min_sim:.4f} ~ {max_sim:.4f}")
print(f"最终分数范围: {min(final_scores):.4f} ~ {max(final_scores):.4f}")
print(f"分数均值: {np.mean(final_scores):.4f}")
print(f"分数标准差: {np.std(final_scores):.4f}")

# ===================== 核心：自适应阈值计算 =====================
print("\n🎯 阈值计算（无标签自适应高召回）")
print("-" * 70)
THRESHOLD, mean_s, std_s = compute_adaptive_threshold(final_scores, RECALL_LEVEL)
print(f"阈值计算方式: 均值 - {0.5 if RECALL_LEVEL == 'high' else 1.0 if RECALL_LEVEL == 'ultra_high' else 0} × 标准差")
print(f"分数均值: {mean_s:.4f}, 标准差: {std_s:.4f}")
print(f"最终阈值: {THRESHOLD:.6f}")
print(f"说明: 无强制比例，检出数量由样本分数分布自动决定，优先覆盖疑似幻觉")

# 生成预测结果
predictions = [1 if s > THRESHOLD else 0 for s in final_scores]
hallucination_samples = [s for s, p in zip(sample_details, predictions) if p == 1]

# 输出检测结果
print("\n📊 最终检测结果")
print("=" * 70)
print(f"总样本数: {len(predictions)}")
print(f"✅ 正常样本: {len(predictions) - len(hallucination_samples)} 个")
print(f"🔴 检测到幻觉: {len(hallucination_samples)} 个")

# 有标签时补充指标（兼容有标签场景）
if has_label:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_true = df["label"].tolist()
    print("\n📈 评估指标（仅作参考）")
    print("-" * 70)
    print(f"召回率 Recall: {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"精确率 Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"准确率 Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print(f"F1 分数: {f1_score(y_true, predictions, zero_division=0):.4f}")
    print(f"AUC 分数: {roc_auc_score(y_true, final_scores):.4f}")

# 打印幻觉样本列表
print("\n🔴 被检测为幻觉的样本列表（按分数从高到低排序）")
print("-" * 90)
print(f"{'样本序号':<10} {'原始距离':<12} {'原始相似度':<14} {'最终幻觉分数':<14}")
print("-" * 90)
hallucination_samples_sorted = sorted(hallucination_samples, key=lambda x: x["最终幻觉分数"], reverse=True)
for s in hallucination_samples_sorted:
    print(f"{s['样本序号']:<10} {s['原始距离']:<12} {s['原始相似度']:<14} {s['最终幻觉分数']:<14}")

# 保存结果
full_result = pd.DataFrame(sample_details)
full_result["预测(1=幻觉)"] = predictions
if has_label:
    full_result["真实标签"] = df["label"].tolist()
full_result.to_csv("detection_result_unlabeled_high_recall.csv", index=False, encoding="utf-8-sig")
print(f"\n💾 完整结果已保存到: detection_result_unlabeled_high_recall.csv")