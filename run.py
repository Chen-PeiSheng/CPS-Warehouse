import pandas as pd
import os
from hallucination_trainer import HallucinationDetectorTrainer
from hallucination_detector import run_detection

# ==============================================！
# 示例路径（请替换成你实际的文件位置）
NORMAL_TRAIN_CSV = "D:/PycharmProjects/PythonProject/跑分/幻觉/data/correct_embeddings.csv"
HALLUCINATION_TRAIN_CSV = "D:/PycharmProjects/PythonProject/跑分/幻觉/data/halluc_embeddings.csv"
TEST_CSV = "D:/PycharmProjects/PythonProject/跑分/幻觉/data/test_embeddings.csv"
EMBEDDING_DIM = 768  # 你的embedding维度（默认768，如不同请修改）


def merge_train_data(normal_csv, hallucination_csv):
    """自动合并正常/幻觉样本，添加label列（0=正常，1=幻觉）"""
    print("=" * 60)
    print("🔧 合并训练数据（正常+幻觉）")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(normal_csv):
        raise FileNotFoundError(f"正常样本文件不存在：{normal_csv}")
    if not os.path.exists(hallucination_csv):
        raise FileNotFoundError(f"幻觉样本文件不存在：{hallucination_csv}")

    # 加载并添加label
    normal_df = pd.read_csv(normal_csv)
    normal_df["label"] = 0
    halluc_df = pd.read_csv(hallucination_csv)
    halluc_df["label"] = 1

    # 合并打乱
    combined_df = pd.concat([normal_df, halluc_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ 正常样本：{len(normal_df)} 个（来自 correct_embeddings.csv）")
    print(f"✅ 幻觉样本：{len(halluc_df)} 个（来自 halluc_embeddings.csv）")
    print(f"✅ 合并后总训练样本：{len(combined_df)} 个")
    print("=" * 60)

    return combined_df


def main():
    # 1. 合并训练数据
    train_df = merge_train_data(NORMAL_TRAIN_CSV, HALLUCINATION_TRAIN_CSV)

    # 2. 初始化训练器
    trainer = HallucinationDetectorTrainer(embedding_dim=EMBEDDING_DIM)

    # 3. 训练模型（自动选32个标注样本）
    print("\n🚀 开始训练模型")
    labeled_df = train_df
    trainer.train_model(
        train_df=train_df,
        labeled_df=labeled_df,
        use_pnas_upweight=True
    )

    # 4. 保存模型（保存在当前项目的models目录）
    model_dir = "/跑分/幻觉/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = trainer.save_model(save_dir=model_dir)

    # 5. 运行检测（结果保存在results目录）
    print("\n🔍 开始批量检测")
    result_dir = "/跑分/幻觉/results"
    os.makedirs(result_dir, exist_ok=True)
    results_df = run_detection(
        test_csv=TEST_CSV,
        model_path=model_path,
        output_csv=f"{result_dir}/detection_results.csv"
    )

    print("\n🎉 所有流程完成！")
    print(f"📁 模型保存路径：{model_path}")
    print(f"📁 检测结果路径：{result_dir}/detection_results.csv")


if __name__ == "__main__":
    main()