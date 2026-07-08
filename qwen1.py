from openai import OpenAI
import os
import base64
import pandas as pd

# ===================== 配置 =====================
API_KEY = "sk-efe21d7c416741fb89d8233380370ac8"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.6-plus"

IMAGE_FOLDER = r"C:\Users\33136\Desktop\photo"
OUTPUT_CSV = "hallucination_final.csv"
GRID_SIZE = 24
# ================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def img2b64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def detect(img_path):
    b64 = img2b64(img_path)

    prompt = f"""
将图片分为{GRID_SIZE}×{GRID_SIZE}网格，左上角(0,0)，右下角({GRID_SIZE-1},{GRID_SIZE-1})。
全面检测所有AI生成幻觉与异常缺陷，包括：
人体畸形、多肢体少肢体、五官错乱缺失、手指脚趾数量异常、肢体错位变形、生物畸形、动物结构崩坏；
物体逻辑错误、形态扭曲、部件缺失多余、比例失调、空间透视错误、前后层次错乱、遮挡关系异常；
场景逻辑幻觉、环境结构矛盾、边界模糊粘连、物体悬浮穿模；
语义违背、常识错误、属性错乱、计数错误、重复冗余；
材质光影异常、纹理错乱、色彩畸变、反光阴影错误、细节崩坏；
局部扭曲拉伸、边缘锯齿、元素拼接错误、不合理拼接、违和异物。

【输出格式严格如下，不可改变】
坐标(x,y)+方位：幻觉描述
示例：
(3,5)左上：手指数量异常，多一根手指
(20,18)右下：人物肢体扭曲变形
无幻觉只输出：无幻觉
不要多余文字，不要解释，严格按格式输出！
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        stream=False
    )
    return resp.choices[0].message.content.strip()

def main():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(("jpg","jpeg","png","bmp"))]
    results = []

    for img in images:
        # ========== 已修复：os.path.splitext ==========
        img_id = os.path.splitext(img)[0]
        full_path = os.path.join(IMAGE_FOLDER, img)
        print(f"处理：{img}")

        res = detect(full_path)
        final_prompt = f"图片{img_id} 24×24网格幻觉检测：{res}"

        results.append({"图片ID": img_id, "Prompt": final_prompt})
        print(f"结果：{final_prompt}\n")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("✅ 全部完成！")

if __name__ == "__main__":
    main()