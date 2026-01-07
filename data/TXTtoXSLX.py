import re
import pandas as pd

input_txt = "scared.txt"
output_xlsx = "output_scared.xlsx"

rows = []

with open(input_txt, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 匹配：标签 + [三个浮点数]
        match = re.match(
            r"(\w+)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
            line
        )

        if match:
            label, v, a, d = match.groups()
            rows.append([label, float(v), float(a), float(d)])

# 转成 DataFrame
df = pd.DataFrame(rows, columns=["label", "valence", "arousal", "dominance"])

# 导出 xlsx
df.to_excel(output_xlsx, index=False)

print("✅ 转换完成，已生成:", output_xlsx)
