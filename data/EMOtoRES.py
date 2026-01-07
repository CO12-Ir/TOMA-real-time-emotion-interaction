import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib

# 加载 Excel 文件
df = pd.read_excel("RES.xlsx")

# 查看前几行
print(df.head())


# 模型 A：包含 label 情绪分类
X_a = df[['label', 'valence', 'arousal', 'dominance']]
# 编码label：
custom_classes = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
le_input_label = LabelEncoder()
le_input_label.classes_ = np.array(custom_classes)

# 将 label 列映射为整数
X_a['label'] = le_input_label.transform(X_a['label'])

# 模型 B：只用 VAD
X_b = df[['valence', 'arousal', 'dominance']]


# 统一目标
y = df['response']
le = LabelEncoder()
y_encoded = le.fit_transform(y) 

# 拆分数据（保持 response 分布一致）
X_a_train, X_a_test, y_train, y_test = train_test_split(X_a, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42)
X_b_train, X_b_test, _, _ = train_test_split(X_b, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42)

# 模型 A 训练
clf_a = LGBMClassifier(
    n_estimators=152, max_depth=4,
    learning_rate=0.0341, num_leaves=39,
    min_child_samples=27, random_state=42
)
clf_a.fit(X_a_train, y_train)
pred_a = clf_a.predict(X_a_test)
labels_a = le.inverse_transform(np.round(pred_a).astype(int))

# 模型 B 训练
clf_b = LGBMClassifier(
    n_estimators=152, max_depth=4,
    learning_rate=0.0341, num_leaves=39,
    min_child_samples=27, random_state=42
)
clf_b.fit(X_b_train, y_train)
pred_b = clf_b.predict(X_b_test)
labels_b = le.inverse_transform(np.round(pred_b).astype(int))



# 输出对比
print("模型 A（带label）分类报告：")
print(classification_report(y_test, pred_a))
print("模型 B（纯VAD）分类报告：")
print(classification_report(y_test, pred_b ))




joblib.dump(clf_a, 'toma_response_model.pkl')
joblib.dump(clf_b, 'toma_response_model2.pkl')