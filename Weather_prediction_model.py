import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('D:\\coding\\WeatherDataCrawlingandVisualization-main\\output\\shenzhenmodel.xlsx')

# 提取特征和标签
x = df.iloc[1:, 6:8].values
y = df.iloc[1:, -1].values

# 划分数据集，80% 用于训练，20% 用于测试
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 训练决策树模型
clf.fit(x_train, y_train)

# 预测测试集
y_pred = clf.predict(x_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 使用 seaborn 绘制热力图
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no rain', 'rainy'], yticklabels=['no rain', 'rainy'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 计算精确率
precision = precision_score(y_test, y_pred)

# 计算召回率
recall = recall_score(y_test, y_pred)

# 计算F1值
f1 = f1_score(y_test, y_pred)

# 输出结果
print(f"准确率: {accuracy:.2f}")
print(f"精确率: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1值: {f1:.2f}")

# 可视化决策树
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=df.columns[6:8], class_names=[str(i) for i in set(y)], filled=True)
plt.show()