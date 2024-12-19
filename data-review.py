import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import re

# 设置Seaborn样式
sns.set(style="whitegrid")

# 读取Excel文件
excel_file = 'D:\\coding\\WeatherDataCrawlingandVisualization-main\\output\\nanjingLast30DaysWeather.xlsx'  # 替换为你的Excel文件路径
try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    print(f"错误：未找到文件 {excel_file}。请检查文件路径是否正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时出错：{e}")
    exit()

# 清理列名：移除前后空格
df.columns = df.columns.str.strip()

# 打印列名以确认
print("清理后的列名:")
print(df.columns.tolist())

# 数据预览
print("\n数据预览:")
print(df.head())

# 确保日期列为日期格式
df['日期'] = pd.to_datetime(df['日期'])

# 分离天气状况为白天和夜间
if '天气状况(白天/夜间)' in df.columns:
    # 假设格式为 "晴/多云" 或 "晴, 多云"
    df[['天气状况白天', '天气状况夜间']] = df['天气状况(白天/夜间)'].str.split('[/,，]', expand=True)
else:
    print("错误：Excel 文件中缺少 '天气状况(白天/夜间)' 列。请检查列名是否正确。")
    exit()

# 清洗温度数据：移除'°C'并转换为数值类型
def clean_temperature(temp_str):
    if isinstance(temp_str, str):
        # 使用正则表达式提取数字，包括负数
        match = re.search(r'(-?\d+)', temp_str)
        if match:
            return int(match.group(1))
    elif isinstance(temp_str, (int, float)):
        return temp_str
    return np.nan  # 如果无法转换，返回NaN

df['最高气温'] = df['最高气温'].apply(clean_temperature)
df['最低气温'] = df['最低气温'].apply(clean_temperature)

# 检查是否有缺失值
if df['最高气温'].isnull().any() or df['最低气温'].isnull().any():
    print("警告：部分温度数据无法转换为数值类型。请检查数据源。")

# 分解风力风向
def parse_wind(wind_str):
    if isinstance(wind_str, str):
        direction = ''.join(filter(str.isalpha, wind_str))
        strength = ''.join(filter(str.isdigit, wind_str))
        strength = int(strength) if strength.isdigit() else 0
        return direction, strength
    return ("未知", 0)

df[['白天风向', '白天风力']] = df['风力风向(白天)'].apply(lambda x: pd.Series(parse_wind(x)))
df[['夜间风向', '夜间风力']] = df['风力风向(夜间)'].apply(lambda x: pd.Series(parse_wind(x)))

# 打印处理后的数据以确认
print("\n处理后的数据预览:")
print(df.head())

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图，进一步增加高度并调整间距
fig, axs = plt.subplots(3, 1, figsize=(16, 30))  # 增大图形高度

# 手动调整子图间的间距
plt.subplots_adjust(hspace=0.6)  # 增加hspace到0.6

# 1. 显示天气状况与温度
axs[0].set_title('天气状况与温度', fontsize=22, pad=25)  # 增加标题与子图之间的间距
# 绘制最高气温和最低气温的柱状图
sns.barplot(x='日期', y='最高气温', data=df, palette='Blues', ax=axs[0], label='最高气温')
sns.barplot(x='日期', y='最低气温', data=df, palette='Oranges', ax=axs[0], label='最低气温')

axs[0].set_ylabel('温度 (°C)', fontsize=18)
axs[0].set_xlabel('日期', fontsize=18)
axs[0].legend(title='温度')

# 在柱状图上方添加天气状况的文本标签
for idx, row in df.iterrows():
    # 调整y位置以防止重叠
    axs[0].text(idx, row['最高气温'] + 1, row['天气状况白天'], ha='center', va='bottom', fontsize=14, color='black')
    axs[0].text(idx, row['最低气温'] - 2, row['天气状况夜间'], ha='center', va='top', fontsize=14, color='gray')

handles, labels = axs[0].get_legend_handles_labels()
selected_handles = [handles[28], handles[57]]
selected_labels = [labels[28], labels[57]]
axs[0].legend(handles=selected_handles, labels=selected_labels, title='温度', loc='center right', bbox_to_anchor=(1.1, 0.5))

# 2. 显示白天和夜间风力风向
axs[1].set_title('白天和夜间风力风向', fontsize=22, pad=25)  # 增加标题与子图之间的间距

# 绘制白天风力和夜间风力的柱状图
sns.barplot(x='日期', y='白天风力', data=df, palette='Greens', ax=axs[1], label='白天风力')
sns.barplot(x='日期', y='夜间风力', data=df, palette='Purples', ax=axs[1], label='夜间风力')

axs[1].set_ylabel('风力等级', fontsize=18)
axs[1].set_xlabel('日期', fontsize=18)
axs[1].legend(title='风力')

# 在柱状图上方添加风向信息的文本标签
for idx, row in df.iterrows():
    # 调整y位置以防止重叠
    axs[1].text(idx, row['白天风力'] + 1, row['白天风向'], ha='center', va='bottom', fontsize=12, color='gray')
    axs[1].text(idx, row['夜间风力'] - 2, row['夜间风向'], ha='center', va='top', fontsize=12, color='black')

handles, labels = axs[1].get_legend_handles_labels()
selected_handles = [handles[28], handles[57]]
selected_labels = [labels[28], labels[57]]
axs[1].legend(handles=selected_handles, labels=selected_labels, title='风力', loc='center right', bbox_to_anchor=(1.1, 0.5))

# 3. 显示温度变化趋势
axs[2].set_title('温度变化趋势', fontsize=22, pad=25)  # 增加标题与子图之间的间距
sns.lineplot(x='日期', y='最高气温', data=df, marker='o', label='最高气温', ax=axs[2])
sns.lineplot(x='日期', y='最低气温', data=df, marker='o', label='最低气温', ax=axs[2])

axs[2].set_ylabel('温度 (°C)', fontsize=18)
axs[2].set_xlabel('日期', fontsize=18)
axs[2].legend(title='温度')

# 美化x轴日期显示
for ax in axs:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

# 旋转标签并减小字体
for ax in axs:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)
# axs[0].legend(title='温度', loc='center left', bbox_to_anchor=(-0.1, 0.5))
axs[2].legend(title='温度', loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.tight_layout()
# 显示图表
plt.show()
