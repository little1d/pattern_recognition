import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 CSV 文件路径
csv_paths = ['iris_results.csv', 'sonar_results.csv', 'mnist_results.csv']
dataset_names = ['iris', 'sonar', 'mnist']
print(csv_paths)
for i in range(3):
    data = pd.read_csv(csv_paths[i])  # 读取 CSV 文件
    dataset_name = dataset_names[i]   # 获取数据集名称
    
    # 将方法作为分类数据处理
    data['method'] = data['method'].astype('category')

    # 设置图表风格
    sns.set(style='whitegrid')

    # 可视化 silhouette Coefficient
    plt.figure(figsize=(10, 6))
    sns.barplot(x='method', y='silhouette_Coefficient', data=data, palette='viridis')
    plt.title(f'Silhouette Coefficient by Method - {dataset_name.capitalize()}')
    plt.xlabel('Method')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_silhouette_coefficient.png')
    plt.show()

    # 可视化 Calinski-Harabasz Score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='method', y='calinski_harabasz_score', data=data, palette='viridis')
    plt.title(f'Calinski-Harabasz Score by Method - {dataset_name.capitalize()}')
    plt.xlabel('Method')
    plt.ylabel('Calinski-Harabasz Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_calinski_harabasz_score.png')
    plt.show()

    # 可视化 FMI Score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='method', y='fmi_score', data=data, palette='viridis')
    plt.title(f'FMI Score by Method - {dataset_name.capitalize()}')
    plt.xlabel('Method')
    plt.ylabel('FMI Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_fmi_score.png')
    plt.show()

    # 可视化迭代次数
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='method', y='num_iterations', data=data, palette='viridis')
    # plt.title(f'Number of Iterations by Method - {dataset_name.capitalize()}')
    # plt.xlabel('Method')
    # plt.ylabel('Number of Iterations')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f'{dataset_name}_num_iterations.png')
    # plt.show()
