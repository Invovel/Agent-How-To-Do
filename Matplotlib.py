# Matplotlib 基础操作教程
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------- 1. 基本设置 ----------------------
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 2. 基本图表类型 ----------------------

# 2.1 折线图
def line_chart():
    print("2.1 折线图示例")
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2, marker='o', markersize=3)
    plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2, marker='s', markersize=3)
    
    plt.title('正弦和余弦函数曲线', fontsize=15)  # 标题
    plt.xlabel('x轴', fontsize=12)  # x轴标签
    plt.ylabel('y轴', fontsize=12)  # y轴标签
    plt.grid(True, linestyle='--', alpha=0.7)  # 网格线
    plt.legend(fontsize=10, loc='upper right')  # 图例
    
    plt.xlim(0, 10)  # x轴范围
    plt.ylim(-1.2, 1.2)  # y轴范围
    
    plt.savefig('line_chart.png', dpi=300, bbox_inches='tight')  # 保存图表
    plt.show()

# 2.2 散点图
def scatter_chart():
    print("\n2.2 散点图示例")
    x = np.random.randn(100)
    y = np.random.randn(100)
    sizes = np.random.randint(10, 100, 100)
    colors = np.random.randn(100)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7, cmap='viridis', edgecolors='black')
    
    plt.title('随机散点图', fontsize=15)
    plt.xlabel('x轴', fontsize=12)
    plt.ylabel('y轴', fontsize=12)
    plt.colorbar(label='颜色值')  # 颜色条
    plt.grid(True, alpha=0.5)
    
    plt.savefig('scatter_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.3 柱状图
def bar_chart():
    print("\n2.3 柱状图示例")
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [20, 35, 30, 45, 38]
    values2 = [25, 32, 34, 20, 28]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, values1, width, label='系列1', color='skyblue')
    plt.bar(x + width/2, values2, width, label='系列2', color='salmon')
    
    plt.title('分组柱状图', fontsize=15)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.xticks(x, categories, fontsize=10)
    plt.legend()
    plt.grid(axis='y', alpha=0.7)
    
    plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.4 直方图
def histogram_chart():
    print("\n2.4 直方图示例")
    data = np.random.randn(1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    
    # 添加正态分布曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    from scipy.stats import norm
    y = norm.pdf(x, 0, 1)
    plt.plot(x, y, 'r--', linewidth=2)
    
    plt.title('数据分布直方图', fontsize=15)
    plt.xlabel('数值', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    
    plt.savefig('histogram_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.5 饼图
def pie_chart():
    print("\n2.5 饼图示例")
    labels = ['A', 'B', 'C', 'D', 'E']
    sizes = [15, 30, 25, 10, 20]
    explode = (0, 0.1, 0, 0, 0)  # 突出显示第二个部分
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    
    plt.title('饼图示例', fontsize=15)
    plt.axis('equal')  # 保持圆形
    
    plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.6 箱线图
def box_chart():
    print("\n2.6 箱线图示例")
    data = [np.random.normal(0, std, 100) for std in range(1, 6)]
    
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, labels=['组1', '组2', '组3', '组4', '组5'], patch_artist=True)
    
    # 设置箱子颜色
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('箱线图示例', fontsize=15)
    plt.xlabel('组别', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    
    plt.savefig('box_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.7 热力图
def heatmap_chart():
    print("\n2.7 热力图示例")
    
    # 示例1: 基础热力图
    print("\n2.7.1 基础热力图")
    # 创建随机数据矩阵
    np.random.seed(42)  # 设置随机种子，确保结果可复现
    data = np.random.rand(10, 10)  # 10x10的随机数据
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, cmap='viridis')  # 使用viridis颜色映射
    
    # 添加颜色条
    plt.colorbar(im, label='数值大小')
    
    # 设置标题和标签
    plt.title('基础热力图示例', fontsize=15)
    plt.xlabel('X轴', fontsize=12)
    plt.ylabel('Y轴', fontsize=12)
    
    # 保存图表
    plt.savefig('basic_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 示例2: 带数值标签的热力图
    print("\n2.7.2 带数值标签的热力图")
    # 创建相关系数矩阵
    corr_matrix = np.corrcoef(np.random.randn(10, 10))
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)  # 使用coolwarm颜色映射，范围-1到1
    
    # 添加颜色条
    plt.colorbar(im, label='相关系数')
    
    # 设置标题
    plt.title('相关系数矩阵热力图', fontsize=15)
    
    # 添加数值标签
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     ha='center', va='center', color='white' if np.abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    # 保存图表
    plt.savefig('labeled_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 示例3: 使用Seaborn创建更美观的热力图（需要安装seaborn）
    try:
        import seaborn as sns
        print("\n2.7.3 使用Seaborn的热力图")
        
        # 创建示例数据
        data = pd.DataFrame(np.random.rand(10, 5), columns=[f'特征{i}' for i in range(1, 6)],
                           index=[f'样本{i}' for i in range(1, 11)])
        
        plt.figure(figsize=(12, 10))
        # 使用seaborn的heatmap函数
        sns.heatmap(data, annot=True, cmap='YlGnBu', linewidths=0.5, fmt='.2f')
        
        plt.title('Seaborn热力图示例', fontsize=15)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('seaborn_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSeaborn热力图示例已完成。若未显示，请确保已安装seaborn库：pip install seaborn")
    except ImportError:
        print("\n提示：Seaborn库未安装，无法显示Seaborn热力图示例。")
        print("可通过以下命令安装：pip install seaborn")

# ---------------------- 3. 多子图绘制 ----------------------
def subplots_example():
    print("\n3. 多子图示例")
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2*x)
    y4 = np.cos(2*x)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2行2列子图
    
    # 第一个子图
    axes[0, 0].plot(x, y1, color='blue')
    axes[0, 0].set_title('sin(x)')
    axes[0, 0].grid(True)
    
    # 第二个子图
    axes[0, 1].plot(x, y2, color='red')
    axes[0, 1].set_title('cos(x)')
    axes[0, 1].grid(True)
    
    # 第三个子图
    axes[1, 0].plot(x, y3, color='green')
    axes[1, 0].set_title('sin(2x)')
    axes[1, 0].grid(True)
    
    # 第四个子图
    axes[1, 1].plot(x, y4, color='purple')
    axes[1, 1].set_title('cos(2x)')
    axes[1, 1].grid(True)
    
    # 调整子图间距
    plt.tight_layout()
    plt.suptitle('多子图示例', fontsize=18, y=1.02)
    
    plt.savefig('subplots_example.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- 4. 与Pandas结合 ----------------------
def pandas_integration():
    print("\n4. 与Pandas结合示例")
    # 创建示例数据
    data = {
        '月份': ['1月', '2月', '3月', '4月', '5月', '6月'],
        '销售额': [20000, 25000, 30000, 35000, 38000, 42000],
        '利润': [2000, 2500, 3000, 3500, 3800, 4200]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    # 柱状图显示销售额
    plt.bar(df['月份'], df['销售额'], color='lightblue', label='销售额')
    
    # 折线图显示利润
    plt.twinx()  # 创建第二个y轴
    plt.plot(df['月份'], df['利润'], color='red', marker='o', label='利润')
    
    plt.title('销售额与利润趋势', fontsize=15)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('销售额(元)', fontsize=12)
    plt.ylabel('利润(元)', fontsize=12)
    
    # 组合图例
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = plt.gca().get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.grid(True, alpha=0.5)
    plt.savefig('pandas_integration.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- 5. 3D图表 ----------------------
def three_d_chart():
    print("\n5. 3D图表示例")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_title('3D曲面图', fontsize=15)
    ax.set_xlabel('X轴', fontsize=12)
    ax.set_ylabel('Y轴', fontsize=12)
    ax.set_zlabel('Z轴', fontsize=12)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)  # 颜色条
    
    plt.savefig('3d_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- 主函数 ----------------------
def main():
    print("Matplotlib 基础操作教程")
    print("=" * 50)
    
    # 依次运行所有示例
    line_chart()
    scatter_chart()
    bar_chart()
    histogram_chart()
    pie_chart()
    box_chart()
    heatmap_chart()  # 添加热力图示例
    subplots_example()
    pandas_integration()
    three_d_chart()
    
    print("\n" + "=" * 50)
    print("所有示例已运行完成，图表已保存到当前目录！")

if __name__ == "__main__":
    main()