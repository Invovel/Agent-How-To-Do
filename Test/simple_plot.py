# 极简Matplotlib测试脚本
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='正弦曲线')
plt.title('简单测试图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()
plt.grid(True)

# 保存图形
plt.savefig('simple_test.png')
print("图形已保存为 simple_test.png")

# 尝试不同的显示方式
try:
    # 方法1：直接显示
    plt.show()
except Exception as e:
    print(f"直接显示失败: {e}")
    
    # 方法2：使用阻塞模式显示
    try:
        plt.show(block=True)
        print("使用block=True成功显示")
    except Exception as e2:
        print(f"block=True也失败: {e2}")
