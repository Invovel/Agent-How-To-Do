# Matplotlib 诊断脚本
import sys
import os

def print_info(title, content):
    print(f"\n{title}:")
    print(f"{content}")

# 1. 检查Python版本
print_info("Python版本", sys.version)

# 2. 检查Matplotlib是否安装
try:
    import matplotlib
    print_info("Matplotlib版本", matplotlib.__version__)
    print_info("Matplotlib安装路径", matplotlib.__file__)
    
    # 3. 检查后端配置
    print_info("默认后端", matplotlib.get_backend())
    
    # 4. 检查可用后端
    print_info("可用后端", matplotlib.rcsetup.all_backends)
    
    # 5. 测试基本功能
    try:
        import numpy as np
        print_info("Numpy版本", np.__version__)
        
        # 创建简单图形并保存为文件
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # 显式设置后端
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(x, y)
        plt.title('测试图形')
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        
        # 保存图形到文件
        save_path = os.path.join(os.getcwd(), 'test_plot.png')
        plt.savefig(save_path)
        plt.close()
        
        if os.path.exists(save_path):
            print_info("图形保存", f"成功！文件路径：{save_path}")
        else:
            print_info("图形保存", "失败！")
            
    except Exception as e:
        print_info("绘图测试错误", f"{type(e).__name__}: {e}")
        
except ImportError as e:
    print_info("Matplotlib安装检查", f"失败！{e}")
    print("\n请尝试重新安装Matplotlib:")
    print("pip install matplotlib --upgrade --force-reinstall")
except Exception as e:
    print_info("意外错误", f"{type(e).__name__}: {e}")

print("\n诊断完成！")
