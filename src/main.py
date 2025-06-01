import sys
import os # 新增导入
import io

# 设置控制台输出编码为UTF-8，解决中文乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from PyQt5.QtWidgets import QApplication # QMainWindow is imported by MainWindow

# 新增以下几行代码，用于修正包的导入路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 导入其他模块（将在后续步骤中创建）
from src.ui.main_window import MainWindow # 修正导入路径，从相对导入改为绝对导入

class Application(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        # 在这里可以进行应用的全局初始化设置

if __name__ == '__main__':
    app = Application(sys.argv)
    
    # 创建主窗口
    # 注意：MainWindow 类将在 ui/main_window.py 中定义
    main_win = MainWindow() 
    main_win.show()
    
    # 临时的占位窗口，直到 MainWindow 完成
    # main_win = QMainWindow()
    # main_win.setWindowTitle("微信表情包开发神器")
    # main_win.setGeometry(100, 100, 800, 600)
    # main_win.show()
    
    sys.exit(app.exec_()) 