import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from ..utils import config

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setFixedSize(480, 550)
        
        layout = QVBoxLayout(self)
        
        # 应用标题
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        
        title_label = QLabel(f"{config.APP_NAME}")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        
        # 版本标签
        version_font = QFont()
        version_font.setPointSize(11)
        
        version_label = QLabel(f"版本 {config.APP_VERSION} - 内测版")
        version_label.setFont(version_font)
        version_label.setAlignment(Qt.AlignCenter)
        
        # 加载LOGO
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LOGO.jpg"))
        if not logo_pixmap.isNull():
            # 调整尺寸确保LOGO完整显示，宽度足够显示标题文字
            logo_pixmap = logo_pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setMinimumWidth(400)  # 设置最小宽度确保不被压缩
        
        # 作者信息
        info_font = QFont()
        info_font.setPointSize(11)
        
        author_label = QLabel("本软件由 <span style='font-weight:bold; font-size:12pt;'>TONY老师</span> 精心制作")
        author_label.setFont(info_font)
        author_label.setAlignment(Qt.AlignCenter)
        
        # 微信号信息
        wechat_label = QLabel("欢迎关注微信视频号：")
        wechat_label.setFont(info_font)
        wechat_label.setAlignment(Qt.AlignCenter)
        
        channel_font = QFont()
        channel_font.setPointSize(14)
        channel_font.setBold(True)
        
        channel_label = QLabel("TONY老师教AI")
        channel_label.setFont(channel_font)
        channel_label.setAlignment(Qt.AlignCenter)
        channel_label.setStyleSheet("color: #0078D7;") # 使用微信蓝色
        
        # 版权信息
        copyright_label = QLabel("版权所有 © 2024 保留所有权利")
        copyright_label.setAlignment(Qt.AlignCenter)
        
        # 说明文本
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setHtml(
            "<p style='text-align:center'><b>微信表情包制作工具</b></p>"
            "<p>这是一款专为微信表情包开发者打造的工具，能够将视频转换为符合微信表情包规范的9宫格GIF动画。</p>"
            "<p><b>主要功能：</b></p>"
            "<ul>"
            "<li>支持MP4视频导入</li>"
            "<li>交互式九宫格半自动调整</li>"
            "<li>符合微信表情规范的GIF导出</li>"
            "<li>智能优化（大小控制、帧率自适应）</li>"
            "</ul>"
            "<p><b>技术支持：</b>如有问题，请通过微信视频号联系作者</p>"
        )
        desc_text.setMaximumHeight(200)
        
        # 添加所有组件到布局
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addSpacing(10)
        layout.addWidget(logo_label)
        layout.addSpacing(10)
        layout.addWidget(author_label)
        layout.addWidget(wechat_label)
        layout.addWidget(channel_label)
        layout.addSpacing(5)
        layout.addWidget(desc_text)
        layout.addSpacing(5)
        layout.addWidget(copyright_label)
        
        # 关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box) 