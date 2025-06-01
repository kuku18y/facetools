import os
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QScrollArea, QGridLayout, QSizePolicy, QFrame, QSplitter,
    QMenuBar, QAction, QStatusBar, QProgressDialog, QMessageBox, QSlider,
    QSpinBox, QCheckBox, QGroupBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsItem, QToolTip, QStyle, QApplication, QMenu,
    QGraphicsTextItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QFont, QTransform
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QSize, QPoint, QTimer

from ..core.video import VideoProcessor
from ..core.image import ImageOptimizer
from ..core.gif import GifGenerator
from ..utils import config
from .about_dialog import AboutDialog

class MainWindow(QMainWindow):
    """应用程序的主窗口。"""
    
    # 定义信号
    progress_signal = pyqtSignal(int, str) # value, message
    processing_finished_signal = pyqtSignal(bool, str) # success, message_or_path

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.APP_NAME} v{config.APP_VERSION} - 内测版")
        self.setGeometry(100, 100, 1200, 800) # 初始窗口大小
        self.setMinimumSize(800, 600)

        self.video_processor = None
        self.video_frames_rgb = [] # 存储从视频中提取的原始RGB帧
        self.current_video_path = None
        self.grid_items_data = [] # 存储每个九宫格单元的配置数据
        self.preview_frames_for_gif = {} # key: grid_id, value: list of QPixmap for preview

        self._init_ui()
        self._connect_signals()
        
        # 为处理线程和进度条做准备
        self.processing_thread = None
        self.progress_dialog = None
        self.progress_signal.connect(self.update_progress_dialog)
        self.processing_finished_signal.connect(self.on_processing_finished)

    def _init_ui(self):
        """初始化用户界面组件。"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # 1. 菜单栏
        self._create_menu_bar()

        # 2. 主体内容区 (使用QSplitter分隔)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # 2.1 左侧面板: 视频预览、操作说明和LOGO
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.splitter.addWidget(left_panel)
        
        # 确保左侧面板宽度足够
        left_panel.setMinimumWidth(350)

        # 视频导入按钮
        self.btn_load_video = QPushButton("导入视频 (.mp4)")
        self.btn_load_video.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        left_layout.addWidget(self.btn_load_video)

        # 视频预览区 (使用QGraphicsView以支持缩放和平移)
        self.video_preview_view = QGraphicsView()
        self.video_preview_scene = QGraphicsScene(self)
        self.video_preview_view.setScene(self.video_preview_scene)
        self.video_preview_view.setRenderHint(QPainter.Antialiasing)
        self.video_preview_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.video_preview_view.setMinimumHeight(200)
        self.video_preview_item = QGraphicsPixmapItem()
        self.video_preview_scene.addItem(self.video_preview_item)
        left_layout.addWidget(self.video_preview_view)
        
        # 添加操作说明
        instruction_group = QGroupBox("操作说明")
        instruction_layout = QVBoxLayout(instruction_group)
        
        # 使用HTML格式化操作说明，使用表格实现更专业的排版
        instructions_text = QLabel(
            "<style>"
            "table { width: 100%; border-collapse: collapse; }"
            "td { padding: 5px; vertical-align: top; }"
            "td.num { font-weight: bold; text-align: right; width: 30px; }"
            "td.step { font-weight: bold; }"
            "ul { margin: 5px 0 5px 0; padding-left: 20px; }"
            "li { margin-bottom: 3px; }"
            ".note { color: #E74C3C; font-weight: bold; font-size: 12pt; margin-top: 10px; padding: 5px; border-radius: 3px; }"
            "</style>"
            "<table>"
            "<tr><td class='num'>1.</td><td class='step'>导入视频：</td><td>点击上方按钮选择MP4视频文件</td></tr>"
            "<tr><td class='num'>2.</td><td class='step'>半自动调整：</td><td>在右侧九宫格中，通过以下方式调整每个区块：</td></tr>"
            "</table>"
            "<ul>"
            "<li><b>拖动图像</b>：移动图像内容到红框中</li>"
            "<li><b>鼠标滚轮</b>：放大或缩小图像</li>"
            "<li><b>右键菜单</b>：重置视图</li>"
            "</ul>"
            "<table>"
            "<tr><td class='num'>3.</td><td class='step'>导出GIF：</td><td>设置帧率后点击导出按钮</td></tr>"
            "</table>"
            "<div class='note'>注意：红框内的内容即为最终表情的效果</div>"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setTextFormat(Qt.RichText)
        instruction_layout.addWidget(instructions_text)
        left_layout.addWidget(instruction_group)

        # 添加LOGO和作者信息，使用更美观的样式
        logo_group = QGroupBox("关于")
        logo_layout = QVBoxLayout(logo_group)
        
        # 加载LOGO - 确保完全显示LOGO和标题文字
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "LOGO.jpg"))
        if not logo_pixmap.isNull():
            # 大幅增加宽度，确保LOGO和红字标题完全显示
            logo_pixmap = logo_pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setMinimumWidth(320)  # 设置最小宽度确保不被压缩
        
        # 设置标题和版本信息
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        
        app_title = QLabel(f"{config.APP_NAME}")
        app_title.setFont(title_font)
        app_title.setAlignment(Qt.AlignCenter)
        
        version_font = QFont()
        version_font.setPointSize(9)
        
        version_label = QLabel(f"v{config.APP_VERSION} - 内测版")
        version_label.setFont(version_font)
        version_label.setAlignment(Qt.AlignCenter)
        
        author_font = QFont()
        author_font.setPointSize(9)
        
        author_label = QLabel("本软件由<b>TONY老师</b>制作")
        author_label.setFont(author_font)
        author_label.setAlignment(Qt.AlignCenter)
        
        follow_label = QLabel("欢迎关注微信视频号：")
        follow_label.setFont(author_font)
        follow_label.setAlignment(Qt.AlignCenter)
        
        channel_font = QFont()
        channel_font.setPointSize(10)
        channel_font.setBold(True)
        
        channel_label = QLabel("TONY老师教AI")
        channel_label.setFont(channel_font)
        channel_label.setAlignment(Qt.AlignCenter)
        channel_label.setStyleSheet("color: #0078D7;") # 使用微信蓝色
        
        logo_layout.addWidget(logo_label)
        logo_layout.addWidget(app_title)
        logo_layout.addWidget(version_label)
        logo_layout.addSpacing(5)
        logo_layout.addWidget(author_label)
        logo_layout.addWidget(follow_label)
        logo_layout.addWidget(channel_label)
        logo_layout.addStretch()
        
        # 调整splitter比例，确保左侧区域有足够空间显示LOGO
        left_layout.addWidget(logo_group)

        # 2.2 右侧面板: 九宫格编辑区和导出控制
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.splitter.addWidget(right_panel)

        # 九宫格编辑区 (使用QScrollArea)
        self.scroll_area_grid = QScrollArea()
        self.scroll_area_grid.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget) # 将在加载视频后填充
        self.grid_layout.setSpacing(5)
        self.scroll_area_grid.setWidget(self.grid_widget)
        right_layout.addWidget(self.scroll_area_grid, 1) # 占据更多空间

        # 导出控制区
        export_group = QGroupBox("导出设置与操作")
        export_layout = QVBoxLayout(export_group)
        
        # 添加帧率滑块
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)
        self.fps_spinbox.setValue(config.DEFAULT_GIF_FPS)
        self.fps_spinbox.setSuffix(" FPS")
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("目标帧率:"))
        fps_layout.addWidget(self.fps_spinbox)
        export_layout.addLayout(fps_layout)

        self.btn_export_all_gifs = QPushButton("导出所有GIF")
        self.btn_export_all_gifs.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_export_all_gifs.setEnabled(False)
        export_layout.addWidget(self.btn_export_all_gifs)
        right_layout.addWidget(export_group)

        # 初始时，调整分割比例，让左侧有足够空间显示LOGO
        self.splitter.setSizes([400, self.width() - 400])

        # 3. 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("请先导入视频文件")

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        # 文件菜单
        file_menu = menu_bar.addMenu("&文件")
        load_action = QAction(QIcon.fromTheme("document-open", self.style().standardIcon(QStyle.SP_DialogOpenButton)), "导入视频...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_video_dialog)
        file_menu.addAction(load_action)

        exit_action = QAction(QIcon.fromTheme("application-exit", self.style().standardIcon(QStyle.SP_DialogCloseButton)), "退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menu_bar.addMenu("&帮助")
        about_action = QAction("关于...", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    def _connect_signals(self):
        self.btn_load_video.clicked.connect(self.load_video_dialog)
        self.btn_export_all_gifs.clicked.connect(self.export_all_gifs_action)
        # self.fps_spinbox.valueChanged.connect(self._update_gif_previews_if_needed) # 预留，如果需要实时更新预览

    def load_video_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择MP4视频文件", "", "MP4视频文件 (*.mp4);;所有文件 (*)")
        if file_path:
            try:
                self.status_bar.showMessage(f"正在加载视频: {os.path.basename(file_path)}...")
                QApplication.processEvents() # 确保状态栏更新
                if self.video_processor:
                    self.video_processor.release()
                
                self.video_processor = VideoProcessor(file_path)
                self.current_video_path = file_path
                self.status_bar.showMessage(f"视频已加载: {os.path.basename(file_path)} - {self.video_processor.width}x{self.video_processor.height} @ {self.video_processor.original_fps:.2f} FPS, 时长: {self.video_processor.duration:.2f}s")
                
                # 显示第一帧作为预览
                first_frame_rgb = self.video_processor.get_frame_at_index(0)
                if first_frame_rgb is not None:
                    pixmap = VideoProcessor.frame_to_qpixmap(first_frame_rgb)
                    self.video_preview_item.setPixmap(pixmap)
                    self.video_preview_view.setSceneRect(self.video_preview_item.boundingRect())
                    self.video_preview_view.fitInView(self.video_preview_item, Qt.KeepAspectRatio)
                
                # 初始化九宫格数据和UI
                self._setup_grid_editor()
                self.btn_export_all_gifs.setEnabled(True)
                self.status_bar.showMessage(f"视频加载完成: {os.path.basename(file_path)}. 请在右侧编辑各区块。", 5000)

            except FileNotFoundError as e:
                QMessageBox.critical(self, "错误", f"无法找到视频文件: {e}")
                self.status_bar.showMessage("视频加载失败", 5000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载视频时发生错误: {e}")
                self.status_bar.showMessage("视频加载失败", 5000)
                if self.video_processor: self.video_processor.release()
                self.video_processor = None
                self.current_video_path = None
                self.btn_export_all_gifs.setEnabled(False)

    def _setup_grid_editor(self):
        """根据加载的视频初始化或重置九宫格编辑器。"""
        if not self.video_processor: return

        # 清理旧的网格项 (如果存在)
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.grid_items_data.clear()
        self.preview_frames_for_gif.clear()

        # 创建九宫格位置数据，每个区块初始时都显示完整的视频帧
        initial_grid_positions = self.video_processor.create_grid_positions(
            rows=config.DEFAULT_GRID_ROWS, 
            cols=config.DEFAULT_GRID_COLS
        )
        self.grid_items_data = initial_grid_positions # 存储每个格子的数据

        # 提取少量帧用于快速预览
        # 这里的帧提取仅用于UI显示，实际GIF生成时会重新提取
        preview_duration = min(2.0, self.video_processor.duration) # 最多预览2秒
        thumb_frames_rgb, thumb_fps = self.video_processor.extract_frames(target_fps=5, end_time=preview_duration)

        for i, grid_data in enumerate(self.grid_items_data):
            row = i // config.DEFAULT_GRID_COLS
            col = i % config.DEFAULT_GRID_COLS
            
            # 创建单个网格单元的UI
            grid_cell_widget = QFrame()
            grid_cell_widget.setFrameShape(QFrame.StyledPanel)
            grid_cell_layout = QVBoxLayout(grid_cell_widget)
            
            # 使用QGraphicsView以支持交互
            cell_view = EditableGraphicsView(grid_id=grid_data['id'], parent_window=self)
            cell_view.setMinimumSize(config.WECHAT_STICKER_WIDTH + 20, config.WECHAT_STICKER_HEIGHT + 20) # 留出边框和操作空间
            cell_view.crop_rect_updated.connect(self.update_grid_cell_preview)
            
            grid_cell_layout.addWidget(QLabel(f"区块 {grid_data['id'] + 1}"))
            grid_cell_layout.addWidget(cell_view)
            
            self.grid_layout.addWidget(grid_cell_widget, row, col)

            # 为该单元设置初始预览 (使用完整的第一帧)
            if thumb_frames_rgb and len(thumb_frames_rgb) > 0: # 首先确保有预览帧
                # 使用完整的视频帧作为初始预览
                pixmap = VideoProcessor.frame_to_qpixmap(thumb_frames_rgb[0])
                if not pixmap.isNull():
                    cell_view.set_initial_pixmap(pixmap)
                else:
                    print(f"警告：区块 {grid_data['id'] + 1} 的预览图像为空")
                    placeholder = QPixmap(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT)
                    placeholder.fill(Qt.lightGray)
                    cell_view.set_initial_pixmap(placeholder)
            else: # 无视频帧时 (例如视频加载失败或视频太短)
                print(f"警告：没有可用的预览帧")
                placeholder = QPixmap(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT)
                placeholder.fill(Qt.lightGray)
                cell_view.set_initial_pixmap(placeholder)
        
        # 更新滚动区域的布局
        self.grid_widget.adjustSize()
        self.scroll_area_grid.setWidget(self.grid_widget)

    def update_grid_cell_preview(self, grid_id, updated_crop_rect):
        """当用户在EditableGraphicsView中修改了裁剪区域后，更新该单元的预览"""
        # 使用第一帧作为预览源
        preview_source_frame = None
        if self.video_processor:
            preview_source_frame = self.video_processor.get_frame_at_index(0)

        if preview_source_frame is None:
            print(f"更新网格单元 {grid_id} 预览失败：无可用源视频帧")
            return

        # 更新 self.grid_items_data 中的对应项
        for item_data in self.grid_items_data:
            if item_data['id'] == grid_id:
                # 只更新裁剪相关的参数
                for key in ['source_x', 'source_y', 'source_width', 'source_height', 'scale']:
                    if key in updated_crop_rect:
                        item_data[key] = updated_crop_rect[key]
                break
        
        # 无需更新UI，因为在新的EditableGraphicsView实现中，
        # 用户能够直接在界面上看到实时预览效果
        # 所有的视觉更新都在EditableGraphicsView类的内部处理

    def export_all_gifs_action(self):
        if not self.video_processor or not self.grid_items_data:
            QMessageBox.warning(self, "无法导出", "请先加载视频并设置九宫格。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择GIF导出文件夹")
        if not output_dir:
            return
            
        self.progress_dialog = QProgressDialog("正在导出GIF...", "取消", 0, len(self.grid_items_data), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False) # 手动关闭
        self.progress_dialog.setAutoReset(False) # 手动重置
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        QApplication.processEvents()

        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "提示", "当前有导出任务正在进行中。")
            return

        # 确保所有grid_items_data中的数据都是最新的
        # 在新的EditableGraphicsView实现中，所有更新都已经在update_crop_params方法中处理
        # 所以这里不需要额外处理
        
        self.processing_thread = GifExportThread(
            video_path=self.current_video_path,
            grid_configs=self.grid_items_data, # 传递每个格子的裁剪配置
            output_dir=output_dir,
            target_fps=self.fps_spinbox.value(),
            parent_signal_progress=self.progress_signal,
            parent_signal_finished=self.processing_finished_signal
        )
        self.progress_dialog.canceled.connect(self.processing_thread.terminate_processing) # 允许用户取消
        self.processing_thread.start()

    def update_progress_dialog(self, value, message):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()

    def on_processing_finished(self, success, message_or_path):
        if self.progress_dialog:
            self.progress_dialog.setValue(self.progress_dialog.maximum()) # 完成进度条
            self.progress_dialog.hide()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None
        
        if success:
            QMessageBox.information(self, "导出完成", f"所有GIF已成功导出到: {message_or_path}")
            self.status_bar.showMessage(f"导出完成到 {message_or_path}", 5000)
            # 可选：打开导出文件夹
            # QDesktopServices.openUrl(QUrl.fromLocalFile(message_or_path))
        else:
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误: {message_or_path}")
            self.status_bar.showMessage("导出失败", 5000)

    def show_about_dialog(self):
        about_dialog = AboutDialog(self)
        about_dialog.exec_()

    def closeEvent(self, event):
        if self.video_processor:
            self.video_processor.release()
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, '退出确认', 
                                       "有导出任务正在进行中，确定要退出吗？",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.processing_thread.terminate_processing() # 尝试停止线程
                self.processing_thread.wait(1000) # 等待一小会
                event.accept()
            else:
                event.ignore()
                return
        super().closeEvent(event)


class EditableGraphicsView(QWidget):
    """简化版的可编辑视图，使用QLabel显示图像和红框"""
    # 信号：当裁剪区域通过用户交互（缩放/拖动）更新时发出
    crop_rect_updated = pyqtSignal(int, dict)

    def __init__(self, grid_id, parent_window=None):
        super().__init__(parent_window)
        self.grid_id = grid_id
        self.parent_window = parent_window
        
        # 设置固定大小
        self.setMinimumSize(config.WECHAT_STICKER_WIDTH + 20, config.WECHAT_STICKER_HEIGHT + 20)
        
        # 创建用于显示图像的标签
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # 创建用于显示裁剪框的标签
        self.crop_frame = QLabel(self)
        self.crop_frame.setFixedSize(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT)
        self.crop_frame.setStyleSheet("border: 2px solid red; background-color: transparent;")
        
        # 放置裁剪框在中心
        self.crop_frame.move(10, 10)  # 留出10像素边距
        
        # 状态变量
        self.original_pixmap = None  # 原始图像
        self.current_pixmap = None   # 当前显示的图像
        self.scale_factor = 1.0      # 当前缩放比例
        self.offset_x = 0            # 图像X偏移
        self.offset_y = 0            # 图像Y偏移
        self.dragging = False        # 是否正在拖动
        self.last_mouse_pos = None   # 上次鼠标位置
        self.is_selected = False     # 是否被选中
        
        # 右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # 设置鼠标追踪，用于显示提示
        self.setMouseTracking(True)
        
        # 显示帮助提示
        QTimer.singleShot(1000, self.show_help_briefly)

    def show_help_briefly(self):
        """短暂显示帮助提示"""
        if self.isVisible():
            QToolTip.showText(self.mapToGlobal(self.rect().center()), 
                              "点击选中后再拖动或缩放", self, self.rect(), 3000)
    
    def set_initial_pixmap(self, pixmap):
        """设置初始图像"""
        if pixmap.isNull():
            # 如果是空图像，显示灰色占位符
            placeholder = QPixmap(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT)
            placeholder.fill(Qt.lightGray)
            pixmap = placeholder
            
        # 保存原始图像
        self.original_pixmap = pixmap
        
        # 复位图像位置和缩放
        self.reset_view()
        
    def update_display(self):
        """更新图像显示"""
        if self.original_pixmap is None:
            return
            
        # 创建一个大于裁剪框的画布
        canvas_size = max(self.width(), self.height()) * 2
        canvas = QPixmap(canvas_size, canvas_size)
        canvas.fill(Qt.white)
        
        # 计算缩放后的图像大小
        scaled_width = int(self.original_pixmap.width() * self.scale_factor)
        scaled_height = int(self.original_pixmap.height() * self.scale_factor)
        
        # 缩放图像
        scaled_pixmap = self.original_pixmap.scaled(scaled_width, scaled_height, 
                                                    Qt.KeepAspectRatio, 
                                                    Qt.SmoothTransformation)
        
        # 计算图像在画布上的位置（相对于裁剪框中心）
        center_x = canvas_size // 2 + self.offset_x
        center_y = canvas_size // 2 + self.offset_y
        img_x = center_x - scaled_pixmap.width() // 2
        img_y = center_y - scaled_pixmap.height() // 2
        
        # 在画布上绘制图像
        painter = QPainter(canvas)
        painter.drawPixmap(img_x, img_y, scaled_pixmap)
        painter.end()
        
        # 从画布中裁剪出裁剪框区域作为当前显示图像
        crop_x = canvas_size // 2 - self.crop_frame.width() // 2
        crop_y = canvas_size // 2 - self.crop_frame.height() // 2
        cropped = canvas.copy(crop_x, crop_y, 
                             self.crop_frame.width(), 
                             self.crop_frame.height())
        
        # 更新图像标签
        self.image_label.setPixmap(cropped)
        self.image_label.setFixedSize(cropped.size())
        self.image_label.move(10, 10)  # 与裁剪框位置相同
        
        # 保存当前显示的图像
        self.current_pixmap = cropped
        
        # 更新裁剪参数
        self.update_crop_params()
        
        # 更新选中状态的视觉反馈
        self.update_selection_style()
        
    def update_selection_style(self):
        """更新选中状态的视觉样式"""
        if self.is_selected:
            self.crop_frame.setStyleSheet("border: 3px solid #2196F3; background-color: transparent;")
            self.setStyleSheet("background-color: #E3F2FD;")
        else:
            self.crop_frame.setStyleSheet("border: 2px solid red; background-color: transparent;")
            self.setStyleSheet("")
        
    def wheelEvent(self, event):
        """处理鼠标滚轮事件（缩放）"""
        if self.original_pixmap is None or not self.is_selected:
            return
            
        # 计算缩放因子
        delta = event.angleDelta().y()
        if delta > 0:
            # 放大
            self.scale_factor *= 1.1
        else:
            # 缩小
            self.scale_factor *= 0.9
            
        # 限制缩放范围
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))
        
        # 更新显示
        self.update_display()
        
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        # 首先处理选中状态
        if event.button() == Qt.LeftButton:
            # 取消其他所有单元格的选中状态
            if self.parent_window:
                for row in range(self.parent_window.grid_layout.rowCount()):
                    for col in range(self.parent_window.grid_layout.columnCount()):
                        item = self.parent_window.grid_layout.itemAtPosition(row, col)
                        if item and item.widget():
                            # 找到EditableGraphicsView实例
                            for child in item.widget().findChildren(EditableGraphicsView):
                                if child != self:
                                    child.is_selected = False
                                    child.update_selection_style()
            
            # 设置当前单元格为选中状态
            self.is_selected = True
            self.update_selection_style()
            
            # 如果已经选中，则启动拖动
            if self.is_selected:
                self.dragging = True
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.dragging and self.last_mouse_pos and self.is_selected:
            # 计算移动距离
            delta_x = event.x() - self.last_mouse_pos.x()
            delta_y = event.y() - self.last_mouse_pos.y()
            
            # 更新偏移
            self.offset_x += delta_x
            self.offset_y += delta_y
            
            # 更新鼠标位置
            self.last_mouse_pos = event.pos()
            
            # 更新显示
            self.update_display()
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            
    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu(self)
        reset_action = menu.addAction("重置视图")
        reset_action.triggered.connect(self.reset_view)
        menu.exec_(self.mapToGlobal(position))
        
    def reset_view(self):
        """重置视图到初始状态"""
        if self.original_pixmap is None:
            return
            
        # 计算适合的初始缩放比例
        view_width = self.crop_frame.width()
        view_height = self.crop_frame.height()
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        
        # 计算适合视图的缩放比例，保持图像完整显示
        scale_x = view_width / img_width
        scale_y = view_height / img_height
        self.scale_factor = min(scale_x, scale_y) * 0.9  # 稍微缩小一点留出边距
        
        # 重置偏移
        self.offset_x = 0
        self.offset_y = 0
        
        # 更新显示
        self.update_display()
        
    def update_crop_params(self):
        """更新裁剪参数并发出信号"""
        if not self.parent_window or not self.parent_window.video_processor or self.original_pixmap is None:
            return
            
        # 获取原始视频尺寸
        video_width = self.parent_window.video_processor.width
        video_height = self.parent_window.video_processor.height
        
        # 计算当前图像中心相对于原始图像的位置
        # 裁剪框固定在中心，所以我们实际上是移动图像
        center_offset_x = -self.offset_x / self.scale_factor
        center_offset_y = -self.offset_y / self.scale_factor
        
        # 计算裁剪区域在原始图像中的大小
        crop_width = self.crop_frame.width() / self.scale_factor
        crop_height = self.crop_frame.height() / self.scale_factor
        
        # 计算裁剪区域左上角在原始图像中的位置
        # 中心点偏移加上半个裁剪框大小
        crop_x = (self.original_pixmap.width() / 2) + center_offset_x - (crop_width / 2)
        crop_y = (self.original_pixmap.height() / 2) + center_offset_y - (crop_height / 2)
        
        # 计算原始视频到显示图像的比例
        scale_to_video_x = video_width / self.original_pixmap.width()
        scale_to_video_y = video_height / self.original_pixmap.height()
        
        # 计算裁剪区域在原始视频中的位置和大小
        source_x = crop_x * scale_to_video_x
        source_y = crop_y * scale_to_video_y
        source_width = crop_width * scale_to_video_x
        source_height = crop_height * scale_to_video_y
        
        # 边界检查
        source_x = max(0, min(source_x, video_width - 1))
        source_y = max(0, min(source_y, video_height - 1))
        source_width = max(1, min(source_width, video_width - source_x))
        source_height = max(1, min(source_height, video_height - source_y))
        
        # 创建更新的参数
        updated_params = {
            'source_x': source_x,
            'source_y': source_y,
            'source_width': source_width,
            'source_height': source_height,
            'scale': self.scale_factor
        }
        
        # 发出信号
        self.crop_rect_updated.emit(self.grid_id, updated_params)


class GifExportThread(QThread):
    """用于在后台线程中导出GIF的线程。"""
    def __init__(self, video_path, grid_configs, output_dir, target_fps, 
                 parent_signal_progress, parent_signal_finished, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.grid_configs = grid_configs
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.progress_signal = parent_signal_progress
        self.finished_signal = parent_signal_finished
        self._is_terminated = False

    def run(self):
        try:
            video_processor = VideoProcessor(self.video_path) # 在线程内创建，避免跨线程问题
            num_grids = len(self.grid_configs)
            self.progress_signal.emit(0, f"开始导出 {num_grids} 个GIF...")

            # 1. 提取整个视频的所有相关帧 (可以优化为只提取一次)
            # 考虑到每个grid可能有不同的时间范围或帧率需求（如果未来支持），这里暂时为每个grid提取
            # 但为了效率，如果所有grid使用相同的时间范围和帧率，应该只提取一次。
            # 假设所有GIF使用整个视频长度和统一的目标FPS
            all_video_frames_rgb, actual_fps = video_processor.extract_frames(target_fps=self.target_fps)
            if not all_video_frames_rgb:
                raise Exception("无法从视频中提取帧。")
            
            self.progress_signal.emit(0, f"已提取 {len(all_video_frames_rgb)} 帧 @ {actual_fps:.2f} FPS")
            QApplication.processEvents() # 允许UI更新

            for i, grid_config in enumerate(self.grid_configs):
                if self._is_terminated: 
                    self.finished_signal.emit(False, "导出已取消。")
                    video_processor.release()
                    return
                
                grid_id = grid_config.get('id', i)
                self.progress_signal.emit(i, f"处理区块 {grid_id + 1}/{num_grids}...")
                
                # a. 根据grid_config裁剪每一帧
                cropped_frames_for_gif = []
                for frame_rgb in all_video_frames_rgb:
                    # crop_rect 使用 grid_config 中的 source_x, source_y, source_width, source_height
                    # 这些值应该是用户在UI上调整后的最终裁剪区域
                    current_crop_rect = {
                        'x': grid_config.get('source_x'),
                        'y': grid_config.get('source_y'),
                        'width': grid_config.get('source_width'),
                        'height': grid_config.get('source_height'),
                    }
                    # 校验一下，防止None值
                    if any(v is None for v in current_crop_rect.values()):
                        # print(f"警告: 区块 {grid_id+1} 的裁剪参数不完整，跳过此帧。Params: {current_crop_rect}")
                        continue

                    cropped_frame = VideoProcessor.crop_frame(frame_rgb, current_crop_rect, 
                                                              target_size=(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT))
                    if cropped_frame is not None:
                        cropped_frames_for_gif.append(cropped_frame)
                
                if not cropped_frames_for_gif:
                    # print(f"警告: 区块 {grid_id + 1} 没有有效的裁剪帧，跳过GIF生成。")
                    self.progress_signal.emit(i + 1, f"区块 {grid_id + 1} 无有效帧，已跳过。")
                    continue

                # b. 生成GIF
                gif_filename = f"block_{grid_id + 1}.gif"
                gif_output_path = os.path.join(self.output_dir, gif_filename)
                
                self.progress_signal.emit(i, f"区块 {grid_id + 1}: 生成GIF到 {gif_filename}...")
                QApplication.processEvents()
                
                # 预处理帧以提高压缩效率
                # 1. 检查帧数，如果帧数过多，尝试通过关键帧检测减少
                if len(cropped_frames_for_gif) > 30:  # 如果帧数超过30
                    self.progress_signal.emit(i, f"区块 {grid_id + 1}: 优化帧数...")
                    # 使用关键帧检测减少帧数
                    keyframe_indices = GifGenerator.detect_keyframes(
                        cropped_frames_for_gif, 
                        threshold=0.02  # 使用较低阈值捕获更多变化
                    )
                    if len(keyframe_indices) < len(cropped_frames_for_gif) * 0.8:  # 如果至少减少了20%
                        optimized_frames = [cropped_frames_for_gif[i] for i in keyframe_indices]
                        print(f"区块 {grid_id + 1}: 通过关键帧检测将帧数从 {len(cropped_frames_for_gif)} 减少到 {len(optimized_frames)}")
                        cropped_frames_for_gif = optimized_frames
                
                # 2. 预调整帧率，基于帧数初步估计
                estimated_fps = min(actual_fps, max(5, 15 - (len(cropped_frames_for_gif) // 20)))
                
                # 3. 如果尺寸不是240x240，预先调整
                if (cropped_frames_for_gif[0].shape[0] != config.WECHAT_STICKER_HEIGHT or 
                    cropped_frames_for_gif[0].shape[1] != config.WECHAT_STICKER_WIDTH):
                    self.progress_signal.emit(i, f"区块 {grid_id + 1}: 调整尺寸...")
                    resized_frames = []
                    for frame in cropped_frames_for_gif:
                        resized = cv2.resize(frame, (config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT), 
                                            interpolation=cv2.INTER_LANCZOS4)
                        resized_frames.append(resized)
                    cropped_frames_for_gif = resized_frames
                
                # 在这里调用GifGenerator
                # 这里的 target_size_kb=config.WECHAT_STICKER_MAX_SIZE_KB 会触发内部的大小优化逻辑
                gif_size_kb = GifGenerator.create_optimized_gif(
                    cropped_frames_for_gif, 
                    gif_output_path, 
                    fps=estimated_fps,  # 使用估计的较低帧率
                    max_colors=config.MAX_GIF_COLORS,
                    target_size_kb=config.WECHAT_STICKER_MAX_SIZE_KB * 0.95,  # 预留5%的余量确保不超过500KB
                    dither=True,
                    optimize_palette_globally=True  # 推荐为表情包使用全局调色板
                )

                if gif_size_kb > 0:
                    self.progress_signal.emit(i + 1, f"区块 {grid_id + 1} GIF已保存 ({gif_size_kb:.1f}KB)")
                    # c. (可选) 生成预览图和缩略图
                    if cropped_frames_for_gif: # 使用第一帧或中间帧
                        preview_pil = ImageOptimizer.create_preview_image(cropped_frames_for_gif[0])
                        thumb_pil = ImageOptimizer.create_thumbnail_image(cropped_frames_for_gif[0])
                        preview_pil.save(os.path.join(self.output_dir, f"preview_{grid_id+1}.png"))
                        thumb_pil.save(os.path.join(self.output_dir, f"thumb_{grid_id+1}.png"))
                else:
                    self.progress_signal.emit(i + 1, f"区块 {grid_id + 1} GIF生成失败。")
            
            video_processor.release()
            if not self._is_terminated:
                self.finished_signal.emit(True, self.output_dir)

        except Exception as e:
            # print(f"GIF导出线程错误: {e}")
            # import traceback
            # traceback.print_exc()
            if hasattr(self, 'video_processor') and self.video_processor: self.video_processor.release()
            self.finished_signal.emit(False, str(e))

    def terminate_processing(self):
        self._is_terminated = True
        self.progress_signal.emit(self.progress_dialog.value(), "正在取消导出...")


if __name__ == '__main__':
    # 这个main主要用于独立测试 MainWindow，但实际运行应通过 src/main.py
    import sys
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_()) 