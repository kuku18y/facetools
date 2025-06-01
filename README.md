# 微信表情包开发神器

本工具旨在帮助微信表情包开发者快速、便捷地将MP4视频转换为符合微信表情包规范的GIF动画。

## 主要功能

- **视频导入与预览**：支持拖拽导入MP4视频，实时预览。
- **九宫格编辑**：提供交互式的九宫格编辑界面，可对每个小格进行缩放、拖动等操作。
- **智能GIF生成**：内置智能算法，优化GIF的帧率、颜色、大小，确保符合微信平台要求。
- **合规性检查**：自动检查生成文件是否符合微信表情包规范（尺寸、大小等）。
- **一键导出**：支持导出GIF、预览图和缩略图。
- **微信平台集成**：未来将支持直接上传到微信表情开放平台。

## 技术架构

- **后端**：Python
- **GUI框架**：PyQt5 (替代WPF以便跨平台兼容性，且Python生态更完善)
- **核心处理库**：
    - OpenCV: 视频解码、图像处理
    - Pillow: 图像处理、格式转换
    - ImageIO: GIF编码与优化
    - Requests: 与微信API交互

## 安装与运行

1.  **克隆项目**：
    ```bash
    git clone <repository_url>
    cd wechat-sticker-app
    ```

2.  **创建虚拟环境 (推荐)**：
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

4.  **运行应用**：
    ```bash
    python src/main.py
    ```

## 项目结构

详细的项目结构请参见：`.cursor/rules/project-structure.mdc`

## 开发规范

代码规范、版本控制、测试等详细信息请参见：`.cursor/rules/development-guidelines.mdc`

## 核心算法

视频处理、GIF生成等核心算法的实现细节请参见：`.cursor/rules/algorithm-implementation.mdc`

## 微信API集成

与微信表情开放平台API集成的相关信息请参见：`.cursor/rules/wechat-api-integration.mdc`

## 贡献

欢迎提交 Pull Request 或 Issue！

## 许可证

[MIT License](LICENSE) (请根据实际情况选择) 