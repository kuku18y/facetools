@echo off
chcp 65001 >nul
echo 正在安装微信表情包开发神器所需的依赖项...
echo 作者: TONY老师 (微信视频号: TONY老师教AI)
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python安装，请先安装Python 3.6或更高版本。
    echo 可以从 https://www.python.org/downloads/ 下载Python。
    pause
    exit /b
)

:: 安装依赖项
echo 正在安装依赖项，这可能需要几分钟时间...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [错误] 安装依赖项失败。
    pause
    exit /b
) else (
    echo.
    echo 依赖项安装完成！
    echo 现在您可以双击"启动表情包制作工具.bat"来启动程序。
    echo.
    pause
)
