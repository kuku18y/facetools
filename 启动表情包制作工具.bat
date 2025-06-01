@echo off
chcp 65001 >nul
echo 正在启动微信表情包开发神器...
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

:: 检查依赖项是否已安装
echo 检查依赖项...
pip show PyQt5 >nul 2>&1
if %errorlevel% neq 0 (
    echo 首次运行，正在安装依赖项，请稍候...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 安装依赖项失败。
        pause
        exit /b
    )
)

:: 启动应用程序
echo 启动应用程序...
python src/main.py

:: 如果应用异常退出，暂停以显示错误信息
if %errorlevel% neq 0 (
    echo.
    echo [错误] 应用程序异常退出，错误代码: %errorlevel%
    pause
)
