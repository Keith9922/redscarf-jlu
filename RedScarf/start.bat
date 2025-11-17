@echo off
REM 红领巾检测系统 - Windows快速启动脚本

echo ==========================================
echo     红领巾检测系统 - 快速启动
echo ==========================================
echo.

REM 检查Python环境
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

echo 使用Python: python
echo.

:menu
echo 请选择启动模式:
echo   1) Web界面 (推荐)
echo   2) 摄像头实时检测
echo   3) 检测图片
echo   4) 检测视频
echo   5) 安装依赖
echo   0) 退出
echo.

set /p choice="请输入选项 [1-5]: "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto camera
if "%choice%"=="3" goto image
if "%choice%"=="4" goto video
if "%choice%"=="5" goto install
if "%choice%"=="0" goto exit
echo 无效选项
goto menu

:web
echo.
echo 正在启动Web界面...
python app.py
goto end

:camera
echo.
set /p camera_id="请输入摄像头ID [默认0]: "
if "%camera_id%"=="" set camera_id=0
echo 正在启动摄像头检测 (Camera ID: %camera_id%)...
python Main.py --camera %camera_id%
goto end

:image
echo.
set /p image_path="请输入图片路径: "
if "%image_path%"=="" (
    echo 错误: 未指定图片路径
    pause
    exit /b 1
)
echo 正在检测图片...
python Main.py --image "%image_path%"
goto end

:video
echo.
set /p video_path="请输入视频路径: "
if "%video_path%"=="" (
    echo 错误: 未指定视频路径
    pause
    exit /b 1
)
set /p save_result="是否保存结果? (y/n) [默认n]: "
if /i "%save_result%"=="y" (
    set /p output_path="请输入输出路径 [默认output.mp4]: "
    if "!output_path!"=="" set output_path=output.mp4
    echo 正在检测视频并保存...
    python Main.py --video "%video_path%" --output "!output_path!"
) else (
    echo 正在检测视频...
    python Main.py --video "%video_path%"
)
goto end

:install
echo.
echo 正在安装依赖...
python -m pip install -r requirements.txt
echo.
echo 依赖安装完成!
pause
goto end

:exit
echo 退出程序
exit /b 0

:end
pause
