@echo off
echo ================================================
echo Healthcare Detection - App Launcher
echo ================================================
echo.
echo Choose which app to run:
echo.
echo 1. Main Pneumonia Detection App
echo 2. Interactive Training Interface
echo 3. Both (in separate windows)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Main App...
    streamlit run app/streamlit_app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Interactive Training...
    streamlit run app/interactive_training.py
) else if "%choice%"=="3" (
    echo.
    echo Starting both apps...
    start cmd /k "streamlit run app/streamlit_app.py --server.port=8501"
    timeout /t 2 /nobreak >nul
    start cmd /k "streamlit run app/interactive_training.py --server.port=8502"
    echo.
    echo Main App: http://localhost:8501
    echo Training App: http://localhost:8502
) else (
    echo Invalid choice!
    pause
)
