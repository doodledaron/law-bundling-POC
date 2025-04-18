@echo off
echo CUAD Processing Log Analyzer
echo ------------------------------

if "%~1"=="" (
    echo Error: Please provide the log file path.
    echo Usage: analyze_log.bat [log_file_path]
    exit /b 1
)

set LOG_FILE=%~1
set OUTPUT_FILE=%~n1_analysis.json

echo Analyzing log file: %LOG_FILE%
echo Output will be saved to: %OUTPUT_FILE%

python analyze_processing_log.py "%LOG_FILE%" "%OUTPUT_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo Analysis failed. Please check the log file path.
    exit /b 1
)

echo.
echo Analysis complete! Check %OUTPUT_FILE% and %~n1_analysis.txt for results.
echo. 