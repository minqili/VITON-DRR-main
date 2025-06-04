@echo off
CALL conda activate torch
echo Running first Python script...
python ".\segment\test_condition.py"
if %errorlevel% neq 0 (
    echo Error: First script failed to execute.
    exit /b 1
)

echo First script executed successfully.
echo Running second Python script...
python "test.py"
if %errorlevel% neq 0 (
    echo Error: Second script failed to execute.
    exit /b 1
)

echo Second script executed successfully.
pause
