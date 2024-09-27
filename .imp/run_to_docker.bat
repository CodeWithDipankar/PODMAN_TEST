
@REM @echo off
@REM :: Allow custom image name or default to "my-test"
@REM set IMAGE_NAME=%1
@REM if "%IMAGE_NAME%"=="" set IMAGE_NAME=my-test

@REM :: Define the Result folder path
@REM set RESULT_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\Result

@REM :: Create the Result directory if it doesn't exist
@REM if not exist "%RESULT_DIR%" (
@REM     mkdir "%RESULT_DIR%"
@REM )

@REM :: Build the image
@REM echo Building the container image...
@REM podman build -t %IMAGE_NAME% .

@REM :: Check if the build was successful
@REM if %errorlevel% neq 0 (
@REM     echo Failed to build the image. Exiting.
@REM     exit /b %errorlevel%
@REM )

@REM :: Run the container, mounting the Result directory to /app/output
@REM echo Running the container...
@REM podman run -it -v "%RESULT_DIR%:/app/output" "%IMAGE_NAME%"

@REM :: Check if the container ran successfully
@REM if %errorlevel% neq 0 (
@REM     echo Failed to run the container. Exiting.
@REM     exit /b %errorlevel%
@REM )

@REM echo Container ran successfully.
@REM pause


@echo off
:: Allow custom image name or default to "my-test"
set IMAGE_NAME=%1
if "%IMAGE_NAME%"=="" set IMAGE_NAME=my-test

:: Define input and output folder paths
set INPUT_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\data\INPUT
set OUTPUT_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\data\OUTPUT

:: Create the OUTPUT directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

:: Build the image
echo Building the container image...
podman build -t %IMAGE_NAME% .

:: Check if the build was successful
if %errorlevel% neq 0 (
    echo Failed to build the image. Exiting.
    exit /b %errorlevel%
)

:: Run the container, mounting both INPUT and OUTPUT directories
echo Running the container...
podman run -it -v "%INPUT_DIR%:/app/input" -v "%OUTPUT_DIR%:/app/output" "%IMAGE_NAME%"

:: Check if the container ran successfully
if %errorlevel% neq 0 (
    echo Failed to run the container. Exiting.
    exit /b %errorlevel%
)

echo Container ran successfully.
pause
