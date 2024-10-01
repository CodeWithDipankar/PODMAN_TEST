@echo off
:: Allow custom image name or default to "my-test"
set IMAGE_NAME=%1
if "%IMAGE_NAME%"=="" set IMAGE_NAME=model_engine

:: Allow custom container name or default to "my-container"
set CONTAINER_NAME=%2
if "%CONTAINER_NAME%"=="" set CONTAINER_NAME=model_engine

:: Define input, output, and AWS credentials folder paths
set INPUT_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\data\INPUT
set OUTPUT_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\data\OUTPUT
set AWS_DIR=C:\Users\Dipankar.Mandal\.aws
set REPO_DIR=C:\Users\Dipankar.Mandal\OneDrive - Ipsos\1.CORE_CODE\PODMAN_TEST\code  :: Directory with your source code
set LAST_BUILD_FILE=%REPO_DIR%\last_build.txt  :: File to track the last build hash

:: Create the OUTPUT directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

:: Calculate the hash of the source code files and requirements.txt
for /f "delims=" %%i in ('certutil -hashfile "%REPO_DIR%\requirements.txt" SHA256') do (
    set REQ_HASH=%%i
)

set CODE_HASH=
for /f "delims=" %%i in ('certutil -hashfile "%REPO_DIR%\your_script.py" SHA256') do (
    set CODE_HASH=%%i
)

:: Combine the hashes into one string
set CURRENT_HASH=%REQ_HASH%_%CODE_HASH%

:: Check if the last build file exists and compare hashes
if exist "%LAST_BUILD_FILE%" (
    set /p LAST_HASH=<"%LAST_BUILD_FILE%"
    if "%LAST_HASH%"=="%CURRENT_HASH%" (
        echo No changes detected. Using existing image: %IMAGE_NAME%
        goto RUN_CONTAINER
    )
)

:: Build the image
echo Building the container image...
podman build -t %IMAGE_NAME% .

:: Check if the build was successful
if %errorlevel% neq 0 (
    echo Failed to build the image. Exiting.
    exit /b %errorlevel%
)

:: Save the current hash to the last build file
echo %CURRENT_HASH% > "%LAST_BUILD_FILE%"

::RUN_CONTAINER
:: Stop and remove the existing container if it's running
podman stop %CONTAINER_NAME% 2>nul
podman rm %CONTAINER_NAME% 2>nul

:: Run the container, mounting INPUT, OUTPUT, and AWS directories with custom container name
echo Running the container...
podman run -it --name %CONTAINER_NAME% -v "%INPUT_DIR%:/app/input" -v "%OUTPUT_DIR%:/app/output" -v "%AWS_DIR%:/root/.aws:ro" "%IMAGE_NAME%"

:: Check if the container ran successfully
if %errorlevel% neq 0 (
    echo Failed to run the container. Exiting.
    exit /b %errorlevel%
)

echo Container ran successfully.
