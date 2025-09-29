@echo off
REM install_miniforge.bat

REM Default Python version if not set in environment
IF "%PY%"=="" SET "PY=3.12"

REM Error handler
SETLOCAL ENABLEEXTENSIONS
SET SCRIPT=%~dpnx0
SET SCRIPTPATH=%~dp0
SET "ANUGA_CORE_PATH=%~dp0\.."
PUSHD %SCRIPTPATH%
SET ANUGA_CORE_PATH=%CD%\..
POPD

REM Check allowed Python versions
ECHO %PY% | FINDSTR /R "^3\.9 ^3\.10 ^3\.11 ^3\.12 ^3.13" > NUL
IF ERRORLEVEL 1 (
    ECHO Python version must be greater than 3.8 and less than 3.14
    EXIT /B 1
) ELSE (
    ECHO Requested python version is %PY%
    ECHO.
)

ECHO #===========================
ECHO # Install miniforge3
ECHO #===========================

CD /D %USERPROFILE%
IF EXIST "%USERPROFILE%\miniforge3" (
    ECHO miniforge3 seems to already exist.
) ELSE (
    ECHO miniforge3 does not exist.
    IF EXIST "%USERPROFILE%\Miniforge3.exe" (
        ECHO Miniforge3.exe installer exists...
    ) ELSE (
        ECHO Miniforge3.exe does not exist. Downloading...
        curl -fsSLo %USERPROFILE%\Miniforge3.exe https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
        IF ERRORLEVEL 1 (
            ECHO Download failed. Exiting.
            EXIT /B 1
        )
    )
    IF EXIST "%USERPROFILE%\Miniforge3.exe" (
        ECHO Running Miniforge3.exe installer...
    )
    START /WAIT "" "%USERPROFILE%\Miniforge3.exe" /S /D=%USERPROFILE%\miniforge3
    IF ERRORLEVEL 1 (
        ECHO Installation failed. Exiting.
        EXIT /B 1
    )
)

ECHO.
ECHO #===============================================
ECHO # create conda environment anuga_env_%PY%
ECHO #===============================================
ECHO ...

"%USERPROFILE%\miniforge3\Scripts\conda.exe" env create --file "%SCRIPTPATH%\..\environments\environment_%PY%.yml"
IF ERRORLEVEL 1 (
    ECHO Environment creation failed. Exiting.
    EXIT /B 1
)

ECHO.
ECHO #======================================
ECHO # activate environment anuga_env_%PY%
ECHO #======================================
ECHO ...

CALL "%USERPROFILE%\miniforge3\Scripts\activate.bat" anuga_env_%PY%
IF ERRORLEVEL 1 (
    ECHO Activation failed. Exiting.
    EXIT /B 1
)

ECHO #================================================================
ECHO # Install the compilers on Windows
ECHO #================================================================
ECHO ...

CALL conda install -c conda-forge libpython gcc_win-64 gxx_win-64
IF ERRORLEVEL 1 (
    ECHO Compiler installation failed. Exiting.
    EXIT /B 1
)

ECHO #================================================================
ECHO # Installing anuga from the %ANUGA_CORE_PATH% directory
ECHO #================================================================
ECHO ...

CD /D "%SCRIPTPATH%\.."
pip install --no-build-isolation .
IF ERRORLEVEL 1 (
    ECHO anuga install failed. Exiting.
    EXIT /B 1
)

ECHO.
ECHO #===========================
ECHO # Run unittests
ECHO #===========================
ECHO.

CD /D "%SCRIPTPATH%\..\sandpit"
pytest -q --disable-warnings --pyargs anuga
IF ERRORLEVEL 1 (
    ECHO Unit tests failed. Exiting.
    EXIT /B 1
)

ECHO.
ECHO #==================================================================
ECHO # Congratulations, Looks like you have successfully installed anuga
ECHO #==================================================================
ECHO #==================================================================
ECHO # To use anuga you must activate the python environment anuga_env_%PY%
ECHO # that has just been created. Run the command
ECHO # 
ECHO # call %%USERPROFILE%%\miniforge3\Scripts\activate.bat anuga_env_%PY%
ECHO #
ECHO # Or use conda activate anuga_env_%PY% if conda is initialized
ECHO #==================================================================
ECHO # NOTE: You can run
ECHO #
ECHO # conda init
ECHO #
ECHO # to enable 'conda activate anuga_env_%PY%' in all new shells
ECHO #==================================================================
EXIT /B 0
