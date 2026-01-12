@echo off
echo DiffRhythm Model Symlink Resolver for Windows
echo ==================================================

cd /d "%~dp0"

if not exist "pretrained" (
    echo Error: pretrained directory not found
    pause
    exit /b 1
)

echo Resolving symlinks in pretrained directory...

:: DiffRhythm-1_2 model
if exist "pretrained\models--ASLP-lab--DiffRhythm-1_2\snapshots" (
    for /d %%d in ("pretrained\models--ASLP-lab--DiffRhythm-1_2\snapshots\*") do (
        if exist "%%d\cfm_model.pt" (
            echo Resolving DiffRhythm-1_2 model...
            del "%%d\cfm_model.pt" 2>nul
            copy "pretrained\models--ASLP-lab--DiffRhythm-1_2\blobs\*" "%%d\cfm_model.pt" >nul
            if exist "%%d\cfm_model.pt" echo ✓ DiffRhythm-1_2 resolved
        )
    )
)

:: DiffRhythm-1_2-full model
if exist "pretrained\models--ASLP-lab--DiffRhythm-1_2-full\snapshots" (
    for /d %%d in ("pretrained\models--ASLP-lab--DiffRhythm-1_2-full\snapshots\*") do (
        if exist "%%d\cfm_model.pt" (
            echo Resolving DiffRhythm-1_2-full model...
            del "%%d\cfm_model.pt" 2>nul
            copy "pretrained\models--ASLP-lab--DiffRhythm-1_2-full\blobs\*" "%%d\cfm_model.pt" >nul
            if exist "%%d\cfm_model.pt" echo ✓ DiffRhythm-1_2-full resolved
        )
    )
)

:: VAE model
if exist "pretrained\models--ASLP-lab--DiffRhythm-vae\snapshots" (
    for /d %%d in ("pretrained\models--ASLP-lab--DiffRhythm-vae\snapshots\*") do (
        if exist "%%d\vae_model.pt" (
            echo Resolving VAE model...
            del "%%d\vae_model.pt" 2>nul
            copy "pretrained\models--ASLP-lab--DiffRhythm-vae\blobs\*" "%%d\vae_model.pt" >nul
            if exist "%%d\vae_model.pt" echo ✓ VAE model resolved
        )
    )
)

echo.
echo ✅ Symlink resolution complete!
echo Your models are now ready for AWS deployment.
echo.
pause