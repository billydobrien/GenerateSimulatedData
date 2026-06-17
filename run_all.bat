@echo off
REM ====================================================================
REM  Regenerate the full simulated-data pipeline from scratch.
REM  Deletes figures/ and fitsOutputs/, then runs a_ -> h_ in sequence.
REM  Stops immediately if any step fails.
REM ====================================================================
setlocal
cd /d "%~dp0"

REM ===== NUMBER OF FRAMES TO SIMULATE (set this) ======================
set SIM_N_FRAMES=200
REM    d_ generates this many OPD screens; e_/f_/g_ process them all;
REM    h_ keeps the unsaturated paired subset (~32%% at r0=2.0 cm).
REM ====================================================================

echo === Cleaning output folders ===
if exist "figures"     rmdir /s /q "figures"
if exist "fitsOutputs" rmdir /s /q "fitsOutputs"

echo.
echo === STEP a_ : pupil model ===
python a_generatePupilModel.py     || goto :error
echo.
echo === STEP b_ : object spectrum ===
python b_initializeObject.py       || goto :error
echo.
echo === STEP c_ : effective transmission ===
python c_effectiveTransmission.py  || goto :error
echo.
echo === STEP d_ : phase screens  (SIM_N_FRAMES=%SIM_N_FRAMES%) ===
python d_generatePhaseScreens.py   || goto :error
echo.
echo === STEP e_ : clean imaging frames ===
python e_generateImagingFrames.py  || goto :error
echo.
echo === STEP f_ : imaging photon budget + noise + dark ===
python f_addNoise.py               || goto :error
echo.
echo === STEP g_ : WFS frames (+ noise + dark) ===
python g_generateWFSFrames.py      || goto :error
echo.
echo === STEP h_ : pair + filter valid (unsaturated) frames ===
python h_filterValidFrames.py      || goto :error

echo.
echo === ALL STEPS COMPLETE  (SIM_N_FRAMES=%SIM_N_FRAMES%) ===
exit /b 0

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%) - stopping ***
exit /b %errorlevel%
