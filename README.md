# GenerateSimulatedData

Builds **simulated Arcturus speckle data** (imaging frames + Shack–Hartmann WFS
frames) with a *known* wavefront and spectrum, so the NewDataPipeline can be
tested against ground truth.

Everything propagates from one set of atmospheric wavefronts, through the real
CDK700 + relay optics, to realistic noisy detector frames.

---

## How to run

1. Open `run_all.bat` and set the frame count near the top:
   ```
   set SIM_N_FRAMES=50
   ```
2. Run it:
   ```
   run_all.bat
   ```
   It wipes `fitsOutputs/` and `figures/`, then runs steps `a → h` in order and
   stops if any step fails.

You can also run any script on its own (`python e_generateImagingFrames.py`).
Standalone, `d_` defaults to 50 frames and `e_/f_/g_` process whatever exists.

> Heads-up: `e_` (200 wavelength sub-bands) is the slow step. ~50 frames is a
> few minutes; for ~280 *usable* frames set `SIM_N_FRAMES≈875` (only ~32% survive
> the saturation cut) — that's a few hours.

---

## The steps

| Step | Script | What it does | Main output |
|---|---|---|---|
| a | `a_generatePupilModel` | Builds the 0.7 m telescope pupil — circular aperture with the central obstruction and spider vanes. | `pupil_model_0.70m.fits` |
| b | `b_initializeObject` | The **true object spectrum**: the real Arcturus SED (MELCHIORS + UVES-POP), 320–1000 nm at 0.5 nm. | `object_spectrum_arcturus_*.fits` |
| c | `c_effectiveTransmission` | System throughput (atmosphere × mirrors × optics × detector QE) for the WFS and imaging legs. | `system_throughput_{wfs,imaging}.fits` |
| d | `d_generatePhaseScreens` | The **true atmospheric wavefronts** (OPD screens) via HCIPy — Kolmogorov turbulence at the set r₀, blown by wind so frames evolve. | `phase_screens_opd_nm.fits` |
| e | `e_generateImagingFrames` | Clean broadband images: each wavelength's PSF (wavefront + chromatic defocus + dispersion), summed weighted by spectrum × throughput. | `imaging_frames_clean.fits` |
| f | `f_addNoise` | Scales to real photon counts, adds shot noise, read noise, and the **dark** → raw imaging frames. Flags saturated frames. | `imaging_frames_raw.fits` |
| g | `g_generateWFSFrames` | Shack–Hartmann spot frames from the *same* wavefronts (secondary + spiders blank lenslets), with noise + dark. | `wfs_frames_raw.fits` |
| h | `h_filterValidFrames` | Keeps only frames where **both** imaging and WFS are unsaturated, kept frame-aligned, plus a manifest back to the truth. | `imaging_frames_valid.fits`, `wfs_frames_valid.fits`, `valid_frames.npz` |

---

## Folders

- `configFiles/` — inputs: optics throughput curves, Arcturus/MELCHIORS spectra, AC254 lens BFD files.
- `darks/` — dark calibration: average darks (imaging + WFS) and the read-noise map. *(Optional: add the per-frame dark stacks for the real-dark noise floor.)*
- `fitsOutputs/` — all generated FITS.
- `figures/` — diagnostic plots from each step.

---

## Ground truth (for the recovery test)

When you point the pipeline at `imaging_frames_valid.fits` + `wfs_frames_valid.fits`,
compare what it recovers against the known truth:

- **Wavefront:** `phase_screens_opd_nm.fits` (use `valid_frames.npz → valid_indices` to match frames).
- **Spectrum:** `object_spectrum_arcturus_*.fits`.
- **Per-frame flux:** `ground_truth_flux.npz → flux_scalar`.

`valid_frames.npz` links each kept frame back to its original OPD-screen index,
and the imaging/WFS valid cubes are frame-for-frame aligned.

---

## Key settings (top of each script)

- `d_`: `R0_M` (Fried parameter, currently 0.020 m), `WIND_MS`, `L0_M`.
- `e_`: `N_SUBBANDS` (200), telescope/relay focal lengths, `ALTITUDE_DEG` and
  `ZENITH_ON_DET_DEG` (set these for the dispersion geometry).
- `f_`: `TARGET_SIGNAL_ADU` (3.92e6), `FLUX_JITTER_RMS` (scintillation), `SAT_LEVEL`.
- `g_`: `WFS_TARGET_SIGNAL_ADU` (3.05e6), `LAMBDA_WFS_NM`.
