"""
Add a realistic photon budget + noise + the DARK to the clean imaging frames,
producing RAW simulated frames (the analogue of the real raw data, before the
pipeline subtracts the dark).

Composition of each raw frame (in ADU):

    raw = Poisson(expected_e) / gain          # star signal + shot noise
        + real_dark_frame                      # pedestal + read + hot pixels

Photon budget:
  * The clean frame is relative (sum ~ 1).  We scale the STAR signal so its total
    matches the OBSERVED dark-subtracted total (~3.92e6 ADU from the flux plot).
  * A per-frame flux scalar s_f ~ Normal(1, 3.6%) models scintillation /
    transparency (the frame-to-frame total-counts jitter).  This is the thing the
    pipeline's mean-anchored per-frame scalar is meant to absorb, so we save the
    ground-truth s_f to check the pipeline recovers it.

The DARK is the real cropped dark stack (pedestal + read noise + hot pixels), one
frame drawn per simulated frame — so no separate read-noise model is needed.

Position jitter (tip/tilt) is already in the phase screens; this script only adds
the flux jitter + photon noise + dark.

Run:  python f_addNoise.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.io import fits

# ── Config ───────────────────────────────────────────────────────────────────
GAIN              = 0.46        # e-/ADU
TARGET_SIGNAL_ADU = 3.92e6      # observed dark-subtracted total per frame
FLUX_JITTER_RMS   = 0.036       # per-frame total-counts scatter (scintillation)
SAT_LEVEL         = 4095.0      # ADC saturation level [ADU] (raw)
SEED              = 1

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
CLEAN_FILE  = os.path.join(OUTPUT_DIR, "imaging_frames_clean.fits")

# Dark calibration (local).  The small average dark + read-noise map are
# included; the large per-frame dark STACKS are optional — drop them into darks/
# to use the real-dark noise floor, otherwise f_ falls back to
# average-dark + Gaussian read noise (per-pixel sigma from the read map).
_DARKDIR   = os.path.join(SCRIPT_DIR, "darks")
DARK_STACKS = [os.path.join(_DARKDIR, "dark_p005_1_cropped.fits"),
               os.path.join(_DARKDIR, "dark_p005_2_cropped.fits")]
AVG_DARK_FILE = os.path.join(_DARKDIR, "average_dark_imaging_arcturus_3_cropped.fits")
READ_MAP_FILE = os.path.join(_DARKDIR, "read_noise_variance_map.fits")   # fallback only


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    clean = fits.getdata(CLEAN_FILE).astype(np.float64)      # (Nf, 256, 256), sum~1
    nf, ny, nx = clean.shape
    avg_dark = fits.getdata(AVG_DARK_FILE).astype(np.float64)

    # ── Dark noise floor: pool the real cropped dark frames ──────────────────
    use_real_darks = all(os.path.exists(p) for p in DARK_STACKS)
    if use_real_darks:
        pool = np.concatenate([fits.getdata(p).astype(np.float32) for p in DARK_STACKS], axis=0)
        print(f"[sim-f] dark noise floor: {pool.shape[0]} real dark frames "
              f"(pedestal+read+hot pixels)")
    else:
        read_sigma = np.sqrt(fits.getdata(READ_MAP_FILE).astype(np.float64))   # ADU
        print(f"[sim-f] dark stacks not found — using avg_dark + Gaussian read noise (per-pixel map)")

    # ── Per-frame flux scalars (scintillation) ───────────────────────────────
    s_f = np.clip(rng.normal(1.0, FLUX_JITTER_RMS, nf), 0.2, None)

    raw = np.empty((nf, ny, nx), dtype=np.float32)
    sig_total, raw_total, ds_total = [], [], []
    target_e = TARGET_SIGNAL_ADU * GAIN                     # target signal electrons

    for i in range(nf):
        sig_norm = clean[i] / clean[i].sum()                # sum = 1
        expected_e = sig_norm * (target_e * s_f[i])         # expected electrons / pixel
        signal_adu = rng.poisson(expected_e) / GAIN         # shot-noisy signal [ADU]

        if use_real_darks:
            dark_frame = pool[rng.integers(pool.shape[0])].astype(np.float64)
        else:
            dark_frame = avg_dark + rng.normal(0.0, read_sigma)

        frame = signal_adu + dark_frame                     # RAW (dark included)
        raw[i] = frame.astype(np.float32)

        sig_total.append(signal_adu.sum())
        raw_total.append(frame.sum())
        ds_total.append((frame - avg_dark).sum())           # what the pipeline recovers

    sig_total = np.array(sig_total); raw_total = np.array(raw_total); ds_total = np.array(ds_total)
    print(f"[sim-f] signal total      : mean {sig_total.mean():.3e} ADU  (target {TARGET_SIGNAL_ADU:.2e})")
    print(f"[sim-f] dark pedestal sum : {avg_dark.sum():.3e} ADU  -> raw total mean {raw_total.mean():.3e} ADU")
    print(f"[sim-f] dark-subtracted   : mean {ds_total.mean():.3e} ADU  RMS {100*ds_total.std()/ds_total.mean():.2f}%")

    # ── Saturation: flag (raw peak >= SAT_LEVEL), clamp at the ADC level ─────
    raw_peak = raw.max(axis=(1, 2))
    unsat = raw_peak < SAT_LEVEL
    n_unsat = int(unsat.sum())
    print(f"[sim-f] saturation: {n_unsat}/{nf} frames unsaturated "
          f"({100*n_unsat/nf:.1f}%)  [target ~28% -> tune r0 in d_]")
    raw = np.minimum(raw, SAT_LEVEL).astype(np.float32)   # realistic ADC clamp

    # ── Save: full (clamped) + the unsaturated subset, + ground truth ────────
    hdr = fits.Header()
    hdr["BUNIT"]    = ("ADU", "RAW frame: signal + shot + dark, ADC-clamped")
    hdr["NFRAMES"]  = (nf, "Frames (all)")
    hdr["GAIN"]     = (GAIN, "e-/ADU")
    hdr["SIGADU"]   = (TARGET_SIGNAL_ADU, "Target dark-subtracted signal total [ADU]")
    hdr["FLUXJIT"]  = (FLUX_JITTER_RMS, "Per-frame flux-scalar RMS (scintillation)")
    hdr["SATLEVEL"] = (SAT_LEVEL, "ADC saturation level [ADU]")
    hdr["NUNSAT"]   = (n_unsat, "Unsaturated frames (raw peak < SATLEVEL)")
    hdr["DARKSRC"]  = ("real frames" if use_real_darks else "avg+gauss", "Dark noise floor")
    raw_path = os.path.join(OUTPUT_DIR, "imaging_frames_raw.fits")
    fits.writeto(raw_path, raw, header=hdr, overwrite=True)
    print(f"[sim-f] all raw frames     -> {os.path.relpath(raw_path, SCRIPT_DIR)}  {raw.shape}")

    notsat_path = os.path.join(OUTPUT_DIR, "imaging_frames_raw_notSat.fits")
    fits.writeto(notsat_path, raw[unsat], header=hdr, overwrite=True)
    print(f"[sim-f] unsaturated subset -> {os.path.relpath(notsat_path, SCRIPT_DIR)}  {raw[unsat].shape}")

    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_flux.npz")
    np.savez(gt_path, flux_scalar=s_f, signal_total_adu=sig_total, raw_total_adu=raw_total,
             unsat_mask=unsat)
    print(f"[sim-f] ground-truth flux scalars -> {os.path.relpath(gt_path, SCRIPT_DIR)}")

    # ── Figures ──────────────────────────────────────────────────────────────
    ng = min(6, nf)
    # sqrt (PowerNorm) stretch to the TRUE max so bright speckle cores show their
    # real value while the faint halo stays visible.
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(raw[:ng].max()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, ax in enumerate(axes.ravel()):
        if k < ng:
            im = ax.imshow(raw[k], origin="lower", cmap="inferno", norm=norm)
            ax.set_title(f"raw frame {k}   (peak {raw[k].max():.0f} ADU)", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.set_visible(False)
    fig.suptitle("Simulated RAW frames (signal + shot + dark)", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "imaging_frames_raw_gallery.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(ds_total, "o-", ms=3, label="dark-subtracted total counts")
    ax.axhline(TARGET_SIGNAL_ADU, color="r", ls="--", label=f"target = {TARGET_SIGNAL_ADU:.2e} ADU")
    ax.set_xlabel("simulated frame index"); ax.set_ylabel("total counts in frame [ADU]")
    ax.set_title(f"Per-frame flux (injected scintillation {100*FLUX_JITTER_RMS:.1f}% RMS)",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "imaging_frames_flux.png"), dpi=120)
    plt.close(fig)
    print(f"[sim-f] figures -> {os.path.relpath(FIGURES_DIR, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
