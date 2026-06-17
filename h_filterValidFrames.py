"""
Pair + filter the simulated imaging and WFS frames to the VALID (unsaturated)
subset, keeping the two cubes frame-for-frame aligned.

Imaging and WFS are simultaneous (same OPD screen, same index), so a frame is
usable only if BOTH are unsaturated.  In practice the WFS spots are faint and
never saturate, so the binding constraint is the imaging peak — but we check
both and keep the intersection.

Outputs the aligned valid cubes plus a manifest that maps each kept frame back
to its ground-truth OPD screen index (so the recovery test can compare the
pipeline's retrieved OPD/spectrum against the known truth).

Requires imaging and WFS cubes with the SAME frame count (run e_/f_ and g_ on
the same MAX_FRAMES / OPD screens).

Run:  python h_filterValidFrames.py
"""
import os
import numpy as np
from astropy.io import fits

SAT_LEVEL = 4095.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fitsOutputs")
IMG_RAW    = os.path.join(OUTPUT_DIR, "imaging_frames_raw.fits")
WFS_RAW    = os.path.join(OUTPUT_DIR, "wfs_frames_raw.fits")
GT_FLUX    = os.path.join(OUTPUT_DIR, "ground_truth_flux.npz")


def main():
    img = np.asarray(fits.getdata(IMG_RAW), float)
    wfs = np.asarray(fits.getdata(WFS_RAW), float)
    if img.shape[0] != wfs.shape[0]:
        raise SystemExit(
            f"[sim-h] frame-count mismatch: imaging {img.shape[0]} vs WFS {wfs.shape[0]}.\n"
            f"        Run e_/f_ and g_ with the SAME MAX_FRAMES so the frames pair up.")
    nf = img.shape[0]

    img_peak = img.max(axis=(1, 2))
    wfs_peak = wfs.max(axis=(1, 2))
    img_ok = img_peak < SAT_LEVEL
    wfs_ok = wfs_peak < SAT_LEVEL
    valid  = img_ok & wfs_ok
    idx    = np.where(valid)[0]

    print(f"[sim-h] {nf} frames | imaging unsat {img_ok.sum()} ({100*img_ok.mean():.1f}%) | "
          f"WFS unsat {wfs_ok.sum()} ({100*wfs_ok.mean():.1f}%)")
    print(f"[sim-h] BOTH valid: {valid.sum()}/{nf} ({100*valid.mean():.1f}%)  "
          f"-> kept frame indices align imaging<->WFS<->OPD screens")

    # ── Save aligned valid cubes ─────────────────────────────────────────────
    for cube, name in ((img[valid], "imaging_frames_valid.fits"),
                       (wfs[valid], "wfs_frames_valid.fits")):
        hdr = fits.Header()
        hdr["NFRAMES"] = (int(valid.sum()), "Valid (unsaturated) frames")
        hdr["NTOTAL"]  = (nf, "Frames before filtering")
        hdr["SATLEVEL"] = (SAT_LEVEL, "ADC saturation level [ADU]")
        hdr["PAIRED"]  = (True, "Imaging & WFS share these frame indices")
        fits.writeto(os.path.join(OUTPUT_DIR, name), cube.astype(np.float32), header=hdr, overwrite=True)
        print(f"[sim-h] {name:30s} -> {cube.shape}")

    # ── Manifest: map kept frames back to the ground truth ───────────────────
    man = {"valid_indices": idx, "n_total": nf, "n_valid": int(valid.sum())}
    if os.path.exists(GT_FLUX):
        gt = np.load(GT_FLUX)
        if "flux_scalar" in gt and gt["flux_scalar"].size >= nf:
            man["flux_scalar"] = gt["flux_scalar"][:nf][valid]
    np.savez(os.path.join(OUTPUT_DIR, "valid_frames.npz"), **man)
    print(f"[sim-h] manifest (valid_indices -> OPD screens) -> "
          f"{os.path.relpath(os.path.join(OUTPUT_DIR, 'valid_frames.npz'), SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
