"""
Generate a synthetic telescope pupil mask for the simulated-data pipeline.

Mirrors NewDataPipeline/a_generatePupilModel.py — same annulus + crossed-spider
geometry and the same r_primary = grid/4 convention — but is self-contained (no
real pupil-frame dependency) and records the PHYSICAL aperture scale so that
downstream HCIPy propagation knows metres-per-pixel.

A pupil mask itself is dimensionless (0/1 pixels).  "0.7 m" enters only as
metadata: the physical primary diameter (DIAM_M) and the pupil-plane sampling
(PIXSCALE = D / diameter_in_pixels).

Run:  python a_generatePupilModel.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ── Config (matches NewDataPipeline geometry) ────────────────────────────────
# A physical pupil carries no wavelength: it is just the 0.7 m aperture in
# metres.  Wavelength enters only downstream, at propagation time.
PUPIL_GRID_SIZE      = 512      # array side [px]
APERTURE_DIAMETER_M  = 0.70     # physical primary diameter [m]  (HLCO)
OBSTRUCTION_FRACTION = 0.47     # secondary diam / primary diam
SPIDER_THETA_DEG     = 60.5     # spider rotation from north [deg]
SPIDER_WIDTH_PIXELS  = 3        # spider vane full width [px]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")


def make_pupil_model(array_size, r_primary, obstruction_fraction,
                     spider_theta_deg, spider_width_pixels):
    """Binary annular pupil with two crossed spider vanes.

    Identical geometry to NewDataPipeline.a_generatePupilModel.make_pupil_model
    so the simulated pupil is drop-in compatible with the pipeline.
    """
    cx = cy = array_size / 2.0
    r_secondary = r_primary * obstruction_fraction
    half_width = spider_width_pixels / 2.0

    yy, xx = np.ogrid[:array_size, :array_size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (r <= r_primary).astype(np.float64)
    mask[r <= r_secondary] = 0.0

    theta = np.radians(spider_theta_deg)
    yyf, xxf = np.mgrid[:array_size, :array_size]
    dx, dy = xxf - cx, yyf - cy
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = cos_t * dx - sin_t * dy
    y_rot = sin_t * dx + cos_t * dy
    inside = r <= r_primary
    mask[(np.abs(x_rot) <= half_width) & inside] = 0.0
    mask[(np.abs(y_rot) <= half_width) & inside] = 0.0
    return mask


def main():
    r_primary   = PUPIL_GRID_SIZE / 4.0             # diameter = grid/2 (pipeline convention)
    r_secondary = r_primary * OBSTRUCTION_FRACTION
    diam_px     = 2.0 * r_primary
    pix_scale_m = APERTURE_DIAMETER_M / diam_px      # metres per pixel

    model = make_pupil_model(PUPIL_GRID_SIZE, r_primary, OBSTRUCTION_FRACTION,
                             SPIDER_THETA_DEG, SPIDER_WIDTH_PIXELS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"pupil_model_{APERTURE_DIAMETER_M:.2f}m.fits"
    path  = os.path.join(OUTPUT_DIR, fname)

    hdr = fits.Header()
    hdr["BUNIT"]    = ("binary", "Binary pupil mask (0/1)")
    hdr["ARRSIZE"]  = (PUPIL_GRID_SIZE, "Array side length [px]")
    hdr["PUPRAD"]   = (r_primary, "Primary radius [px]")
    hdr["SECRAD"]   = (round(r_secondary, 4), "Secondary radius [px]")
    hdr["OBSFRAC"]  = (OBSTRUCTION_FRACTION, "Secondary diam / primary diam")
    hdr["SPTHETA"]  = (SPIDER_THETA_DEG, "Spider rotation from north [deg]")
    hdr["SPWIDTH"]  = (SPIDER_WIDTH_PIXELS, "Spider vane full width [px]")
    hdr["DIAM_M"]   = (APERTURE_DIAMETER_M, "Physical primary diameter [m]")
    hdr["PIXSCALE"] = (pix_scale_m, "Pupil-plane sampling [m/px]")
    fits.writeto(path, model.astype(np.float32), header=hdr, overwrite=True)

    print(f"[sim-a] Pupil model saved -> {os.path.relpath(path, SCRIPT_DIR)}")
    print(f"        grid {PUPIL_GRID_SIZE} px | D = {APERTURE_DIAMETER_M} m | "
          f"diameter = {diam_px:.0f} px | scale = {1e3 * pix_scale_m:.4f} mm/px")
    print(f"        obstruction = {OBSTRUCTION_FRACTION} (sec radius {r_secondary:.1f} px), "
          f"spiders {SPIDER_THETA_DEG} deg / {SPIDER_WIDTH_PIXELS} px")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(model, cmap="gray", origin="lower")
    ax.set_title(f"Simulated pupil — D = {APERTURE_DIAMETER_M} m  "
                 f"({1e3 * pix_scale_m:.3f} mm/px)", fontweight="bold")
    ax.set_xlabel("pixels"); ax.set_ylabel("pixels")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png = os.path.join(FIGURES_DIR, "pupil_model_diagnostic.png")
    fig.savefig(png, dpi=120)
    print(f"        diagnostic plot -> {os.path.relpath(png, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
