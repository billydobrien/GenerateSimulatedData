"""
Initialize the ground-truth OBJECT spectrum for the simulated-data pipeline.

Unlike NewDataPipeline/b_initializeObject.py (a blackbody on a 512x512 cube),
the simulation needs the *true* input SED to inject — so here the object is the
real stitched Arcturus reference (MELCHIORS core + UVES-POP wings, the same
spectrum m_plotSpectrum.py builds), resampled onto a fine 300-1000 nm / 1 nm
grid.  No spatial cube: Arcturus is an unresolved point source, so the object is
just the 1-D spectrum array.

Resampling is by BINNING (mean flux within each 1 nm window), which is flux-
conserving and faithful for a high-resolution, line-rich spectrum.  The 300-319
nm region has no reference data (UVES-POP starts ~320 nm) and is flat-filled
from the bluest available bin.  The SED is sum-normalised to 1.

Run:  python b_initializeObject.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ── Config ───────────────────────────────────────────────────────────────────
WL_START_NM = 320.0     # reference (UVES-POP) coverage begins ~320 nm
WL_STOP_NM  = 1000.0
WL_STEP_NM  = 0.5

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
_CFG        = os.path.join(SCRIPT_DIR, "configFiles")   # spectra copied here

MELCHIORS_PATH = os.path.join(_CFG, "327245_melchiors_spectrum.fits")
ARCTURUS_PATH  = os.path.join(_CFG, "Arcturus.fits")
FLUX_COLUMN    = "flux_tac"      # instrument + telluric corrected (truest SED)


# ── Arcturus reference loaders (copied from m_plotSpectrum.py) ────────────────
def load_melchiors_spectrum(spectrum_path, flux_column="flux_tac"):
    """MELCHIORS Arcturus spectrum (FITS table ext 1, 'wave' in Angstrom)."""
    with fits.open(spectrum_path) as hdul:
        tbl = hdul[1].data
        if flux_column not in tbl.names:
            raise KeyError(f"flux column {flux_column!r} not in {spectrum_path}. "
                           f"Available: {list(tbl.names)}")
        wave_nm = np.asarray(tbl["wave"], dtype=np.float64) / 10.0
        flux    = np.asarray(tbl[flux_column], dtype=np.float64)
    finite = np.isfinite(wave_nm) & np.isfinite(flux)
    wave_nm, flux = wave_nm[finite], flux[finite]
    order = np.argsort(wave_nm)
    return wave_nm[order], flux[order]


def load_stitched_spectrum(melchiors_path, arcturus_path, flux_column="flux_tac"):
    """MELCHIORS core (376-900 nm) stitched with UVES-POP Arcturus.fits wings,
    scaled to match in a 50 nm overlap window on each side."""
    mel_wl, mel_flux = load_melchiors_spectrum(melchiors_path, flux_column=flux_column)
    mel_lo, mel_hi = float(mel_wl.min()), float(mel_wl.max())

    with fits.open(arcturus_path) as hdul:
        hdr      = hdul[0].header
        flux_arc = np.asarray(hdul[0].data, dtype=np.float64)
        n_pix    = int(hdr["NAXIS1"])
        crval1   = float(hdr["CRVAL1"]); cdelt1 = float(hdr["CDELT1"]); crpix1 = float(hdr["CRPIX1"])
    pixel  = np.arange(n_pix) + 1.0
    arc_wl = (crval1 + (pixel - crpix1) * cdelt1) / 10.0
    good   = np.isfinite(flux_arc) & np.isfinite(arc_wl)
    arc_wl, flux_arc = arc_wl[good], flux_arc[good]

    def _overlap_scale(lo, hi):
        m = (mel_wl >= lo) & (mel_wl <= hi)
        a = (arc_wl >= lo) & (arc_wl <= hi)
        if m.sum() < 10 or a.sum() < 10:
            return None
        a_med = float(np.median(flux_arc[a]))
        if not np.isfinite(a_med) or a_med == 0.0:
            return None
        return float(np.median(mel_flux[m])) / a_med

    sl = _overlap_scale(mel_lo, mel_lo + 50.0)
    sr = _overlap_scale(mel_hi - 50.0, mel_hi)
    if sl is None or sr is None:
        print("[sim-b] Warning: insufficient overlap — using MELCHIORS only.")
        return mel_wl, mel_flux

    lm, rm = arc_wl < mel_lo, arc_wl > mel_hi
    wave = np.concatenate([arc_wl[lm], mel_wl, arc_wl[rm]])
    flux = np.concatenate([flux_arc[lm] * sl, mel_flux, flux_arc[rm] * sr])
    order = np.argsort(wave)
    return wave[order], flux[order]


def bin_to_grid(wave_nm, flux, grid_nm, step):
    """Mean flux within each `step`-wide bin centred on grid_nm; empty bins
    (outside the reference coverage) are flat-filled by interpolation."""
    edges = np.concatenate([[grid_nm[0] - step / 2],
                            (grid_nm[:-1] + grid_nm[1:]) / 2,
                            [grid_nm[-1] + step / 2]])
    binned = np.full(len(grid_nm), np.nan)
    for i in range(len(grid_nm)):
        m = (wave_nm >= edges[i]) & (wave_nm < edges[i + 1])
        if m.any():
            binned[i] = flux[m].mean()
    valid = np.isfinite(binned)
    n_gap = int((~valid).sum())
    # interp over the gaps; np.interp flat-holds the endpoints for extrapolation
    binned = np.interp(grid_nm, grid_nm[valid], binned[valid])
    return np.clip(binned, 0.0, None), n_gap, grid_nm[valid].min(), grid_nm[valid].max()


def main():
    wave, flux = load_stitched_spectrum(MELCHIORS_PATH, ARCTURUS_PATH, FLUX_COLUMN)
    print(f"[sim-b] Stitched Arcturus reference: {wave.min():.1f}-{wave.max():.1f} nm, "
          f"{wave.size} samples")

    grid = np.arange(WL_START_NM, WL_STOP_NM + WL_STEP_NM / 2, WL_STEP_NM)
    sed, n_gap, cov_lo, cov_hi = bin_to_grid(wave, flux, grid, WL_STEP_NM)
    sed = sed / sed.sum()                       # sum-normalised SED

    print(f"[sim-b] Grid {WL_START_NM:.0f}-{WL_STOP_NM:.0f} nm @ {WL_STEP_NM:g} nm "
          f"= {grid.size} points | reference covers {cov_lo:.0f}-{cov_hi:.0f} nm "
          f"| {n_gap} empty bins interpolated (sparse reference / edges)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"object_spectrum_arcturus_{WL_START_NM:.0f}-{WL_STOP_NM:.0f}nm_{WL_STEP_NM:g}nm.fits"
    path = os.path.join(OUTPUT_DIR, fname)
    hdr = fits.Header()
    hdr["BUNIT"]   = ("relative", "Sum-normalised relative SED (sum=1)")
    hdr["OBJECT"]  = ("Arcturus", "Ground-truth point-source SED")
    hdr["SOURCE"]  = ("MELCHIORS+UVES-POP", "Stitched reference (m_plotSpectrum)")
    hdr["WLSTART"] = (WL_START_NM, "First wavelength [nm]")
    hdr["WLSTOP"]  = (WL_STOP_NM, "Last wavelength [nm]")
    hdr["WLSTEP"]  = (WL_STEP_NM, "Wavelength step [nm]")
    hdr["NORM"]    = ("sum=1", "Normalisation")
    hdr["NGAPFILL"] = (n_gap, "Bins flat-filled outside reference coverage")
    # Linear WCS so wavelength is recoverable: lambda = CRVAL1 + (i+1-CRPIX1)*CDELT1
    hdr["CTYPE1"] = ("WAVE", "Wavelength axis")
    hdr["CUNIT1"] = ("nm", "")
    hdr["CRPIX1"] = (1.0, "Reference pixel (1-indexed)")
    hdr["CRVAL1"] = (WL_START_NM, "Wavelength at reference pixel [nm]")
    hdr["CDELT1"] = (WL_STEP_NM, "Wavelength increment [nm/px]")
    fits.writeto(path, sed.astype(np.float64), header=hdr, overwrite=True)
    print(f"[sim-b] Object SED saved -> {os.path.relpath(path, SCRIPT_DIR)}  (sum={sed.sum():.4f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, sed, lw=0.8, color="darkred")
    if cov_lo > WL_START_NM + WL_STEP_NM:
        ax.axvspan(WL_START_NM, cov_lo, color="grey", alpha=0.2, label="flat-filled (no data)")
        ax.legend(fontsize=8)
    ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("relative flux (sum=1)")
    ax.set_title(f"Ground-truth Arcturus SED — {WL_START_NM:.0f}-{WL_STOP_NM:.0f} nm "
                 f"@ {WL_STEP_NM:g} nm", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png = os.path.join(FIGURES_DIR, "object_spectrum_diagnostic.png")
    fig.savefig(png, dpi=120)
    print(f"[sim-b] diagnostic plot -> {os.path.relpath(png, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
