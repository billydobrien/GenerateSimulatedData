"""
Generate CLEAN broadband imaging frames (simulated-data pipeline) with HCIPy.

For each ground-truth OPD screen (from d_), build the noiseless detector image:

    image = SUM_lambda  w(lambda) * PSF_lambda(OPD + W020(lambda)*rho^2, dispersion)

  * W020(lambda): chromatic defocus of the 100/400 mm achromat relay, from the
    AC254 BFD curves (ported from the pipeline; beam radius from the CDK700).
  * dispersion: differential atmospheric refraction, set by YOUR altitude and
    zenith-direction-on-detector.
  * plate scale: COMPUTED from the optics (EFL = f_tel * f_foc/f_col) and the
    Zyla pixel; asserted to match ~0.074"/px.
  * w(lambda): Arcturus SED x imaging-leg throughput, integrated over each of
    200 sub-bands at the native 0.5 nm resolution (keeps line structure).

Noise / photon budget are intentionally NOT here — that is a separate f_ step.
Exposure & frame rate are recorded for that step; intra-exposure motion blur is
deferred (1 ms ~ 3.6 px, small).

Run:  python e_generateImagingFrames.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as nd_shift
import hcipy

# ── Telescope + relay (CDK700) ───────────────────────────────────────────────
D_TEL_M      = 0.70
F_TEL_MM     = 4540.0        # CDK700 focal length (f/6.5)
F_COL_MM     = 100.0         # collimator
F_FOC_MM     = 400.0         # focuser
RELAY_SEP_MM = 233.0         # lens separation d_12
FOCUS_REF_WL = 516.0         # wavelength in focus, W020 = 0 [nm]
PIXEL_UM     = 6.5           # Zyla pixel [micron]

# ── Wavelengths ──────────────────────────────────────────────────────────────
WL_START_NM  = 320.0
WL_STOP_NM   = 1000.0
N_SUBBANDS   = 200

# ── Dispersion (SET THESE) ───────────────────────────────────────────────────
ALTITUDE_DEG      = 73.8     # <-- observational altitude
ZENITH_ON_DET_DEG = 240.0    # <-- zenith direction on detector
DISP_REF_WL_NM    = 627.0
ELEVATION_M  = 219.0
AIR_TEMP_F   = 59.2
RH_PCT       = 72.1

# ── Detector / run ───────────────────────────────────────────────────────────
FOCAL_FIELD_PX = 512         # PSF computed over this field (captures blue spill)
CROP           = 256         # saved frame size (matches real data)
MAX_FRAMES     = int(os.environ["SIM_N_FRAMES"]) if os.environ.get("SIM_N_FRAMES") else None  # run_all.bat sets it; None = all
EXPTIME_S      = 0.001       # metadata (photon budget -> f_)
FRAME_RATE_HZ  = 70.0        # metadata

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
CFG_DIR     = os.path.join(SCRIPT_DIR, "configFiles")
PUPIL_FILE  = os.path.join(OUTPUT_DIR, "pupil_model_0.70m.fits")
OPD_FILE    = os.path.join(OUTPUT_DIR, "phase_screens_opd_nm.fits")
SED_FILE    = os.path.join(OUTPUT_DIR, "object_spectrum_arcturus_320-1000nm_0.5nm.fits")
TPUT_FILE   = os.path.join(OUTPUT_DIR, "system_throughput_imaging.fits")


# ── Chromatic defocus W020 (ported from f_initialPSFwithDispersion) ──────────
def _load_bfd_curve(filename):
    with open(os.path.join(CFG_DIR, filename)) as f:
        d = json.load(f)
    return np.array(d["wavelength_nm"]), np.array(d["bfd_shift_mm"])


def _compute_delta_z(wl_nm, bc_wl, bc, bf_wl, bf, f_c, f_f, d12):
    dc = np.interp(wl_nm, bc_wl, bc)
    df = np.interp(wl_nm, bf_wl, bf)
    f_col, f_foc = f_c + dc, f_f + df
    u = -f_c
    v1 = np.where(np.abs(1 / f_col + 1 / u) > 1e-15, 1 / (1 / f_col + 1 / u), 1e15)
    u2 = v1 - d12
    return 1 / (1 / f_foc + 1 / u2)


def compute_defocus_w020_nm(wl_nm):
    r_beam = F_COL_MM * (D_TEL_M * 1e3) / (2 * F_TEL_MM)        # collimated beam radius [mm]
    bc_wl, bc = _load_bfd_curve("ac254-100-a_flshift.json")
    bf_wl, bf = _load_bfd_curve("ac254-400-a_flshift.json")
    v2  = _compute_delta_z(wl_nm, bc_wl, bc, bf_wl, bf, F_COL_MM, F_FOC_MM, RELAY_SEP_MM)
    v2r = _compute_delta_z(np.array([FOCUS_REF_WL]), bc_wl, bc, bf_wl, bf,
                           F_COL_MM, F_FOC_MM, RELAY_SEP_MM)[0]
    delta_z = v2 - v2r
    return delta_z * r_beam ** 2 / (2 * F_FOC_MM ** 2) * 1e6    # nm


# ── Atmospheric dispersion (ported) ──────────────────────────────────────────
def refractive_index_air(wl_nm, p_pa, t_k, rh):
    s2 = (1.0 / (wl_nm * 1e-3)) ** 2
    n_std = 8342.54 + 2406147.0 / (130.0 - s2) + 15998.0 / (38.9 - s2)
    n_dry = n_std * (p_pa / 101325.0) * (288.15 / t_k)
    t_c = t_k - 273.15
    e_w = (rh / 100.0) * 610.78 * np.exp(17.27 * t_c / (t_c + 237.3))
    n_water = -43.49 * (1.0 - 7.956e-3 * s2) * (e_w / 101325.0)
    return 1.0 + (n_dry + n_water) * 1e-8


def dispersion_shifts_px(wl_nm, plate_asec):
    z = np.deg2rad(90.0 - ALTITUDE_DEG)
    p_pa = 101325.0 * np.exp(-ELEVATION_M / 8500.0)
    t_k = (AIR_TEMP_F - 32.0) * 5.0 / 9.0 + 273.15
    n_l = refractive_index_air(wl_nm, p_pa, t_k, RH_PCT)
    n_r = refractive_index_air(DISP_REF_WL_NM, p_pa, t_k, RH_PCT)
    dr_px = (n_l - n_r) * np.tan(z) * (180.0 * 3600.0 / np.pi) / plate_asec
    a = np.deg2rad(ZENITH_ON_DET_DEG)
    return -dr_px * np.sin(a), dr_px * np.cos(a)      # row, col


def load_sed_or_tput(path, is_table):
    if is_table:
        d = fits.getdata(path)
        return np.asarray(d["WAVELENGTH"], float), np.asarray(d["EFF_TRANS"], float)
    with fits.open(path) as h:
        flux = np.asarray(h[0].data, float); hd = h[0].header
        wl = hd["CRVAL1"] + (np.arange(flux.size) + 1 - hd["CRPIX1"]) * hd["CDELT1"]
    return wl, flux


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Plate scale from optics, verified against 0.074 ──────────────────────
    efl_mm = F_TEL_MM * (F_FOC_MM / F_COL_MM)
    plate_rad = (PIXEL_UM * 1e-3) / efl_mm
    plate_asec = plate_rad * 206264.806
    print(f"[sim-e] EFL={efl_mm:.0f} mm  ->  plate scale = {plate_asec:.4f} arcsec/px")
    assert abs(plate_asec - 0.074) < 0.005, "plate scale far from 0.074 — check optics"

    # ── Load inputs ──────────────────────────────────────────────────────────
    with fits.open(PUPIL_FILE) as h:
        pupil_mask = np.asarray(h[0].data, float); ph = h[0].header
        N = int(ph["ARRSIZE"]); D_grid = N * float(ph["PIXSCALE"])   # full grid extent [m] (1.4)
    with fits.open(OPD_FILE) as h:
        opd_cube = np.asarray(h[0].data, float)        # (Nf, N, N) nm
    sed_wl, sed = load_sed_or_tput(SED_FILE, is_table=False)
    tp_wl,  tput = load_sed_or_tput(TPUT_FILE, is_table=True)

    n_frames = opd_cube.shape[0] if MAX_FRAMES is None else min(MAX_FRAMES, opd_cube.shape[0])

    # ── Sub-bands (uniform in wavenumber) + integrated SED x throughput weight ─
    k_edges = np.linspace(1/WL_START_NM, 1/WL_STOP_NM, N_SUBBANDS + 1)
    wl_lo = 1.0 / k_edges[:-1]; wl_hi = 1.0 / k_edges[1:]
    wl_cen = 2.0 / (k_edges[:-1] + k_edges[1:])        # harmonic centre [nm]
    sed_x_tp = sed * np.interp(sed_wl, tp_wl, tput)    # on the 0.5 nm grid
    weights = np.array([sed_x_tp[(sed_wl >= lo) & (sed_wl < hi)].sum()
                        for lo, hi in zip(np.minimum(wl_lo, wl_hi), np.maximum(wl_lo, wl_hi))])
    weights /= weights.sum()

    w020 = compute_defocus_w020_nm(wl_cen)             # nm, per sub-band
    row_sh, col_sh = dispersion_shifts_px(wl_cen, plate_asec)
    print(f"[sim-e] {N_SUBBANDS} sub-bands 320-1000 nm | W020 {w020.min():.0f}..{w020.max():.0f} nm "
          f"| disp row {row_sh.min():.2f}..{row_sh.max():.2f}px col {col_sh.min():.2f}..{col_sh.max():.2f}px")

    # ── HCIPy grids / propagator ─────────────────────────────────────────────
    pupil_grid = hcipy.make_pupil_grid(N, D_grid)
    aperture   = hcipy.Field(pupil_mask.ravel(), pupil_grid)
    r_grid     = np.hypot(pupil_grid.x, pupil_grid.y)
    rho2       = hcipy.Field((r_grid / (D_TEL_M / 2.0)) ** 2, pupil_grid)
    pix_m      = PIXEL_UM * 1e-6
    focal_grid = hcipy.make_uniform_grid([FOCAL_FIELD_PX, FOCAL_FIELD_PX],
                                         [FOCAL_FIELD_PX * pix_m, FOCAL_FIELD_PX * pix_m])
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=efl_mm * 1e-3)

    lo = (FOCAL_FIELD_PX - CROP) // 2
    show_idx = np.linspace(0, N_SUBBANDS - 1, 12).astype(int)   # 12 wavelengths to plot
    mono_unw, mono_wt = [], []                                   # frame-0 mono PSFs

    img_cube = np.empty((n_frames, CROP, CROP), dtype=np.float32)
    crop_frac = []
    for fi in range(n_frames):
        broad = np.zeros((FOCAL_FIELD_PX, FOCAL_FIELD_PX))
        opd_arr = opd_cube[fi].ravel()
        for k in range(N_SUBBANDS):
            wl_m = wl_cen[k] * 1e-9
            phase = 2 * np.pi * (opd_arr + w020[k] * rho2) * 1e-9 / wl_m
            wf = hcipy.Wavefront(aperture * np.exp(1j * phase), wl_m)
            psf = prop(wf).power.shaped
            psf /= psf.sum()
            psf = nd_shift(psf, (row_sh[k], col_sh[k]), order=1, mode="constant")
            broad += weights[k] * psf
            if fi == 0 and k in show_idx:
                mono_unw.append(psf[lo:lo+CROP, lo:lo+CROP].copy())
                mono_wt.append((weights[k] * psf)[lo:lo+CROP, lo:lo+CROP].copy())
        crop_frac.append(broad[lo:lo+CROP, lo:lo+CROP].sum() / broad.sum())
        img_cube[fi] = broad[lo:lo+CROP, lo:lo+CROP].astype(np.float32)
        print(f"[sim-e]   frame {fi}: energy in {CROP}px crop = {100*crop_frac[-1]:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    path = os.path.join(OUTPUT_DIR, "imaging_frames_clean.fits")
    hdr = fits.Header()
    hdr["BUNIT"]   = ("relative", "Clean broadband image, sum~1 per frame (no noise)")
    hdr["NFRAMES"] = (n_frames, "Frames")
    hdr["NSUBND"]  = (N_SUBBANDS, "Wavelength sub-bands")
    hdr["PLATEASC"] = (round(plate_asec, 5), "Plate scale [arcsec/px]")
    hdr["EFL_MM"]  = (efl_mm, "Effective focal length [mm]")
    hdr["ALT_DEG"] = (ALTITUDE_DEG, "Observational altitude [deg]")
    hdr["ZA_DET"]  = (ZENITH_ON_DET_DEG, "Zenith dir on detector [deg]")
    hdr["FOCREF"]  = (FOCUS_REF_WL, "W020=0 wavelength [nm]")
    hdr["EXPTIME"] = (EXPTIME_S, "Exposure [s] (for f_ photon budget)")
    hdr["FRATE"]   = (FRAME_RATE_HZ, "Frame rate [Hz]")
    fits.writeto(path, img_cube, header=hdr, overwrite=True)
    print(f"[sim-e] clean imaging frames saved -> {os.path.relpath(path, SCRIPT_DIR)}  {img_cube.shape}")

    # ── Plot 1: broadband gallery, first 6 frames ────────────────────────────
    n_g = min(6, n_frames)
    vmax = np.percentile(img_cube[:n_g], 99.8)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, ax in enumerate(axes.ravel()):
        if k < n_g:
            im = ax.imshow(img_cube[k], origin="lower", cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"frame {k}", fontweight="bold"); fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.set_visible(False)
    fig.suptitle("Clean broadband imaging frames (shared scale)", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "imaging_frames_gallery.png"), dpi=120)
    plt.close(fig)

    # ── Plots 2 & 3: frame-0 monochromatic PSFs, 4x3, shared scale ───────────
    def grid_plot(psfs, title, fname):
        vmx = max(p.max() for p in psfs)
        fig, axes = plt.subplots(3, 4, figsize=(18, 13))
        for j, ax in enumerate(axes.ravel()):
            im = ax.imshow(psfs[j], origin="lower", cmap="inferno", vmin=0, vmax=vmx)
            ax.set_title(f"λ = {wl_cen[show_idx[j]]:.0f} nm", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=120); plt.close(fig)

    grid_plot(mono_unw, "Frame 0 — monochromatic PSFs (sum-normalised, shared scale)",
              "imaging_mono_psfs_unweighted.png")
    grid_plot(mono_wt, "Frame 0 — monochromatic PSFs x (SED x throughput) weight (shared scale)",
              "imaging_mono_psfs_weighted.png")
    print(f"[sim-e] figures -> {os.path.relpath(FIGURES_DIR, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
