"""
Generate ground-truth atmospheric OPD phase screens with HCIPy (>= 0.7.0).

These are the TRUE per-frame wavefronts for the simulated data — the analogue of
the pipeline's modal_wavefronts_OPD cube, but here we KNOW them exactly.

Design (see notes below):
  * UNITS = optical path difference (OPD) in nanometres.  Turbulence is an
    achromatic path-length perturbation; phase is chromatic (phi = 2*pi*OPD/lambda).
    HCIPy works in phase at a reference wavelength, so OPD = phi * lambda_ref/(2*pi).
    Downstream each wavelength applies its own phi = 2*pi*OPD/lambda, which makes
    the blue automatically more aberrated (r0 ~ lambda^6/5).
  * r0 = 3 cm AT 400 nm (r0 is wavelength-dependent, so the reference matters).
  * GRID matches pupil_model_0.70m.fits: 512 px spanning 1.4 m (2.734 mm/px),
    0.7 m aperture in the central 256 px.  ~11 px per r0 -> well sampled.
  * SINGLE layer, HEIGHT = 0.  Height only matters for scintillation (Fresnel
    between layers) and anisoplanatism (off-axis), neither of which is in the
    pipeline's single-pupil-plane Fraunhofer forward model.
  * Wind 10 m/s, frozen flow evolved at the frame cadence -> time-correlated
    frames like real speckle data.  Outer scale L0 = 25 m (von Karman).
  * Tip/tilt is KEPT (it is the true wavefront); downstream corrects it.

Run:  python d_generatePhaseScreens.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import hcipy

# ── Config ───────────────────────────────────────────────────────────────────
R0_M        = 0.020         # Fried parameter [m]
R0_REF_WL   = 400e-9        # wavelength at which r0 is defined [m]
L0_M        = 25.0          # outer scale [m] (von Karman)
WIND_MS     = np.array([10.0, 0.0])   # frozen-flow wind vector [m/s]
N_FRAMES    = int(os.environ.get("SIM_N_FRAMES", 50))   # OPD screens (run_all.bat sets SIM_N_FRAMES)
FRAME_DT    = 1.0 / 70.0    # time between frames [s] (data is 70 fps)
SEED        = 1

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
PUPIL_FILE  = os.path.join(OUTPUT_DIR, "pupil_model_0.70m.fits")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    np.random.seed(SEED)

    # ── Grid from the pupil model (keep everything consistent) ───────────────
    with fits.open(PUPIL_FILE) as hdul:
        pupil_mask = np.asarray(hdul[0].data, dtype=np.float64)
        ph         = hdul[0].header
        N          = int(ph["ARRSIZE"])
        diam_m     = float(ph["DIAM_M"])          # 0.7 m aperture
        pix_m      = float(ph["PIXSCALE"])        # m/px
    D_grid = N * pix_m                            # full grid extent [m] (1.4 m)
    print(f"[sim-d] grid {N}px, extent {D_grid:.4f} m, {1e3*pix_m:.4f} mm/px, "
          f"aperture {diam_m} m")

    pupil_grid = hcipy.make_pupil_grid(N, D_grid)
    aperture   = pupil_mask > 0.5                 # boolean aperture (for diagnostics)

    # ── Single frozen-flow turbulence layer ──────────────────────────────────
    cn2 = hcipy.Cn_squared_from_fried_parameter(R0_M, R0_REF_WL)
    layer = hcipy.InfiniteAtmosphericLayer(
        pupil_grid, cn2, L0=L0_M, velocity=WIND_MS, height=0.0, seed=SEED)
    print(f"[sim-d] r0={100*R0_M:.1f} cm @ {1e9*R0_REF_WL:.0f} nm  "
          f"(D/r0={diam_m/R0_M:.1f})  L0={L0_M} m  wind={WIND_MS} m/s")

    # ── Generate frames: OPD [nm] from phase at the reference wavelength ──────
    opd_cube = np.empty((N_FRAMES, N, N), dtype=np.float32)
    nm_per_rad = R0_REF_WL / (2 * np.pi) * 1e9     # phase[rad] -> OPD[nm]
    rms_full, rms_ho = [], []
    yy, xx = np.mgrid[:N, :N].astype(float)
    A_tt = np.c_[np.ones(aperture.sum()), xx[aperture], yy[aperture]]  # piston+tilt basis

    for i in range(N_FRAMES):
        layer.evolve_until(i * FRAME_DT)
        phase = np.asarray(layer.phase_for(R0_REF_WL)).reshape(N, N)
        opd = phase * nm_per_rad
        opd = opd - opd[aperture].mean()           # remove piston
        opd_cube[i] = opd.astype(np.float32)

        rms_full.append(np.sqrt(np.mean(opd[aperture] ** 2)))
        coef, *_ = np.linalg.lstsq(A_tt, opd[aperture], rcond=None)
        resid = opd[aperture] - A_tt @ coef        # tip/tilt removed
        rms_ho.append(np.sqrt(np.mean(resid ** 2)))

    rms_full, rms_ho = np.array(rms_full), np.array(rms_ho)

    # ── Sanity: r0 + slope from the multi-frame phase structure function ─────
    # D_phi(r) = <(phi(x+r) - phi(x))^2> averaged over the aperture AND all
    # frames.  Kolmogorov inertial range: D_phi = 6.88 (r/r0)^(5/3).  Fitting
    # over several separations (<< L0) is robust — many pairs x many frames, and
    # high-freq differences decorrelate across the wind-blown frames.
    phase_cube = opd_cube / nm_per_rad             # rad @ ref wl
    seps = np.array([8, 12, 16, 24, 32, 48])       # px
    Dphi = np.empty(len(seps))
    for j, s in enumerate(seps):
        d = phase_cube[:, :, s:] - phase_cube[:, :, :-s]
        m = aperture[:, s:] & aperture[:, :-s]
        Dphi[j] = np.mean(d[:, m] ** 2)
    r_m = seps * pix_m
    slope = np.polyfit(np.log(r_m), np.log(Dphi), 1)[0]
    r0_sf = float(np.median(r_m / (Dphi / 6.88) ** 0.6))
    print(f"[sim-d] structure-function r0={100*r0_sf:.2f} cm (target {100*R0_M:.1f}), "
          f"slope={slope:.2f} (Kolmogorov 1.67)  [avg over {N_FRAMES} frames]")
    print(f"[sim-d] OPD RMS over aperture: full {rms_full.mean():.0f} nm  "
          f"(tip/tilt removed {rms_ho.mean():.0f} nm)")

    # ── Save cube ────────────────────────────────────────────────────────────
    path = os.path.join(OUTPUT_DIR, "phase_screens_opd_nm.fits")
    hdr = fits.Header()
    hdr["BUNIT"]   = ("nm", "Optical path difference (achromatic)")
    hdr["NFRAMES"] = (N_FRAMES, "Number of OPD screens")
    hdr["R0_M"]    = (R0_M, "Fried parameter [m]")
    hdr["R0_REFWL"] = (1e9 * R0_REF_WL, "Wavelength r0 is defined at [nm]")
    hdr["L0_M"]    = (L0_M, "Outer scale [m]")
    hdr["WIND_X"]  = (WIND_MS[0], "Wind vector x [m/s]")
    hdr["WIND_Y"]  = (WIND_MS[1], "Wind vector y [m/s]")
    hdr["FRAME_DT"] = (FRAME_DT, "Time between frames [s]")
    hdr["PIXSCALE"] = (pix_m, "Grid sampling [m/px]")
    hdr["DGRID_M"] = (D_grid, "Full grid extent [m]")
    hdr["APER_M"]  = (diam_m, "Aperture diameter [m]")
    hdr["SEED"]    = (SEED, "RNG seed")
    fits.writeto(path, opd_cube, header=hdr, overwrite=True)
    print(f"[sim-d] OPD screens saved -> {os.path.relpath(path, SCRIPT_DIR)}  ({opd_cube.shape})")

    # ── Figures ──────────────────────────────────────────────────────────────
    masked = np.where(aperture[None], opd_cube, np.nan)
    vlim = np.nanpercentile(np.abs(masked[:6]), 99)
    n_show = min(6, N_FRAMES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, ax in enumerate(axes.ravel()[:n_show]):
        im = ax.imshow(masked[k], origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"frame {k}  (t={k*FRAME_DT*1e3:.1f} ms)", fontweight="bold")
        fig.colorbar(im, ax=ax, label="OPD [nm]", fraction=0.046)
    fig.suptitle(f"Ground-truth atmospheric OPD screens — r0={100*R0_M:.0f} cm @ "
                 f"{1e9*R0_REF_WL:.0f} nm, wind {WIND_MS[0]:.0f} m/s",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase_screens_gallery.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(N_FRAMES) * FRAME_DT * 1e3, rms_full, "o-", label="full OPD RMS")
    ax.plot(np.arange(N_FRAMES) * FRAME_DT * 1e3, rms_ho, "s-", label="tip/tilt removed")
    ax.set_xlabel("time [ms]"); ax.set_ylabel("OPD RMS over aperture [nm]")
    ax.set_title("Per-frame wavefront error", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase_screens_rms.png"), dpi=120)
    plt.close(fig)
    print(f"[sim-d] figures -> {os.path.relpath(FIGURES_DIR, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
