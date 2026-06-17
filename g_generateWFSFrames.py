"""
Generate Shack-Hartmann WFS frames from the ground-truth OPD screens.

A SH frame is a 24x24 grid of spots, one per microlens.  Each spot is the
diffraction pattern (FFT) of that lenslet's patch of the pupil field
(aperture_mask x exp(i*phase)), so:
  * the secondary obstruction and spider vanes blank/dim the lenslets they
    cover -> dark central hole + dark cross in the spot grid (no special-casing);
  * a wavefront slope over a lenslet shifts its spot (the measurement);
  * the relay (MLA focal plane -> detector) is folded into the focal-plane
    sampling so each subaperture lands on 40 detector pixels.

UNITS: the OPD screens are achromatic path [nm]; phase = 2*pi*OPD/lambda at the
WFS effective wavelength (615.6 nm, the WFS-leg CoM).  Atmosphere is common-path
with the imaging leg, so this uses the SAME phase_screens_opd_nm.fits.

Clean spots only (no noise yet) -- analogous to e_ before f_.

Run:  python g_generateWFSFrames.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.io import fits

# ── WFS optics (from the layout) ─────────────────────────────────────────────
LAMBDA_WFS_NM    = 615.6     # WFS-leg CoM wavelength
N_LENS           = 24        # lenslets across the beam (central 24x24)
LENSLET_PITCH_MM = 0.30      # MLA300 pitch
LENSLET_FOCAL_MM = 14.2      # MLA300-14
RELAY_OBJ_MM     = 215.4     # spot plane -> relay lens
RELAY_IMG_MM     = 186.7     # relay lens -> camera
PIXEL_UM         = 6.5       # Zyla pixel
DET_SUBAP_PX     = 40        # detector pixels per subaperture
DET_SIZE         = 1024      # WFS detector side

MAX_FRAMES = int(os.environ["SIM_N_FRAMES"]) if os.environ.get("SIM_N_FRAMES") else None  # run_all.bat sets it; None = all

# ── Photon budget + noise + dark ─────────────────────────────────────────────
GAIN                  = 0.46      # e-/ADU
WFS_TARGET_SIGNAL_ADU = 3.05e6    # real WFS dark-subtracted total/frame (measured)
READ_NOISE_E          = 2.5       # WFS read noise (Zyla); scalar (no WFS read map)
FLUX_JITTER_RMS       = 0.036     # scintillation (common-path with imaging)
SAT_LEVEL             = 4095.0    # ADC clamp (WFS spots are ~700, well below)
SEED                  = 1

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
PUPIL_FILE  = os.path.join(OUTPUT_DIR, "pupil_model_0.70m.fits")
OPD_FILE    = os.path.join(OUTPUT_DIR, "phase_screens_opd_nm.fits")
GT_FLUX_FILE = os.path.join(OUTPUT_DIR, "ground_truth_flux.npz")   # imaging flux scalars
WFS_DARK_FILE = os.path.join(SCRIPT_DIR, "darks", "average_dark_WFS_arcturus_3.fits")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

    with fits.open(PUPIL_FILE) as h:
        pupil = np.asarray(h[0].data, float); ph = h[0].header
        N = int(ph["ARRSIZE"]); r_pri = float(ph["PUPRAD"])    # aperture radius [px]
    opd_cube = np.asarray(fits.getdata(OPD_FILE), float)        # (Nf, N, N) nm
    nf = opd_cube.shape[0] if MAX_FRAMES is None else min(MAX_FRAMES, opd_cube.shape[0])

    relay_mag  = RELAY_IMG_MM / RELAY_OBJ_MM                    # 0.867
    det_foc_um = PIXEL_UM / relay_mag                          # detector sampling at MLA focal plane
    cx = cy    = N / 2.0
    pitch_px   = (2.0 * r_pri) / N_LENS                        # pupil px per lenslet
    delta_pup_um = LENSLET_PITCH_MM * 1e3 / pitch_px          # um per pupil px (lenslet scale)
    # FFT pad size so the focal-plane sampling = det_foc_um  -> subap lands on 40 px
    M = int(round(LAMBDA_WFS_NM * 1e-3 * LENSLET_FOCAL_MM * 1e3 / (det_foc_um * delta_pup_um)))
    P = int(round(pitch_px)) + 1                               # lenslet patch size [pupil px]
    f_eff = LENSLET_FOCAL_MM * relay_mag
    print(f"[sim-g] relay_mag={relay_mag:.3f}  f_eff={f_eff:.2f} mm  pitch={pitch_px:.2f}px/lenslet")
    print(f"[sim-g] FFT pad M={M}px -> crop {DET_SUBAP_PX}px/subap  (spot FWHM ~{0.88*M/P:.1f}px)")

    off = (DET_SIZE - N_LENS * DET_SUBAP_PX) // 2              # centre the grid in 1024
    lo  = (M - DET_SUBAP_PX) // 2

    wfs = np.zeros((nf, DET_SIZE, DET_SIZE), dtype=np.float32)
    for fi in range(nf):
        field = pupil * np.exp(1j * 2 * np.pi * opd_cube[fi] / LAMBDA_WFS_NM)
        det = np.zeros((DET_SIZE, DET_SIZE))
        for i in range(N_LENS):
            rc = cy - r_pri + (i + 0.5) * pitch_px            # lenslet centre row
            r0 = int(round(rc - P / 2))
            for j in range(N_LENS):
                cc = cx - r_pri + (j + 0.5) * pitch_px
                c0 = int(round(cc - P / 2))
                patch = field[r0:r0 + P, c0:c0 + P]
                if np.abs(patch).sum() < 1e-6:                # fully obstructed lenslet
                    continue
                pad = np.zeros((M, M), dtype=complex)
                pp = (M - P) // 2
                pad[pp:pp + P, pp:pp + P] = patch
                spot = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pad)))) ** 2
                cell = spot[lo:lo + DET_SUBAP_PX, lo:lo + DET_SUBAP_PX]
                R = off + i * DET_SUBAP_PX
                C = off + j * DET_SUBAP_PX
                det[R:R + DET_SUBAP_PX, C:C + DET_SUBAP_PX] = cell
        wfs[fi] = det.astype(np.float32)
        n_spots = int((det.reshape(-1) > 0).sum() > 0) and \
            sum(1 for i in range(N_LENS) for j in range(N_LENS)
                if det[off+i*DET_SUBAP_PX:off+(i+1)*DET_SUBAP_PX,
                       off+j*DET_SUBAP_PX:off+(j+1)*DET_SUBAP_PX].sum() > 0)
        print(f"[sim-g]   frame {fi}: {n_spots} illuminated lenslets", flush=True)

    chdr = fits.Header()
    chdr["BUNIT"]   = ("relative", "Clean SH-WFS spot intensity (no noise)")
    chdr["NFRAMES"] = (nf, "Frames"); chdr["WL_NM"] = (LAMBDA_WFS_NM, "WFS wavelength [nm]")
    chdr["FEFF_MM"] = (round(f_eff, 3), "Effective focal length [mm]")
    cpath = os.path.join(OUTPUT_DIR, "wfs_frames_clean.fits")
    fits.writeto(cpath, wfs, header=chdr, overwrite=True)
    print(f"[sim-g] clean WFS frames -> {os.path.relpath(cpath, SCRIPT_DIR)}  {wfs.shape}")

    # ── Photon budget + shot/read noise + dark -> raw WFS frames ─────────────
    rng = np.random.default_rng(SEED)
    wfs_dark = np.asarray(fits.getdata(WFS_DARK_FILE), float)
    if os.path.exists(GT_FLUX_FILE):
        s_all = np.load(GT_FLUX_FILE)["flux_scalar"]                 # common-path scintillation
        s_f = s_all[:nf] if s_all.size >= nf else rng.normal(1.0, FLUX_JITTER_RMS, nf)
    else:
        s_f = rng.normal(1.0, FLUX_JITTER_RMS, nf)
    read_adu = READ_NOISE_E / GAIN
    target_e = WFS_TARGET_SIGNAL_ADU * GAIN
    raw = np.empty_like(wfs)
    for fi in range(nf):
        expected_e = (wfs[fi] / wfs[fi].sum()) * (target_e * s_f[fi])
        noisy = rng.poisson(expected_e) / GAIN + wfs_dark + rng.normal(0, read_adu, wfs[fi].shape)
        raw[fi] = np.minimum(noisy, SAT_LEVEL).astype(np.float32)
    sig_peak = (raw - wfs_dark).max(axis=(1, 2))
    print(f"[sim-g] raw WFS: signal total mean {(raw - wfs_dark).sum(axis=(1,2)).mean():.3e} ADU "
          f"(target {WFS_TARGET_SIGNAL_ADU:.2e}) | spot peak mean {sig_peak.mean():.0f} ADU "
          f"(real ~600)")

    rhdr = fits.Header()
    rhdr["BUNIT"]   = ("ADU", "RAW SH-WFS: signal + shot + dark + read")
    rhdr["NFRAMES"] = (nf, "Frames"); rhdr["GAIN"] = (GAIN, "e-/ADU")
    rhdr["WL_NM"]   = (LAMBDA_WFS_NM, "WFS wavelength [nm]")
    rhdr["SIGADU"]  = (WFS_TARGET_SIGNAL_ADU, "Target dark-subtracted total [ADU]")
    rhdr["READ_E"]  = (READ_NOISE_E, "Read noise [e-]")
    rpath = os.path.join(OUTPUT_DIR, "wfs_frames_raw.fits")
    fits.writeto(rpath, raw, header=rhdr, overwrite=True)
    print(f"[sim-g] raw WFS frames   -> {os.path.relpath(rpath, SCRIPT_DIR)}  {raw.shape}")

    # ── Figure: raw spot grid (dark hole + spiders, realistic counts) ────────
    ng = min(6, nf)
    norm = PowerNorm(gamma=0.5, vmin=float(np.median(wfs_dark)), vmax=float(raw[:ng].max()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for k, ax in enumerate(axes.ravel()):
        if k < ng:
            im = ax.imshow(raw[k], origin="lower", cmap="inferno", norm=norm)
            ax.set_title(f"WFS frame {k}  (peak {raw[k].max():.0f} ADU)", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.set_visible(False)
    fig.suptitle("Simulated raw Shack-Hartmann frames (signal + dark + noise)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "wfs_frames_gallery.png"), dpi=120)
    plt.close(fig)
    print(f"[sim-g] figure -> {os.path.relpath(FIGURES_DIR, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
