"""
Effective transmission / system throughput for the WFS and imaging legs
(simulated-data pipeline).

Copied from NewDataPipeline/c_effectiveTransmission.py.  The optics physics
(which curves multiply in each leg) is UNCHANGED.  Simulation-specific changes:
  * output wavelength grid matches the object SED: 320-1000 nm @ 0.5 nm, so the
    throughput multiplies the injected SED element-wise;
  * optics config curves are read from this folder's local configFiles/;
  * the CoM (effective) wavelength of each leg is weighted by the REAL Arcturus
    SED (from b_initializeObject), not a blackbody;
  * BOTH legs' system throughput are saved (WFS frames are weighted by the WFS
    leg, imaging frames by the imaging leg);
  * diagnostic figures are written to figures/.

Run:  python c_effectiveTransmission.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib.spectrum import spectrl2
from astropy.io import fits

# ── Config ───────────────────────────────────────────────────────────────────
WL_START_NM = 320.0     # match the object SED grid
WL_STOP_NM  = 1000.0
WL_STEP_NM  = 0.5

SITE_NAME    = "Hard Labor Creek Observatory"
ELEVATION_M  = 219
ALTITUDE_DEG = 73.8
DAY_OF_YEAR  = 118
PRECIP_WATER = 2.0
OZONE        = 0.32
AOD_500      = 0.1

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "fitsOutputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
_CFG        = os.path.join(SCRIPT_DIR, "configFiles")   # optics curves copied here

# Ground-truth Arcturus SED (from b_initializeObject) — weights the CoM.
OBJECT_SED_FILE = os.path.join(
    OUTPUT_DIR,
    f"object_spectrum_arcturus_{WL_START_NM:.0f}-{WL_STOP_NM:.0f}nm_{WL_STEP_NM:g}nm.fits")


def _load_json_curve(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    keys = list(data.keys())
    return np.array(data[keys[0]]), np.array(data[keys[1]])


def load_object_sed(path):
    """Arcturus SED (flux + wavelength grid) saved by b_initializeObject.py."""
    with fits.open(path) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float64)
        h = hdul[0].header
        wl = h["CRVAL1"] + (np.arange(flux.size) + 1 - h["CRPIX1"]) * h["CDELT1"]
    return wl, flux


def compute_atmospheric_transmission(altitude_deg, precip_water, ozone,
                                     aod_500, day_of_year, surface_pressure):
    zenith_deg = 90.0 - altitude_deg
    airmass = pvlib.atmosphere.get_relative_airmass(zenith_deg, model='kastenyoung1989')
    result = spectrl2(
        apparent_zenith=zenith_deg, aoi=zenith_deg, surface_tilt=0,
        ground_albedo=0.2, surface_pressure=surface_pressure,
        relative_airmass=airmass, precipitable_water=precip_water, ozone=ozone,
        aerosol_turbidity_500nm=aod_500, scattering_albedo_400nm=0.9, alpha=1.14,
        wavelength_variation_factor=0.095, aerosol_asymmetry_factor=0.65,
        dayofyear=day_of_year,
    )
    wavelength = result['wavelength']
    dni_extra  = result['dni_extra'].flatten()
    dni        = result['dni'].flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = np.where(dni_extra > 0, dni / dni_extra, 0.0)
    mask = wavelength <= 1000
    return wavelength[mask], transmission[mask]


def compute_throughput(star_wl, star_flux, components):
    """Resample every component (and the star SED) onto the simulation grid
    (320-1000 nm @ 0.5 nm) and multiply.  Returns (grid, system throughput,
    star x system)."""
    common_wl = np.arange(WL_START_NM, WL_STOP_NM + WL_STEP_NM / 2, WL_STEP_NM)
    star_interp = np.interp(common_wl, star_wl, star_flux)
    sys_trans = np.ones_like(common_wl)
    for wl_c, val_c in components:
        sys_trans *= np.interp(common_wl, wl_c, val_c)
    eff_trans = star_interp * sys_trans
    return common_wl, sys_trans, eff_trans


def plot_leg_components(star_wl, star_flux, component_curves, eff_wl, eff_trans,
                        altitude_deg, leg_name, save_path):
    colors = ['orange', 'green', 'blue', 'purple', 'grey', 'sienna', 'cyan', 'magenta']
    fig, ax1 = plt.subplots(figsize=(12, 7.5))
    line_star, = ax1.plot(star_wl, star_flux / star_flux.max(), label="Arcturus SED",
                          linewidth=2, color='steelblue')
    ax1.set_xlabel("Wavelength (nm)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Normalized SED", fontsize=13, fontweight='bold')
    ax1.set_title(f"{leg_name} — Component Curves", fontsize=14, fontweight='bold')
    ax1.grid()

    ax2 = ax1.twinx()
    lines = [line_star]
    for i, (wl_c, val_c, lbl) in enumerate(component_curves):
        ln, = ax2.plot(wl_c, val_c * 100, linewidth=2, color=colors[i % len(colors)], label=lbl)
        lines.append(ln)
    ln_eff, = ax2.plot(eff_wl, eff_trans / eff_trans.max() * 100, linewidth=2.5,
                       color='black', linestyle='--', label="Effective (SED x T_sys)")
    lines.append(ln_eff)
    ax2.set_ylabel("Transmission / Reflectance / QE (%)", fontsize=13, fontweight='bold')
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=11)
    ax1.set_xlim(eff_wl[0], eff_wl[-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_effective_transmission(eff_wl, eff_trans, leg_name, save_path):
    com_wl  = np.sum(eff_wl * eff_trans) / np.sum(eff_trans)
    com_val = np.interp(com_wl, eff_wl, eff_trans) / eff_trans.max() * 100

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.plot(eff_wl, eff_trans / eff_trans.max() * 100, linewidth=2.5, color='black')
    ax.fill_between(eff_wl, eff_trans / eff_trans.max() * 100, alpha=0.15, color='yellow')
    ax.axvline(com_wl, color='black', linestyle='--', linewidth=1.5, zorder=4)
    ax.plot(com_wl, com_val, 'ko', markersize=8, zorder=5)
    ax.annotate(f"CoM: {com_wl:.1f} nm", xy=(com_wl, com_val),
                xytext=(com_wl + 40, com_val - 15), fontsize=13, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9))
    ax.set_xlabel("Wavelength (nm)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Effective Transmission (norm. %)", fontsize=13, fontweight='bold')
    ax.set_title(f"{leg_name} — Effective Transmission  (Arcturus SED x T_sys)",
                 fontsize=14, fontweight='bold')
    ax.set_xlim(eff_wl[0], eff_wl[-1]); ax.set_ylim(0, None); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return com_wl


def save_throughput_fits(eff_wl, sys_trans, eff_trans, leg, filename):
    com_wl = np.sum(eff_wl * eff_trans) / np.sum(eff_trans)
    col_wl = fits.Column(name='WAVELENGTH', format='D', unit='nm', array=eff_wl)
    col_et = fits.Column(name='EFF_TRANS',  format='D', array=sys_trans)  # system throughput (no star)
    hdu = fits.BinTableHDU.from_columns([col_wl, col_et])
    hdu.header['EXTNAME'] = 'SYSTEM_THROUGHPUT'
    hdu.header['LEG']     = (leg, 'Optical leg this throughput represents')
    hdu.header['COM_WL']  = (round(com_wl, 3), 'Arcturus-SED-weighted CoM wavelength [nm]')
    hdu.header['COMWGT']  = ('Arcturus SED', 'CoM weighting (not blackbody)')
    hdu.header['WLSTART'] = (WL_START_NM, 'First wavelength [nm]')
    hdu.header['WLSTOP']  = (WL_STOP_NM, 'Last wavelength [nm]')
    hdu.header['WLSTEP']  = (WL_STEP_NM, 'Wavelength step [nm]')
    hdu.header['SITE']    = (SITE_NAME, 'Observatory')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(os.path.join(OUTPUT_DIR, filename), overwrite=True)
    return com_wl


def main():
    surface_pressure = 101325 * np.exp(-ELEVATION_M / 8500)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    wl_a,    t_a    = _load_json_curve(f"{_CFG}/ThorLabs_Acoating_extrapolated.json")
    wl_bs_r, t_bs_r = _load_json_curve(f"{_CFG}/Newport_20B20BS1_Refl45deg_extrapolated.json")
    wl_bs_t, t_bs_t = _load_json_curve(f"{_CFG}/Newport_20B20BS1_Trans45deg_extrapolated.json")
    wl_p01,  t_p01  = _load_json_curve(f"{_CFG}/ThorLabs_P01_45deg_reduced.json")
    wl_g01,  t_g01  = _load_json_curve(f"{_CFG}/ThorLabs_G01_45deg_reduced.json")
    wl_er1,  t_er1  = _load_json_curve(f"{_CFG}/Newport_ER1_45deg_reduced.json")
    wl_mla,  t_mla  = _load_json_curve(f"{_CFG}/Thorlabs_MLA_FusedSilica_reduced.json")
    wl_qe,   t_qe   = _load_json_curve(f"{_CFG}/ZylaQE_extrapolated.json")
    wl_pri,  t_pri  = _load_json_curve(f"{_CFG}/Primary_Reflectance_extrapolated.json")
    wl_sec,  t_sec  = _load_json_curve(f"{_CFG}/Secondary_Reflectance_extrapolated.json")
    wl_ter,  t_ter  = _load_json_curve(f"{_CFG}/Tertiary_Reflectance_extrapolated.json")
    wl_cor,  t_cor  = _load_json_curve(f"{_CFG}/CorrectorPlate_Transmission_extrapolated.json")

    wl_atm, t_atm = compute_atmospheric_transmission(
        ALTITUDE_DEG, PRECIP_WATER, OZONE, AOD_500, DAY_OF_YEAR, surface_pressure)

    star_wl, star_flux = load_object_sed(OBJECT_SED_FILE)   # real Arcturus SED

    # ── WFS leg ──────────────────────────────────────────────────────────────
    wfs_components = [
        (wl_atm, t_atm), (wl_pri, t_pri), (wl_sec, t_sec), (wl_ter, t_ter), (wl_cor, t_cor),
        (wl_a, t_a), (wl_a, t_a), (wl_a, t_a), (wl_a, t_a),
        (wl_bs_r, t_bs_r), (wl_p01, t_p01), (wl_p01, t_p01), (wl_p01, t_p01),
        (wl_mla, t_mla), (wl_qe, t_qe),
    ]
    wfs_wl, wfs_sys, wfs_eff = compute_throughput(star_wl, star_flux, wfs_components)
    wfs_curves = [
        (wl_pri, t_pri, "Primary Refl (x1)"), (wl_sec, t_sec, "Secondary Refl (x1)"),
        (wl_ter, t_ter, "Tertiary Refl (x1)"), (wl_cor, t_cor, "Corrector T (x1)"),
        (wl_a, t_a, "A-Coating T (x4)"), (wl_bs_r, t_bs_r, "BS Reflectance (x1)"),
        (wl_p01, t_p01, "P01 Reflectance (x3)"), (wl_mla, t_mla, "MLA Transmission (x1)"),
        (wl_qe, t_qe, "Zyla QE (x1)"), (wl_atm, t_atm, f"Atm. Transmission (alt={ALTITUDE_DEG})"),
    ]
    plot_leg_components(star_wl, star_flux, wfs_curves, wfs_wl, wfs_eff, ALTITUDE_DEG,
                        "WFS Leg", os.path.join(FIGURES_DIR, "throughput_wfs_components.png"))
    com_wl_wfs = plot_effective_transmission(wfs_wl, wfs_eff, "WFS Leg",
                        os.path.join(FIGURES_DIR, "throughput_wfs_effective.png"))
    save_throughput_fits(wfs_wl, wfs_sys, wfs_eff, "WFS", "system_throughput_wfs.fits")
    print(f"[sim-c] WFS leg     CoM wavelength : {com_wl_wfs:.1f} nm  -> {os.path.join('fitsOutputs', 'system_throughput_wfs.fits')}")

    # ── Imaging leg ──────────────────────────────────────────────────────────
    img_components = [
        (wl_atm, t_atm), (wl_pri, t_pri), (wl_sec, t_sec), (wl_ter, t_ter), (wl_cor, t_cor),
        (wl_a, t_a), (wl_a, t_a), (wl_bs_t, t_bs_t), (wl_g01, t_g01), (wl_er1, t_er1), (wl_qe, t_qe),
    ]
    img_wl, img_sys, img_eff = compute_throughput(star_wl, star_flux, img_components)
    img_curves = [
        (wl_pri, t_pri, "Primary Refl (x1)"), (wl_sec, t_sec, "Secondary Refl (x1)"),
        (wl_ter, t_ter, "Tertiary Refl (x1)"), (wl_cor, t_cor, "Corrector T (x1)"),
        (wl_a, t_a, "A-Coating T (x2)"), (wl_bs_t, t_bs_t, "BS Transmission (x1)"),
        (wl_g01, t_g01, "G01 Reflectance (x1)"), (wl_er1, t_er1, "ER1 Reflectance (x1)"),
        (wl_qe, t_qe, "Zyla QE (x1)"), (wl_atm, t_atm, f"Atm. Transmission (alt={ALTITUDE_DEG})"),
    ]
    plot_leg_components(star_wl, star_flux, img_curves, img_wl, img_eff, ALTITUDE_DEG,
                        "Imaging Leg", os.path.join(FIGURES_DIR, "throughput_imaging_components.png"))
    com_wl_img = plot_effective_transmission(img_wl, img_eff, "Imaging Leg",
                        os.path.join(FIGURES_DIR, "throughput_imaging_effective.png"))
    save_throughput_fits(img_wl, img_sys, img_eff, "IMAGING", "system_throughput_imaging.fits")
    print(f"[sim-c] Imaging leg CoM wavelength : {com_wl_img:.1f} nm  -> {os.path.join('fitsOutputs', 'system_throughput_imaging.fits')}")
    print(f"[sim-c] figures -> {os.path.relpath(FIGURES_DIR, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
