"""
SLAP2 Microscope Health Dashboard
====================================
A Streamlit dashboard for monitoring the health and performance
of a custom two-photon microscope called SLAP2. 

The SLAP2 has two independent DMD-based illumination paths. 
Illumination uniformity along the short axis of the DMD is 
calculated using coeffient of variation (CV) of a flat-field 
image of a fluorescent slide. The long axis of the DMD is assumed 
to be uniform because of the line-scanning nature of the system.


Requirements:
    pip install streamlit numpy scipy scikit-image pandas plotly tifffile

Run:
    streamlit run dashboard.py

    If seeing error: 
    bash: streamlit: command not found
    Try:
    python3 -m streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration & Thresholds
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Threshold configuration for pass/fail decisions."""
    # PSF thresholds (in µm)
    psf_X_fwhm_max: float = 0.7   # typ. diffraction limit ~0.3–0.4 µm
    psf_Y_fwhm_max: float = 0.6   # typ. diffraction limit ~0.3–0.4 µm
    psf_Z_fwhm_max: float = 2.6      # typ. ~1.5–2.0 µm for good alignment
    psf_X_fwhm_warn: float = 0.65
    psf_Y_fwhm_warn: float = 0.55
    psf_Z_fwhm_warn: float = 2.3
    # Illumination uniformity
    uniformity_dmd1_max: float = 0.6       # coefficient of variation
    uniformity_dmd1_warn: float = 0.4
    uniformity_dmd2_max: float = 0.6       # coefficient of variation
    uniformity_dmd2_warn: float = 0.4
    # Laser power
    laser_power_min_mw: float = 30.0      # minimum at objective (mW)
    laser_power_warn_mw: float = 35.0
    # Pixel size for conversions
    pixel_size_x_um: float = 0.25         # µm/pixel X
    pixel_size_y_um: float = 0.25         # µm/pixel Y
    pixel_size_z_um: float = 0.5          # µm/pixel Z

DEFAULT_CONFIG = SystemConfig()

# ---------------------------------------------------------------------------
# Analysis Pipeline
# ---------------------------------------------------------------------------

def gaussian_1d(x, amp, mu, sigma, offset):
    """1D Gaussian for FWHM fitting."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

def fit_fwhm(profile):
    """Fit a 1D Gaussian to a line profile, return FWHM in pixels."""
    x = np.arange(len(profile))
    try:
        p0 = [profile.max() - profile.min(), np.argmax(profile),
              len(profile) / 6, profile.min()]
        popt, _ = curve_fit(gaussian_1d, x, profile, p0=p0, maxfev=5000)
        fwhm_px = 2.355 * abs(popt[2])  # sigma → FWHM
        return fwhm_px, popt
    except RuntimeError:
        return np.nan, None

def analyze_psf_image(img_stack, config: SystemConfig = DEFAULT_CONFIG):
    """
    Analyze a 3D bead image stack to extract PSF metrics.

    Parameters
    ----------
    img_stack : np.ndarray
        3D array (Z, Y, X) of a sub-resolution bead.
    config : SystemConfig
        System configuration with pixel sizes and thresholds.

    Returns
    -------
    dict with X_fwhm_um, Z_fwhm_um, fit profiles, status.
    """
    img = img_stack.astype(np.float64)

    # Find the brightest voxel as bead center
    z_max, y_max, x_max = np.unravel_index(np.argmax(img), img.shape)

    # Extract line profiles through the bead center
    X_x = img[z_max, y_max, :]
    X_y = img[z_max, :, x_max]
    Z_z   = img[:, y_max, x_max]

    # Fit FWHM
    fwhm_x_px, popt_x = fit_fwhm(X_x)
    fwhm_y_px, popt_y = fit_fwhm(X_y)
    fwhm_z_px, popt_z = fit_fwhm(Z_z)

    X_fwhm_um = np.nanmean([fwhm_x_px, fwhm_y_px]) * config.pixel_size_x_um
    Z_fwhm_um   = fwhm_z_px * config.pixel_size_z_um

    # Determine pass/warn/fail
    lat_status = _threshold_status(X_fwhm_um,
                                   config.psf_X_fwhm_warn,
                                   config.psf_X_fwhm_max)
    ax_status  = _threshold_status(Z_fwhm_um,
                                   config.psf_Z_fwhm_warn,
                                   config.psf_Z_fwhm_max)

    return {
        "X_fwhm_um": round(X_fwhm_um, 3),
        "Z_fwhm_um":   round(Z_fwhm_um, 3),
        "bead_center":     (int(z_max), int(y_max), int(x_max)),
        "profiles": {
            "X_x": X_x.tolist(),
            "X_y": X_y.tolist(),
            "Z_z":   Z_z.tolist(),
        },
        "fits": {
            "X_x": popt_x.tolist() if popt_x is not None else None,
            "X_y": popt_y.tolist() if popt_y is not None else None,
            "Z_z":   popt_z.tolist() if popt_z is not None else None,
        },
        "X_status": lat_status,
        "Z_status":   ax_status,
    }

def analyze_psf_multi_bead(img_stack, config: SystemConfig = DEFAULT_CONFIG,
                           min_distance=10, threshold_rel=0.3):
    """
    Detect multiple beads in a 3D stack, fit each, return summary stats.

    Extracts a small ROI around each detected peak and fits independently.
    """
    img = img_stack.astype(np.float64)
    mip = img.max(axis=0)  # max-intensity projection for detection

    coords = peak_local_max(mip, min_distance=min_distance,
                            threshold_rel=threshold_rel)
    results = []
    roi_half = min_distance

    for (y, x) in coords:
        # Find z-plane of max intensity at this (y, x)
        z = int(np.argmax(img[:, y, x]))
        # Extract sub-volume
        z0, z1 = max(z - roi_half, 0), min(z + roi_half + 1, img.shape[0])
        y0, y1 = max(y - roi_half, 0), min(y + roi_half + 1, img.shape[1])
        x0, x1 = max(x - roi_half, 0), min(x + roi_half + 1, img.shape[2])
        sub = img[z0:z1, y0:y1, x0:x1]
        if sub.size == 0:
            continue
        r = analyze_psf_image(sub, config)
        r["bead_x"] = (int(x), int(y))
        results.append(r)

    if not results:
        return {"n_beads": 0, "beads": [], "summary": {}}

    lat = [r["X_fwhm_um"] for r in results if not np.isnan(r["X_fwhm_um"])]
    axi = [r["Z_fwhm_um"]   for r in results if not np.isnan(r["Z_fwhm_um"])]

    return {
        "n_beads": len(results),
        "beads": results,
        "summary": {
            "X_fwhm_mean": round(np.mean(lat), 3) if lat else np.nan,
            "X_fwhm_std":  round(np.std(lat), 3)  if lat else np.nan,
            "Z_fwhm_mean":   round(np.mean(axi), 3) if axi else np.nan,
            "Z_fwhm_std":    round(np.std(axi), 3)  if axi else np.nan,
        }
    }

def analyze_uniformity(img_2d, config: SystemConfig = DEFAULT_CONFIG,
                       smooth_sigma=5, dmd_label="dmd1"):
    """
    Analyze illumination uniformity from a flat-field image.

    Parameters
    ----------
    img_2d : np.ndarray
        2D image of a uniform fluorescent sample.
    config : SystemConfig
    smooth_sigma : float
        Gaussian smoothing to reduce noise before analysis.

    Returns
    -------
    dict with CV, heatmap, status.
    """
    img = gaussian_filter(img_2d.astype(np.float64), sigma=smooth_sigma)
    img_norm = img / img.max()

    # Coefficient of variation
    cv = float(np.std(img_norm) / np.mean(img_norm))

    # Downsample heatmap for display
    h, w = img_norm.shape
    ds = max(1, min(h, w) // 64)
    heatmap = img_norm[::ds, ::ds]

    if str(dmd_label).lower() == "dmd2":
        cv_warn = config.uniformity_dmd2_warn
        cv_fail = config.uniformity_dmd2_max
    else:
        cv_warn = config.uniformity_dmd1_warn
        cv_fail = config.uniformity_dmd1_max

    cv_status = _threshold_status(cv, cv_warn, cv_fail)

    return {
        "cv": round(cv, 4),
        "heatmap": heatmap.tolist(),
        "cv_status": cv_status,
    }

def analyze_laser_power(power_at_objective_mw,
                        wavelength_nm=1030, config: SystemConfig = DEFAULT_CONFIG):
    """
    Evaluate laser power metrics.

    Parameters
    ----------
    power_at_objective_mw : float
    wavelength_nm : int
    config : SystemConfig

    Returns
    -------
    dict with power value and status.
    """
    pwr_status = _threshold_status_lower(power_at_objective_mw,
                                         config.laser_power_warn_mw,
                                         config.laser_power_min_mw)

    return {
        "power_at_objective_mw": round(power_at_objective_mw, 2),
        "wavelength_nm": wavelength_nm,
        "power_status": pwr_status,
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _threshold_status(val, warn_thresh, fail_thresh):
    """For metrics where HIGHER is worse (e.g., FWHM, CV)."""
    if np.isnan(val):
        return "ERROR"
    if val > fail_thresh:
        return "FAIL"
    if val > warn_thresh:
        return "WARN"
    return "PASS"

def _threshold_status_lower(val, warn_thresh, fail_thresh):
    """For metrics where LOWER is worse (e.g., laser power)."""
    if np.isnan(val):
        return "ERROR"
    if val < fail_thresh:
        return "FAIL"
    if val < warn_thresh:
        return "WARN"
    return "PASS"

STATUS_COLORS = {"PASS": "#2ecc71", "WARN": "#f39c12", "FAIL": "#e74c3c", "ERROR": "#95a5a6"}
STATUS_ICONS  = {"PASS": "✅", "WARN": "⚠️", "FAIL": "🔴", "ERROR": "❓"}

def status_badge(status):
    return f"{STATUS_ICONS.get(status, '?')} **{status}**"

# ---------------------------------------------------------------------------
# Synthetic Data Generator (for demo / testing)
# ---------------------------------------------------------------------------

def generate_demo_bead_stack(nz=40, ny=64, nx=64, sigma_x=0.8, sigma_z=2.0,
                             snr=20, n_beads=1):
    """Generate a synthetic 3D bead image stack."""
    stack = np.random.poisson(lam=5, size=(nz, ny, nx)).astype(np.float64)
    for _ in range(n_beads):
        cz = np.random.randint(nz // 4, 3 * nz // 4)
        cy = np.random.randint(ny // 4, 3 * ny // 4)
        cx = np.random.randint(nx // 4, 3 * nx // 4)
        zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
        bead = snr * 100 * np.exp(-(
            (xx - cx)**2 / (2 * sigma_x**2) +
            (yy - cy)**2 / (2 * sigma_x**2) +
            (zz - cz)**2 / (2 * sigma_z**2)
        ))
        stack += bead
    return stack

def generate_demo_uniformity(ny=256, nx=256, vignette_strength=0.15):
    """Generate a synthetic flat-field image with vignetting."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2, nx / 2
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2) / np.sqrt(cy**2 + cx**2)
    img = 1000 * (1 - vignette_strength * r**2)
    img += np.random.normal(0, 15, img.shape)
    return np.clip(img, 0, None)

def generate_demo_history(n_sessions=30):
    """Generate synthetic QC history data."""
    dates = [datetime.now() - timedelta(days=n_sessions - i) for i in range(n_sessions)]
    np.random.seed(42)

    # Simulate gradual degradation + noise
    drift = np.linspace(0, 0.06, n_sessions)
    records = []
    for i, dt in enumerate(dates):
        # Inject a "maintenance event" at session 15
        reset = 0.04 if i >= 15 else 0.0
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "X_fwhm_um": round(0.58 + drift[i] - reset + np.random.normal(0, 0.01), 3),
            "Y_fwhm_um": round(0.48 + drift[i] - reset + np.random.normal(0, 0.01), 3),
            "Z_fwhm_um":   round(1.9 + 2 * drift[i] - 1.5 * reset + np.random.normal(0, 0.05), 3),
            "uniformity_dmd1":   round(0.28 + 0.9 * drift[i] - 0.6 * reset + np.random.normal(0, 0.005), 4),
            "uniformity_dmd2":   round(0.29 + 0.95 * drift[i] - 0.62 * reset + np.random.normal(0, 0.006), 4),
            "power_at_objective_mw": round(43 - 60 * drift[i] + 40 * reset + np.random.normal(0, 2), 1),
            "notes": "Realigned mirrors + cleaned optics" if i == 15 else "",
        })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Streamlit Dashboard
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="SLAP2 Microscope Health Dashboard",
                       page_icon="🔬", layout="wide")

    st.title("🔬 SLAP2 Microscope Health Dashboard")
    st.caption("Systematic monitoring of resolution, illumination uniformity, and laser power")

    # Sidebar: configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        config = SystemConfig()
        config.pixel_size_x_um = st.number_input("Pixel size X (µm)", value=0.25, step=0.01, format="%.3f")
        config.pixel_size_y_um = st.number_input("Pixel size Y (µm)", value=0.25, step=0.01, format="%.3f")
        config.pixel_size_z_um  = st.number_input("Pixel size Z (µm)", value=0.5, step=0.1, format="%.2f")

        st.subheader("PSF Thresholds")
        config.psf_X_fwhm_max = st.number_input("X FWHM fail (µm)", value=0.7, step=0.01)
        config.psf_X_fwhm_warn = st.number_input("X FWHM warn (µm)", value=0.65, step=0.01)
        config.psf_Y_fwhm_max = st.number_input("Y FWHM fail (µm)", value=0.6, step=0.01)
        config.psf_Y_fwhm_warn = st.number_input("Y FWHM warn (µm)", value=0.55, step=0.01)
        config.psf_Z_fwhm_max = st.number_input("Z FWHM fail (µm)", value=2.6, step=0.1)
        config.psf_Z_fwhm_warn = st.number_input("Z FWHM warn (µm)", value=2.3, step=0.1)

        st.subheader("Uniformity Thresholds")
        config.uniformity_dmd1_max = st.number_input("DMD1 CV fail", value=0.6, step=0.01)
        config.uniformity_dmd1_warn = st.number_input("DMD1 CV warn", value=0.4, step=0.01)
        config.uniformity_dmd2_max = st.number_input("DMD2 CV fail", value=0.6, step=0.01)
        config.uniformity_dmd2_warn = st.number_input("DMD2 CV warn", value=0.4, step=0.01)

        st.subheader("Laser Thresholds")
        config.laser_power_min_mw = st.number_input("Power fail (mW)", value=30.0, step=5.0)
        config.laser_power_warn_mw = st.number_input("Power warn (mW)", value=35.0, step=5.0)

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab_live, tab_trends, tab_upload = st.tabs([
        "📊 Live Demo", "📈 Trend Monitoring", "📁 Upload & Analyze"
    ])

    # ===========================
    # TAB 1: Live Analysis (Demo)
    # ===========================
    with tab_live:
        st.subheader("Run QC with demo data")
        st.info("Click below to run analysis on synthetic calibration data. "
                "Use the **Upload & Analyze** tab for real images.")

        if st.button("🔄 Run Demo QC Session", type="primary"):
            with st.spinner("Generating synthetic data and analyzing..."):
                _run_demo_analysis(config)

    # ===========================
    # TAB 2: Trend Monitoring
    # ===========================
    with tab_trends:
        st.subheader("Historical QC Trends")

        # Load or generate history
        history_file = "\\\\allen\\aind\\scratch\\ophys\\SLAP2\\weekly alignment\\history.csv"  #this is the slap2 alignment history file
        # history_file = ""  #use this if you want synthetic data
        if os.path.exists(history_file):
            history = pd.read_csv(history_file)
        else:
            history = generate_demo_history()

        # --- Summary status cards ---
        latest = history.iloc[-1]
        cols = st.columns(6)
        with cols[0]:
            s = _threshold_status(latest["X_fwhm_um"],
                                  config.psf_X_fwhm_warn, config.psf_X_fwhm_max)
            st.metric("X FWHM", f"{latest['X_fwhm_um']} µm",
                      delta=f"{latest['X_fwhm_um'] - history.iloc[-2]['X_fwhm_um']:+.3f}")
            st.markdown(status_badge(s))
        with cols[1]:
            s = _threshold_status(latest["Y_fwhm_um"],
                                  config.psf_Y_fwhm_warn, config.psf_Y_fwhm_max)
            st.metric("Y FWHM", f"{latest['Y_fwhm_um']} µm",
                      delta=f"{latest['Y_fwhm_um'] - history.iloc[-2]['Y_fwhm_um']:+.3f}")
            st.markdown(status_badge(s))
        with cols[2]:
            s = _threshold_status(latest["Z_fwhm_um"],
                                  config.psf_Z_fwhm_warn, config.psf_Z_fwhm_max)
            st.metric("Z FWHM", f"{latest['Z_fwhm_um']} µm",
                      delta=f"{latest['Z_fwhm_um'] - history.iloc[-2]['Z_fwhm_um']:+.3f}")
            st.markdown(status_badge(s))
        with cols[3]:
            s = _threshold_status(latest["uniformity_dmd1"],
                                  config.uniformity_dmd1_warn, config.uniformity_dmd1_max)
            st.metric("Uniformity DMD1 (CV)", f"{latest['uniformity_dmd1']:.4f}",
                      delta=f"{latest['uniformity_dmd1'] - history.iloc[-2]['uniformity_dmd1']:+.4f}")
            st.markdown(status_badge(s))
        with cols[4]:
            s = _threshold_status(latest["uniformity_dmd2"],
                                  config.uniformity_dmd2_warn, config.uniformity_dmd2_max)
            st.metric("Uniformity DMD2 (CV)", f"{latest['uniformity_dmd2']:.4f}",
                      delta=f"{latest['uniformity_dmd2'] - history.iloc[-2]['uniformity_dmd2']:+.4f}")
            st.markdown(status_badge(s))
        with cols[5]:
            s = _threshold_status_lower(latest["power_at_objective_mw"],
                                        config.laser_power_warn_mw, config.laser_power_min_mw)
            st.metric("Laser Power", f"{latest['power_at_objective_mw']} mW",
                      delta=f"{latest['power_at_objective_mw'] - history.iloc[-2]['power_at_objective_mw']:+.1f}")
            st.markdown(status_badge(s))

        st.divider()

        # --- Trend charts ---
        _plot_trend(history, "X_fwhm_um", "X FWHM (µm)",
                    config.psf_X_fwhm_warn, config.psf_X_fwhm_max)
        
        _plot_trend(history, "Y_fwhm_um", "Y FWHM (µm)",
                    config.psf_Y_fwhm_warn, config.psf_Y_fwhm_max)
        
        _plot_trend(history, "Z_fwhm_um", "Z FWHM (µm)",
                    config.psf_Z_fwhm_warn, config.psf_Z_fwhm_max)

        col1, col2 = st.columns(2)
        with col1:
            _plot_trend(history, "uniformity_dmd1", "Uniformity DMD1 (CV)",
                        config.uniformity_dmd1_warn, config.uniformity_dmd1_max)
        with col2:
            _plot_trend(history, "uniformity_dmd2", "Uniformity DMD2 (CV)",
                        config.uniformity_dmd2_warn, config.uniformity_dmd2_max)
            
        _plot_trend(history, "power_at_objective_mw", "Power at Objective (mW)",
                    config.laser_power_warn_mw, config.laser_power_min_mw,
                    lower_is_worse=True)

        # Maintenance log
        maint = history[(history["notes"].notna()) & (history["notes"] != "") & (history["date"].notna())]
        if not maint.empty:
            st.subheader("🔧 Maintenance Events")
            st.dataframe(maint[["date", "notes"]], use_container_width=True, hide_index=True)

        # Raw data
        with st.expander("📋 View raw QC data table"):
            st.dataframe(history, use_container_width=True, hide_index=True)

    # ===========================
    # TAB 3: Upload & Analyze
    # ===========================
    with tab_upload:
        st.subheader("Upload calibration images for analysis")

        upload_type = st.radio("Data type", ["PSF / Bead Stack", "Uniformity Image",
                                             "Laser Power (manual entry)"],
                               horizontal=True)

        if upload_type == "PSF / Bead Stack":
            uploaded = st.file_uploader("Upload bead z-stack (TIFF)", type=["tif", "tiff"])
            multi = st.checkbox("Multi-bead detection", value=False)

            if uploaded is not None:
                try:
                    import tifffile
                    stack = tifffile.imread(uploaded)
                    st.write(f"Loaded stack: shape = {stack.shape}, dtype = {stack.dtype}")

                    if stack.ndim == 3:
                        with st.spinner("Analyzing PSF..."):
                            if multi:
                                result = analyze_psf_multi_bead(stack, config)
                                st.json(result["summary"])
                                for i, b in enumerate(result["beads"]):
                                    st.write(f"**Bead {i+1}** at {b['bead_x']}")
                                    _display_psf_result(b)
                            else:
                                result = analyze_psf_image(stack, config)
                                _display_psf_result(result)
                    else:
                        st.error("Expected a 3D stack (Z, Y, X). Got shape: " + str(stack.shape))
                except ImportError:
                    st.error("Install `tifffile`: pip install tifffile")

        elif upload_type == "Uniformity Image":
            uploaded = st.file_uploader("Upload flat-field image (TIFF/PNG)", type=["tif", "tiff", "png"])
            uniformity_target = st.radio("Uniformity channel", ["DMD1", "DMD2"], horizontal=True)
            if uploaded is not None:
                try:
                    import tifffile
                    img = tifffile.imread(uploaded)
                    if img.ndim == 3:
                        img = img.max(axis=0)  # MIP if 3D
                    with st.spinner("Analyzing uniformity..."):
                        dmd_label = "dmd2" if uniformity_target == "DMD2" else "dmd1"
                        result = analyze_uniformity(img, config, dmd_label=dmd_label)
                        _display_uniformity_result(result, label=uniformity_target)
                except ImportError:
                    st.error("Install `tifffile`: pip install tifffile")

        elif upload_type == "Laser Power (manual entry)":
            col1, col2 = st.columns(2)
            with col1:
                pwr_obj = st.number_input("Power at objective (mW)", value=40.0, step=1.0)
            with col2:
                wl = st.number_input("Wavelength (nm)", value=1030, step=10)

            if st.button("Analyze Laser Power"):
                result = analyze_laser_power(pwr_obj, wl, config)
                _display_laser_result(result)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_psf_result(result):
    """Display PSF analysis results."""
    c1, c2 = st.columns(2)
    with c1:
        st.metric("X FWHM", f"{result['X_fwhm_um']} µm")
        st.markdown(status_badge(result["X_status"]))
    with c2:
        st.metric("Z FWHM", f"{result['Z_fwhm_um']} µm")
        st.markdown(status_badge(result["Z_status"]))

    # Plot profiles with fits
    fig = go.Figure()
    for label, key in [("X X", "X_x"), ("X Y", "X_y"), ("Z Z", "Z_z")]:
        prof = result["profiles"][key]
        fig.add_trace(go.Scatter(y=prof, name=f"{label} (data)", mode="lines"))

        fit_params = result["fits"][key]
        if fit_params is not None:
            x = np.arange(len(prof))
            fit_curve = gaussian_1d(x, *fit_params)
            fig.add_trace(go.Scatter(y=fit_curve, name=f"{label} (fit)",
                                     mode="lines", line=dict(dash="dash")))

    fig.update_layout(title="PSF Line Profiles",
                      xaxis_title="Pixel", yaxis_title="Intensity",
                      height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def _display_uniformity_result(result, label="DMD1"):
    """Display uniformity analysis results."""
    st.metric(f"{label} Coefficient of Variation", f"{result['cv']:.4f}")
    st.markdown(status_badge(result["cv_status"]))

    heatmap = np.array(result["heatmap"])
    fig = px.imshow(heatmap, color_continuous_scale="viridis",
                    title="Normalized Intensity Map", aspect="equal")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def _display_laser_result(result):
    """Display laser power results."""
    st.metric("Power at Objective", f"{result['power_at_objective_mw']} mW")
    st.markdown(status_badge(result["power_status"]))
    st.caption(f"Wavelength: {result['wavelength_nm']} nm")

def _run_demo_analysis(config):
    """Run full demo analysis suite."""
    st.markdown("---")

    # --- PSF ---
    st.subheader("🔍 PSF / Resolution")
    stack = generate_demo_bead_stack(nz=40, ny=64, nx=64, sigma_x=0.8, sigma_z=2.0)
    result_psf = analyze_psf_image(stack, config)
    _display_psf_result(result_psf)

    st.markdown("---")

    # --- Uniformity ---
    st.subheader("💡 Illumination Uniformity")
    uni_col1, uni_col2 = st.columns(2)
    with uni_col1:
        flat_dmd1 = generate_demo_uniformity(vignette_strength=0.15)
        result_uni_dmd1 = analyze_uniformity(flat_dmd1, config, dmd_label="dmd1")
        _display_uniformity_result(result_uni_dmd1, label="DMD1")
    with uni_col2:
        flat_dmd2 = generate_demo_uniformity(vignette_strength=0.2)
        result_uni_dmd2 = analyze_uniformity(flat_dmd2, config, dmd_label="dmd2")
        _display_uniformity_result(result_uni_dmd2, label="DMD2")

    st.markdown("---")

    # --- Laser ---
    st.subheader("⚡ Laser Power")
    result_lsr = analyze_laser_power(
        power_at_objective_mw=42.5 + np.random.normal(0, 2),
        wavelength_nm=1030,
        config=config,
    )
    _display_laser_result(result_lsr)

def _plot_trend(df, col, label, warn_thresh, fail_thresh, lower_is_worse=False):
    """Plot a time-series trend with threshold bands."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[col], mode="lines+markers",
        name=label, line=dict(color="#3498db", width=2),
        marker=dict(size=5),
    ))

    y_min = df[col].min() * 0.9
    y_max = df[col].max() * 1.1

    if lower_is_worse:
        fig.add_hrect(y0=y_min, y1=fail_thresh, fillcolor="#e74c3c",
                      opacity=0.08, line_width=0, annotation_text="FAIL zone")
        fig.add_hrect(y0=fail_thresh, y1=warn_thresh, fillcolor="#f39c12",
                      opacity=0.08, line_width=0)
        fig.add_hline(y=fail_thresh, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Fail")
        fig.add_hline(y=warn_thresh, line_dash="dash", line_color="#f39c12",
                      annotation_text="Warn")
    else:
        fig.add_hrect(y0=fail_thresh, y1=y_max, fillcolor="#e74c3c",
                      opacity=0.08, line_width=0, annotation_text="FAIL zone")
        fig.add_hrect(y0=warn_thresh, y1=fail_thresh, fillcolor="#f39c12",
                      opacity=0.08, line_width=0)
        fig.add_hline(y=fail_thresh, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Fail")
        fig.add_hline(y=warn_thresh, line_dash="dash", line_color="#f39c12",
                      annotation_text="Warn")

    # Mark maintenance events
    maint = df[(df["notes"].notna()) & (df["notes"] != "") & (df["date"].notna())]
    if not maint.empty:
        fig.add_trace(go.Scatter(
            x=maint["date"], y=maint[col], mode="markers",
            marker=dict(symbol="star", size=12, color="#9b59b6"),
            name="Maintenance", hovertext=maint["notes"],
        ))

    fig.update_layout(title=label, xaxis_title="Date", yaxis_title=label,
                      height=300, template="plotly_white",
                      showlegend=True, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
