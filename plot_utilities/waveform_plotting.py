"""SXS/KAN/NRSur7dq4/IMRPhenomD waveform comparison.

The required inputs are an SXS ID, detector-frame total mass in solar masses,
and luminosity distance in Mpc.  SXS and NRSur7dq4 receive the same frozen
training conditioning: dt/M=0.5, a 600 M Planck turn-on, and the adaptive
ringdown-safe turn-off.  

Transform policy
----------------
* SXS and NRSur7dq4 originate in time and receive one time-to-frequency FFT.
* KAN and IMRPhenomD originate in frequency and receive one inverse FFT for
  the time-domain display.
* No waveform follows an FD -> TD -> FD round trip.

The first panel shows Fourier amplitude on log-log axes over the common
trustworthy band, with
the Advanced-LIGO zero-detuned high-power design and O4-intermediate amplitude
sensitivities on a separate right-hand ordinate because sqrt(S_n) and h~ have
different physical units.  The second panel shows unwrapped Fourier phase.
The third panel starts 400 M before the SXS amplitude peak and continues
through the common conditioned merger-ringdown record, with time expressed in
seconds.  All comparison waveforms use the time/phase-maximizing alignment
measured against SXS on the common frequency band.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from KAN_utilities import KAN_Waveform as KW
from . import _mismatch_support as support
from . import consistent_conditioning_mismatch as consistent


MODEL_NAMES = ("KAN", "SXS", "NRSur7dq4", "IMRPhenomD")
MODEL_PLOT_STYLES = {
    "KAN": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 2,
    },
    "NRSur7dq4": {
        "color": "#0072B2",
        "linestyle": (0, (7.0, 2.0, 1.5, 2.0)),
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 3,
    },
    "IMRPhenomD": {
        "color": "#D55E00",
        "linestyle": (0, (1.0, 1.8)),
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    "SXS": {
        "color": "#B3B3B3",
        "linestyle": "-",
        "linewidth": 5.0,
        "alpha": 0.6,
        "zorder": 1,
    },
}
NRSUR_START_TAPER_M = 600.0
DISPLAY_LOW_TAPER_END_HZ = 20.0
DISPLAY_HIGH_TAPER_FRACTION = 0.05
SENSITIVITY_PLOT_MIN_HZ = 20.0
SENSITIVITY_PLOT_MAX_HZ = 1000.0
DESIGN_PSD_NAME = "aLIGOZeroDetHighPower"
O4_INTERMEDIATE_PSD_NAME = "aLIGOAdVO4IntermediateT1800545"


def _positive_number(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _pycbc_psd_on_grid(
    psd_function,
    frequency_hz: np.ndarray,
    delta_f_hz: float,
) -> np.ndarray:
    """Evaluate a PyCBC PSD function on grid-aligned frequencies."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    indices = np.rint(frequency_hz / delta_f_hz).astype(int)
    if np.any(indices < 0) or not np.allclose(
        frequency_hz,
        indices * delta_f_hz,
        rtol=1.0e-9,
        atol=max(1.0e-12, 1.0e-9 * delta_f_hz),
    ):
        raise ValueError("PSD frequencies are not aligned to delta_f")
    full_psd = psd_function(
        int(indices[-1]) + 2,
        float(delta_f_hz),
        SENSITIVITY_PLOT_MIN_HZ,
    )
    return np.asarray(full_psd, dtype=float)[indices]


def _condition_nrsur(
    parameters: dict[str, float],
    total_mass_msun: float,
    luminosity_distance_mpc: float,
    delta_t_seconds: float,
) -> dict[str, Any]:
    """Generate NRSur7dq4 and apply the frozen 600 M training window."""

    consistent._validate_nrsur_calibration(parameters)
    try:
        from pycbc.waveform import get_td_waveform
    except ImportError as exc:
        raise ImportError(
            "PyCBC is required; use the environment containing the project "
            "waveform dependencies."
        ) from exc

    mass1 = total_mass_msun * parameters["mass1_fraction"]
    mass2 = total_mass_msun * parameters["mass2_fraction"]
    hp, hc = get_td_waveform(
        approximant="NRSur7dq4",
        mass1=mass1,
        mass2=mass2,
        spin1z=parameters["chi1z"],
        spin2z=parameters["chi2z"],
        distance=luminosity_distance_mpc,
        inclination=0.0,
        coa_phase=0.0,
        delta_t=delta_t_seconds,
        f_lower=0.0,
        mode_array=[(2, 2), (2, -2)],
    )
    strain = np.asarray(hp, dtype=float) - 1j * np.asarray(hc, dtype=float)
    tau = support.TRAINING_DTAU * np.arange(len(strain), dtype=float)
    peak_index = int(np.argmax(np.abs(strain)))
    peak_tau = float(tau[peak_index])
    damping_time = support._ringdown_damping_time(parameters)
    window, window_information = consistent._adaptive_window(
        tau,
        strain,
        peak_tau,
        damping_time,
        NRSUR_START_TAPER_M,
    )
    if not window_information["ok"]:
        raise ValueError(
            "The adaptive NRSur7dq4 turn-off is not admissible: "
            f"{window_information['reason']}"
        )

    mf_start = consistent._measured_start_mf(tau, strain)
    mf_ringdown = float(
        KW.mf_ringdown(
            parameters["mass1_fraction"],
            parameters["mass2_fraction"],
            parameters["chi1z"],
            parameters["chi2z"],
        )
    )
    return {
        "tau": tau,
        "strain": strain * window,
        "mf_start": mf_start,
        "mf_min": support.TRAINING_K_LO * mf_start,
        "mf_max": support.TRAINING_K_HI * mf_ringdown,
        "mf_ringdown": mf_ringdown,
        "window": window_information,
    }


def _generate_imrphenomd(
    parameters: dict[str, float],
    total_mass_msun: float,
    luminosity_distance_mpc: float,
    delta_f_hz: float,
    f_lower_hz: float,
    f_final_hz: float,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Evaluate IMRPhenomD directly on a frequency grid."""

    try:
        from pycbc.waveform import get_fd_waveform
    except ImportError as exc:
        raise ImportError(
            "PyCBC is required; use the environment containing the project "
            "waveform dependencies."
        ) from exc

    hp, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=total_mass_msun * parameters["mass1_fraction"],
        mass2=total_mass_msun * parameters["mass2_fraction"],
        spin1z=parameters["chi1z"],
        spin2z=parameters["chi2z"],
        distance=luminosity_distance_mpc,
        inclination=0.0,
        coa_phase=0.0,
        delta_f=delta_f_hz,
        f_lower=f_lower_hz,
        f_final=f_final_hz,
    )
    source_frequency = np.asarray(hp.sample_frequencies, dtype=float)
    source_spectrum = np.asarray(hp, dtype=np.complex128)
    output = np.full(
        len(frequency_hz),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    # ``frequency_hz`` starts at delta_f rather than at zero.
    indices = np.rint(source_frequency / delta_f_hz).astype(int) - 1
    usable = (
        (indices >= 0)
        & (indices < len(output))
        & np.isclose(
            source_frequency,
            (indices + 1) * delta_f_hz,
            rtol=1.0e-9,
            atol=max(1.0e-12, 1.0e-9 * delta_f_hz),
        )
    )
    output[indices[usable]] = source_spectrum[usable]
    return output


def _display_band_taper(
    frequency_hz: np.ndarray,
    f_lower_hz: float,
    f_upper_hz: float,
) -> np.ndarray:
    """Reconstruction taper with unit weight from 20 Hz to the high edge."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    taper = np.zeros_like(frequency_hz)
    inside = (frequency_hz >= f_lower_hz) & (frequency_hz <= f_upper_hz)
    taper[inside] = 1.0

    if f_lower_hz < DISPLAY_LOW_TAPER_END_HZ < f_upper_hz:
        low = inside & (frequency_hz < DISPLAY_LOW_TAPER_END_HZ)
        x = (
            (frequency_hz[low] - f_lower_hz)
            / (DISPLAY_LOW_TAPER_END_HZ - f_lower_hz)
        )
        taper[low] = 0.5 * (1.0 - np.cos(np.pi * x))

    high_width = DISPLAY_HIGH_TAPER_FRACTION * (
        f_upper_hz - f_lower_hz
    )
    if high_width > 0.0:
        high = inside & (frequency_hz > f_upper_hz - high_width)
        x = (f_upper_hz - frequency_hz[high]) / high_width
        taper[high] = 0.5 * (1.0 - np.cos(np.pi * x))
    return taper


def _inverse_positive_spectrum(
    positive_spectrum: np.ndarray,
    negative_indices: np.ndarray,
    delta_t_seconds: float,
    taper: np.ndarray,
) -> np.ndarray:
    """Make one FD-to-TD transform using the project's h22 convention."""

    values = np.asarray(positive_spectrum, dtype=np.complex128)
    weights = np.asarray(taper, dtype=float)
    if not (len(values) == len(negative_indices) == len(weights)):
        raise ValueError("Spectrum, frequency indices, and taper must align")
    usable = np.isfinite(values) & (weights > 0.0)
    fft_values = np.zeros(2 * len(negative_indices), dtype=np.complex128)
    fft_values[negative_indices[usable]] = np.conj(
        2.0 * values[usable] * weights[usable]
    )
    return np.fft.ifft(fft_values) / delta_t_seconds


def _frequency_align_to_sxs(
    spectra: dict[str, np.ndarray],
    frequency_mf: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Remove arbitrary time and phase offsets in the frequency panel."""

    reference = np.asarray(spectra["SXS"], dtype=np.complex128)
    aligned = {"SXS": reference.copy()}
    alignment: dict[str, dict[str, float]] = {
        "SXS": {"time_shift_M": 0.0, "phase_shift_radians": 0.0}
    }
    for name in ("KAN", "IMRPhenomD", "NRSur7dq4"):
        candidate = np.asarray(spectra[name], dtype=np.complex128)
        result = support._maximize_time_phase_match(
            reference,
            candidate,
            frequency_mf,
            preferred_time_shift_M=None,
        )
        aligned[name] = candidate * np.exp(
            -2j * np.pi * frequency_mf * result["time_shift_M"]
            + 1j * result["phase_shift_radians"]
        )
        alignment[name] = {
            "time_shift_M": float(result["time_shift_M"]),
            "phase_shift_radians": float(result["phase_shift_radians"]),
        }
    return aligned, alignment


def prepare_waveforms(
    sxs_id: str,
    total_mass_msun: float,
    luminosity_distance_mpc: float,
    *,
    download: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Generate and condition the four waveforms without plotting them."""

    total_mass_msun = _positive_number(total_mass_msun, "total_mass_msun")
    luminosity_distance_mpc = _positive_number(
        luminosity_distance_mpc,
        "luminosity_distance_mpc",
    )

    data_directory = Path(support.DEFAULT_DATA_DIRECTORY).resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    support._ensure_nrsur_data(data_directory, download=download)
    simulation = support._load_sxs(
        sxs_id,
        data_directory,
        download=download,
        progress=progress,
    )
    parameters = support._binary_parameters(simulation.metadata)
    consistent._validate_nrsur_calibration(parameters)
    raw_sxs = support._extract_sxs_mode(simulation)

    mass_seconds = total_mass_msun * support.MTSUN_SI
    distance_seconds = (
        luminosity_distance_mpc * support.MPC_SI / support.C_SI
    )
    delta_t_seconds = support.TRAINING_DTAU * mass_seconds
    sxs_scale = (mass_seconds / distance_seconds) * support.Y22_FACEON
    sxs = consistent._condition_sxs_reference(
        raw_sxs,
        parameters,
        sxs_scale,
    )
    nrsur = _condition_nrsur(
        parameters,
        total_mass_msun,
        luminosity_distance_mpc,
        delta_t_seconds,
    )

    domain = KW.model_domain()
    common_mf_min = max(
        sxs["mf_min"],
        nrsur["mf_min"],
        float(domain["Mf"][0]),
    )
    common_mf_max = min(
        sxs["mf_max"],
        nrsur["mf_max"],
        float(domain["Mf"][1]),
        np.nextafter(0.5 / support.TRAINING_DTAU, 0.0),
    )
    if common_mf_max <= common_mf_min:
        raise ValueError(
            f"Empty common trustworthy band Mf={common_mf_min:.7g}--"
            f"{common_mf_max:.7g}"
        )

    # KAN/IMRPhenomD use a wider valid band for the time-domain display.  This
    # does not change the frequency panel or introduce a round trip.
    display_mf_min = float(domain["Mf"][0])
    display_mf_max = min(
        float(domain["Mf"][1]),
        support.TRAINING_K_HI * sxs["mf_ringdown"],
        np.nextafter(0.5 / support.TRAINING_DTAU, 0.0),
    )
    duration_samples = (
        KW.signal_duration(
            parameters["symmetric_mass_ratio"], display_mf_min
        )
        + 500.0
    ) / support.TRAINING_DTAU
    n_fft = support._next_power_of_two(
        max(
            support.TRAINING_PAD_FACTOR * len(sxs["strain"]),
            support.TRAINING_PAD_FACTOR * len(nrsur["strain"]),
            1.2 * duration_samples,
        )
    )
    mf_full, negative_indices = support._positive_frequency_grid(n_fft)
    frequency_hz_full = mf_full / mass_seconds
    delta_mf = float(mf_full[1] - mf_full[0])
    delta_f_hz = delta_mf / mass_seconds

    spectra_full: dict[str, np.ndarray] = {
        "SXS": support._time_to_positive_spectrum(
            sxs["strain"],
            delta_t_seconds,
            n_fft,
            negative_indices,
        ),
        "NRSur7dq4": support._time_to_positive_spectrum(
            nrsur["strain"],
            delta_t_seconds,
            n_fft,
            negative_indices,
        ),
    }

    display = (mf_full >= display_mf_min) & (mf_full <= display_mf_max)
    display_indices = np.flatnonzero(display)
    # Reconstruct the KAN component masses from eta after clipping only
    # floating-point round-off above the physical equal-mass boundary.  SXS
    # equal-mass cases can otherwise produce eta=0.25000000000000006 when the
    # masses are reconstructed from q, which fails KAN's strict eta<=0.25
    # check even though the physical system lies exactly on the boundary.
    eta_for_kan = min(parameters["symmetric_mass_ratio"], 0.25)
    delta_for_kan = np.sqrt(max(1.0 - 4.0 * eta_for_kan, 0.0))
    mass1_msun = 0.5 * total_mass_msun * (1.0 + delta_for_kan)
    mass2_msun = total_mass_msun - mass1_msun
    evaluator = KW.KAN
    kan_full = np.full_like(
        mf_full,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    kan_full[display_indices] = evaluator(
        frequency_hz_full[display_indices],
        mass1_msun,
        mass2_msun,
        parameters["chi1z"],
        parameters["chi2z"],
        luminosity_distance_mpc,
        strict_domain=True,
    )
    spectra_full["KAN"] = kan_full
    spectra_full["IMRPhenomD"] = _generate_imrphenomd(
        parameters,
        total_mass_msun,
        luminosity_distance_mpc,
        delta_f_hz,
        display_mf_min / mass_seconds,
        display_mf_max / mass_seconds,
        frequency_hz_full,
    )

    common = (mf_full >= common_mf_min) & (mf_full <= common_mf_max)
    for values in spectra_full.values():
        common &= np.isfinite(values) & (np.abs(values) > 0.0)
    common_indices = support._largest_true_run(common)
    if len(common_indices) < 32:
        raise ValueError(
            f"Only {len(common_indices)} contiguous bins are valid for all "
            "four waveforms"
        )

    common_mf = mf_full[common_indices]
    common_frequency_hz = frequency_hz_full[common_indices]
    common_spectra = {
        name: np.asarray(values[common_indices], dtype=np.complex128)
        for name, values in spectra_full.items()
    }
    frequency_spectra, frequency_alignment = _frequency_align_to_sxs(
        common_spectra,
        common_mf,
    )

    sensitivity_first_index = int(
        np.ceil(SENSITIVITY_PLOT_MIN_HZ / delta_f_hz)
    )
    sensitivity_last_index = int(
        np.floor(SENSITIVITY_PLOT_MAX_HZ / delta_f_hz)
    )
    sensitivity_hz = (
        np.arange(
            sensitivity_first_index,
            sensitivity_last_index + 1,
            dtype=float,
        )
        * delta_f_hz
    )
    try:
        from pycbc.psd import (
            aLIGOAdVO4IntermediateT1800545,
            aLIGOZeroDetHighPower,
        )
    except ImportError as exc:
        raise ImportError(
            "PyCBC with the requested aLIGO design and O4-intermediate PSD "
            "models is required."
        ) from exc

    design_psd = _pycbc_psd_on_grid(
        aLIGOZeroDetHighPower,
        sensitivity_hz,
        delta_f_hz,
    )
    o4_intermediate_psd = _pycbc_psd_on_grid(
        aLIGOAdVO4IntermediateT1800545,
        sensitivity_hz,
        delta_f_hz,
    )
    sensitivity_mask = (
        np.isfinite(design_psd)
        & (design_psd > 0.0)
        & np.isfinite(o4_intermediate_psd)
        & (o4_intermediate_psd > 0.0)
    )
    sensitivity_hz = sensitivity_hz[sensitivity_mask]
    design_amplitude_sensitivity = np.sqrt(design_psd[sensitivity_mask])
    o4_intermediate_amplitude_sensitivity = np.sqrt(
        o4_intermediate_psd[sensitivity_mask]
    )

    # Use the common-band time/phase-maximizing shifts for the time panel too.
    # The FFT phase convention treats the first sample of each native TD array
    # as t=0.  Express every series relative to the SXS peak after applying the
    # fitted shift.  A single global phase rotation makes Re[h] positive at the
    # SXS peak without changing any relative phase.
    sxs_peak_index = int(np.argmax(np.abs(sxs["strain"])))
    sxs_peak_time_from_start = sxs_peak_index * delta_t_seconds
    sxs_peak_phase = float(np.angle(sxs["strain"][sxs_peak_index]))
    global_phase_rotation = np.exp(-1j * sxs_peak_phase)
    sxs_time = (
        np.arange(len(sxs["strain"]), dtype=float) * delta_t_seconds
        - sxs_peak_time_from_start
    )
    sxs_time_strain = (
        np.asarray(sxs["strain"], dtype=np.complex128)
        * global_phase_rotation
    )

    nrsur_shift = frequency_alignment["NRSur7dq4"]
    nrsur_time = (
        np.arange(len(nrsur["strain"]), dtype=float) * delta_t_seconds
        + nrsur_shift["time_shift_M"] * mass_seconds
        - sxs_peak_time_from_start
    )
    nrsur_time_strain = (
        np.asarray(nrsur["strain"], dtype=np.complex128)
        * np.exp(-1j * nrsur_shift["phase_shift_radians"])
        * global_phase_rotation
    )

    # One inverse transform for each native FD model, after applying the same
    # common-band alignment to its wider display spectrum.
    display_f_lower_hz = display_mf_min / mass_seconds
    display_f_upper_hz = display_mf_max / mass_seconds
    reconstruction_taper = _display_band_taper(
        frequency_hz_full,
        display_f_lower_hz,
        display_f_upper_hz,
    )
    aligned_display_spectra: dict[str, np.ndarray] = {}
    for name in ("KAN", "IMRPhenomD"):
        shift = frequency_alignment[name]
        aligned_display_spectra[name] = spectra_full[name] * np.exp(
            -2j * np.pi * mf_full * shift["time_shift_M"]
            + 1j * shift["phase_shift_radians"]
        )

    kan_time_strain = _inverse_positive_spectrum(
        aligned_display_spectra["KAN"],
        negative_indices,
        delta_t_seconds,
        reconstruction_taper,
    )
    phenom_time_strain = _inverse_positive_spectrum(
        aligned_display_spectra["IMRPhenomD"],
        negative_indices,
        delta_t_seconds,
        reconstruction_taper,
    )
    inverse_time = (
        np.arange(n_fft, dtype=float) * delta_t_seconds
        - sxs_peak_time_from_start
    )
    kan_time = inverse_time
    phenom_time = inverse_time
    kan_time_strain *= global_phase_rotation
    phenom_time_strain *= global_phase_rotation

    time_series = {
        "KAN": (kan_time, kan_time_strain),
        "IMRPhenomD": (phenom_time, phenom_time_strain),
        "SXS": (sxs_time, sxs_time_strain),
        "NRSur7dq4": (nrsur_time, nrsur_time_strain),
    }
    common_ringdown_end_seconds = min(
        float(time_seconds[-1]) for time_seconds, _ in time_series.values()
    )
    if common_ringdown_end_seconds <= 0.0:
        raise ValueError(
            "The aligned waveforms contain no common post-peak ringdown"
        )
    time_window_seconds = (
        -400.0 * mass_seconds,
        common_ringdown_end_seconds,
    )
    time_window_M = (
        -400.0,
        common_ringdown_end_seconds / mass_seconds,
    )
    for name, (time_seconds, _) in time_series.items():
        samples = (time_seconds >= time_window_seconds[0]) & (
            time_seconds <= time_window_seconds[1]
        )
        if np.count_nonzero(samples) < 32:
            raise ValueError(
                f"{name} has insufficient samples in the requested 400 M "
                "inspiral-plus-ringdown display interval"
            )

    return {
        "sxs_id": str(sxs_id),
        "total_mass_msun": total_mass_msun,
        "luminosity_distance_mpc": luminosity_distance_mpc,
        "parameters": parameters,
        "frequency_hz": common_frequency_hz,
        "frequency_Mf": common_mf,
        "frequency_spectra": frequency_spectra,
        "frequency_alignment": frequency_alignment,
        "sensitivity_frequency_hz": sensitivity_hz,
        "amplitude_sensitivity": design_amplitude_sensitivity,
        "design_amplitude_sensitivity": design_amplitude_sensitivity,
        "o4_intermediate_amplitude_sensitivity": (
            o4_intermediate_amplitude_sensitivity
        ),
        "sensitivity_models": {
            "design": DESIGN_PSD_NAME,
            "O4_intermediate": O4_INTERMEDIATE_PSD_NAME,
        },
        "time_series": time_series,
        "time_window_seconds": time_window_seconds,
        "time_window_M": time_window_M,
        "conditioning": {
            "SXS": sxs["window"],
            "NRSur7dq4": nrsur["window"],
            "start_taper_M": NRSUR_START_TAPER_M,
            "common_Mf_min": float(common_mf[0]),
            "common_Mf_max": float(common_mf[-1]),
            "common_f_min_hz": float(common_frequency_hz[0]),
            "common_f_max_hz": float(common_frequency_hz[-1]),
            "display_f_min_hz": display_f_lower_hz,
            "display_f_max_hz": display_f_upper_hz,
            "sensitivity_plot_f_min_hz": SENSITIVITY_PLOT_MIN_HZ,
            "sensitivity_plot_f_max_hz": SENSITIVITY_PLOT_MAX_HZ,
        },
        "transform_policy": {
            "SXS": "native TD; one TD-to-FD transform",
            "NRSur7dq4": "native TD; one TD-to-FD transform",
            "KAN": (
                "native FD; time/phase aligned in FD; one FD-to-TD "
                "transform for display"
            ),
            "IMRPhenomD": (
                "native FD; time/phase aligned in FD; one FD-to-TD "
                "transform for display"
            ),
            "round_trip": False,
        },
    }


def _plot_stride(length: int, maximum_points: int = 12000) -> int:
    return max(1, int(np.ceil(length / maximum_points)))


def plot_waveforms(
    sxs_id: str,
    total_mass_msun: float,
    luminosity_distance_mpc: float,
    *,
    output_path: str | Path | None = None,
    download: bool = True,
    progress: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """Prepare the waveforms and create the requested two-panel figure."""

    import matplotlib.pyplot as plt

    result = prepare_waveforms(
        sxs_id,
        total_mass_msun,
        luminosity_distance_mpc,
        download=download,
        progress=progress,
    )

    figure, (frequency_amplitude_axis, frequency_phase_axis, time_axis) = (
        plt.subplots(
            3,
            1,
            figsize=(10.0, 11.0),
            constrained_layout=True,
            gridspec_kw={"height_ratios": (1.0, 1.0, 1.25)},
        )
    )

    frequency_hz = result["frequency_hz"]
    frequency_stride = _plot_stride(len(frequency_hz))
    for name in MODEL_NAMES:
        spectrum = result["frequency_spectra"][name]
        frequency_amplitude_axis.plot(
            frequency_hz[::frequency_stride],
            np.abs(spectrum[::frequency_stride]),
            label=name,
            **MODEL_PLOT_STYLES[name],
        )
        frequency_phase_axis.plot(
            frequency_hz[::frequency_stride],
            np.unwrap(np.angle(spectrum))[::frequency_stride],
            label=name,
            **MODEL_PLOT_STYLES[name],
        )
    for axis in (frequency_amplitude_axis, frequency_phase_axis):
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
    frequency_amplitude_axis.legend(
        ncol=2,
        fontsize="small",
        loc="upper left",
    )
    frequency_phase_axis.legend(
        ncol=4,
        fontsize="small",
        loc="upper right",
    )
    frequency_amplitude_axis.set_yscale("log")
    frequency_amplitude_axis.set_ylabel(
        r"$|\tilde h(f)|$ [strain Hz$^{-1}$]"
    )
    frequency_amplitude_axis.set_xlabel("frequency [Hz]")
    frequency_amplitude_axis.set_xlim(
        SENSITIVITY_PLOT_MIN_HZ,
        SENSITIVITY_PLOT_MAX_HZ,
    )
    frequency_amplitude_axis.set_title(
        "Fourier-domain amplitude and detector sensitivity"
    )
    frequency_phase_axis.set_ylabel("unwrapped phase [rad]")
    frequency_phase_axis.set_xlabel("frequency [Hz]")
    frequency_phase_axis.set_title("Fourier-domain phase")

    sensitivity_axis = frequency_amplitude_axis.twinx()
    sensitivity_axis.plot(
        result["sensitivity_frequency_hz"],
        result["design_amplitude_sensitivity"],
        color="#666666",
        linewidth=3.0,
        linestyle=(0, (1.0, 1.5)),
        label="aLIGO design ASD",
        zorder=1,
    )
    sensitivity_axis.plot(
        result["sensitivity_frequency_hz"],
        result["o4_intermediate_amplitude_sensitivity"],
        color="#CC79A7",
        linewidth=3.0,
        linestyle=(0, (8.0, 2.0, 1.5, 2.0)),
        label="aLIGO O4 intermediate ASD",
        zorder=1,
    )
    sensitivity_axis.set_yscale("log")
    sensitivity_axis.set_ylabel(
        r"$\sqrt{S_n(f)}$ [strain Hz$^{-1/2}$]",
        color="0.35",
    )
    sensitivity_axis.tick_params(axis="y", colors="0.35")
    sensitivity_axis.legend(loc="lower right", fontsize="small")

    time_window_min, time_window_max = result["time_window_seconds"]
    for name in MODEL_NAMES:
        time_seconds, strain = result["time_series"][name]
        visible = (time_seconds >= time_window_min) & (
            time_seconds <= time_window_max
        )
        visible_indices = np.flatnonzero(visible)
        stride = _plot_stride(len(visible_indices))
        visible_indices = visible_indices[::stride]
        time_axis.plot(
            time_seconds[visible_indices],
            np.real(strain[visible_indices]),
            label=name,
            **MODEL_PLOT_STYLES[name],
        )
    time_axis.set_xlabel("time from SXS amplitude peak [s]")
    time_axis.set_ylabel(r"Re$[h(t)]$ [strain]")
    time_axis.set_xlim(time_window_min, time_window_max)
    time_axis.grid(True, alpha=0.25)
    time_axis.legend(loc="best")
    ringdown_end_M = result["time_window_M"][1]
    time_axis.set_title(
        "Time-domain waveforms: "
        rf"$-400M$ through the common ringdown end "
        rf"$(+{ringdown_end_M:.0f}M)$"
    )

    figure.suptitle(
        f"{result['sxs_id']}: "
        rf"$M={result['total_mass_msun']:g}\,M_\odot$, "
        rf"$D_L={result['luminosity_distance_mpc']:g}$ Mpc"
    )
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        result["output_path"] = destination
    if show:
        plt.show()

    result["figure"] = figure
    result["frequency_axis"] = frequency_amplitude_axis
    result["frequency_amplitude_axis"] = frequency_amplitude_axis
    result["frequency_phase_axis"] = frequency_phase_axis
    result["sensitivity_axis"] = sensitivity_axis
    result["time_axis"] = time_axis
    return result


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sxs_id", help="for example SXS:BBH:0198")
    parser.add_argument("total_mass_msun", type=float, help="for example 60")
    parser.add_argument(
        "luminosity_distance_mpc",
        type=float,
        help="for example 400",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    output_path = args.output
    if output_path is None:
        safe_id = args.sxs_id.replace(":", "_")
        output_path = Path("figures") / f"{safe_id}_waveform_plot.png"
    result = plot_waveforms(
        args.sxs_id,
        args.total_mass_msun,
        args.luminosity_distance_mpc,
        output_path=output_path,
        download=not args.no_download,
        progress=not args.no_progress,
        show=args.show,
    )
    print(f"Saved {result['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
