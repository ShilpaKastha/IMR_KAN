"""Focused local utilities for the two mismatch command-line modules.

"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from KAN_utilities import KAN_Waveform as KW


MTSUN_SI = 4.925491025543576e-6
MPC_SI = 3.085677581491367e22
C_SI = 299792458.0
Y22_FACEON = float(np.sqrt(5.0 / (4.0 * np.pi)))

FIDUCIAL_TOTAL_MASS_MSUN = 60.0
FIDUCIAL_DISTANCE_MPC = 100.0

TRAINING_DTAU = 0.5
TRAINING_JUNK_PAD_M = 200.0
TRAINING_PAD_FACTOR = 4
TRAINING_WINDOW_ON_M = 600.0
TRAINING_WINDOW_OFF_START_FRAC = 1.0e-2
TRAINING_WINDOW_OFF_END_FRAC = 1.0e-4
TRAINING_MIN_WINDOW_OFF_M = 10.0
TRAINING_PROTECT_TAU = 4.0
TRAINING_PROTECT_PRE_M = 50.0
TRAINING_K_LO = 2.0
TRAINING_K_HI = 1.25
TRAINING_HARD_MF_MIN = 0.0059

DEFAULT_MINIMAL_EDGE_FRACTION = 0.005

MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = MODULE_DIRECTORY / "waveform_data"

NRSUR_FILENAME = "NRSur7dq4_v1.0.h5"
NRSUR_URL = "https://dcc.ligo.org/public/0198/T2500012/004/NRSur7dq4_v1.0.h5"
NRSUR_MD5 = "ec4a0a9c6af39ad14dc91d42957144e9"


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} ...")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if total:
                print(
                    f"\r  {received / (1024**2):.1f} / "
                    f"{total / (1024**2):.1f} MiB",
                    end="",
                    flush=True,
                )
    if total:
        print()
    actual_md5 = _file_md5(partial)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Checksum failure for {partial}: expected {expected_md5}, "
            f"received {actual_md5}"
        )
    partial.replace(destination)


def _ensure_nrsur_data(data_directory: Path, download: bool) -> Path:
    path = data_directory / NRSUR_FILENAME
    if path.exists():
        actual_md5 = _file_md5(path)
        if actual_md5 != NRSUR_MD5:
            raise RuntimeError(
                f"Existing NRSur7dq4 file has checksum {actual_md5}, "
                f"expected {NRSUR_MD5}: {path}"
            )
    elif download:
        _download_file(NRSUR_URL, path, NRSUR_MD5)
    else:
        raise FileNotFoundError(
            f"Missing NRSur7dq4 coefficients: {path}. Enable downloading."
        )
    os.environ["LAL_DATA_PATH"] = str(data_directory.resolve())
    return path


def _load_sxs(
    sxs_id: str,
    data_directory: Path,
    download: bool,
    progress: bool,
) -> Any:
    cache = data_directory / "sxs_cache"
    config = data_directory / "sxs_config"
    cache.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    os.environ["SXSCACHEDIR"] = str(cache.resolve())
    os.environ["SXSCONFIGDIR"] = str(config.resolve())
    try:
        import sxs
    except ImportError as exc:
        raise ImportError(
            "The 'sxs' package is required; install the dependencies in "
            "requirements-waveform-comparison.txt."
        ) from exc
    return sxs.load(
        str(sxs_id),
        download=bool(download),
        cache=True,
        progress=bool(progress),
    )


def _binary_parameters(metadata: Any) -> dict[str, float]:
    mass1 = float(metadata.reference_mass1)
    mass2 = float(metadata.reference_mass2)
    spin1 = np.asarray(metadata.reference_dimensionless_spin1, dtype=float)
    spin2 = np.asarray(metadata.reference_dimensionless_spin2, dtype=float)
    if spin1.shape != (3,) or spin2.shape != (3,):
        raise ValueError("SXS metadata does not contain two 3-component spins")
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1

    in_plane = float(np.max(np.abs(np.r_[spin1[:2], spin2[:2]])))
    if in_plane > 5.0e-3:
        raise ValueError(
            "This SXS case is precessing, while the comparison uses aligned-"
            f"spin models (largest in-plane spin={in_plane:.4g})."
        )

    total = mass1 + mass2
    fraction1 = mass1 / total
    fraction2 = mass2 / total
    mass_ratio = fraction1 / fraction2
    chi1z = float(spin1[2])
    chi2z = float(spin2[2])
    if mass_ratio > 6.01:
        raise ValueError(
            f"q={mass_ratio:.5g} is outside NRSur7dq4's usable range."
        )
    if max(abs(chi1z), abs(chi2z)) > 0.801:
        raise ValueError("The aligned spins exceed |chi_z|<=0.8.")
    return {
        "mass1_fraction": float(fraction1),
        "mass2_fraction": float(fraction2),
        "mass_ratio": float(mass_ratio),
        "symmetric_mass_ratio": float(fraction1 * fraction2),
        "chi1z": chi1z,
        "chi2z": chi2z,
    }


def _extract_sxs_mode(simulation: Any) -> dict[str, Any]:
    waveform = simulation.h
    tau = np.asarray(waveform.t, dtype=float)
    h22 = np.asarray(
        waveform.data[:, waveform.index(2, 2)],
        dtype=np.complex128,
    )
    if len(tau) < 32 or not np.all(np.diff(tau) > 0.0):
        raise ValueError("The SXS (2,2) mode has an invalid time grid")
    if not np.all(np.isfinite(h22)):
        raise ValueError("The SXS (2,2) mode contains non-finite values")
    return {
        "tau": tau,
        "h22": h22,
        "t_peak": float(waveform.max_norm_time()),
        "reference_time": float(simulation.metadata.reference_time),
    }


def _resample_complex(
    source_time: np.ndarray,
    source_values: np.ndarray,
    start: float,
    stop: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(float(start), float(stop), float(step))
    if len(grid) < 32:
        raise ValueError("A waveform segment is too short after resampling")
    real = np.interp(grid, source_time, np.real(source_values))
    imag = np.interp(grid, source_time, np.imag(source_values))
    return grid, real + 1j * imag


def _planck_window(length: int, n_on: int, n_off: int) -> np.ndarray:
    window = np.ones(int(length), dtype=float)
    if n_on > 2:
        x = np.linspace(0.0, 1.0, int(n_on), endpoint=False)[1:]
        window[0] = 0.0
        z = np.clip(1.0 / x - 1.0 / (1.0 - x), -500.0, 500.0)
        window[1:int(n_on)] = 1.0 / (1.0 + np.exp(z))
    if n_off > 2:
        x = np.linspace(0.0, 1.0, int(n_off), endpoint=False)[1:]
        window[-1] = 0.0
        z = np.clip(1.0 / x - 1.0 / (1.0 - x), -500.0, 500.0)
        window[-int(n_off):-1] = (1.0 / (1.0 + np.exp(z)))[::-1]
    return window


def _ringdown_damping_time(parameters: dict[str, float]) -> float:
    m1 = parameters["mass1_fraction"]
    m2 = parameters["mass2_fraction"]
    chi1z = parameters["chi1z"]
    chi2z = parameters["chi2z"]
    final_spin = float(KW.final_spin_estimate(m1, m2, chi1z, chi2z))
    final_mass_fraction = 1.0 - float(
        KW.radiated_energy_estimate(m1, m2, chi1z, chi2z)
    )
    return float(
        final_mass_fraction
        * 2.0
        * KW.qnm_quality_22(final_spin)
        / KW.qnm_omega_22(final_spin)
    )


def _instantaneous_mf(tau: np.ndarray, strain: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(np.asarray(strain)))
    return np.abs(np.gradient(phase, np.asarray(tau))) / (2.0 * np.pi)


def _minimal_endpoint_window(
    length: int,
    edge_fraction: float,
) -> tuple[np.ndarray, int]:
    if not 0.0 < edge_fraction <= 0.05:
        raise ValueError("minimal edge fraction must lie in (0, 0.05]")
    edge_samples = max(8, int(np.ceil(edge_fraction * length)))
    edge_samples = min(edge_samples, length // 4)
    ramp = np.sin(np.linspace(0.0, 0.5 * np.pi, edge_samples)) ** 2
    window = np.ones(length, dtype=float)
    window[:edge_samples] = ramp
    window[-edge_samples:] = ramp[::-1]
    return window, edge_samples


def _minimal_sxs(
    raw: dict[str, Any],
    physical_strain_scale: float,
    edge_fraction: float,
) -> dict[str, Any]:
    tau, h22 = _resample_complex(
        raw["tau"],
        raw["h22"],
        float(raw["tau"][0]),
        float(raw["tau"][-1]),
        TRAINING_DTAU,
    )
    window, edge_samples = _minimal_endpoint_window(len(tau), edge_fraction)
    return {
        "strain": physical_strain_scale * h22 * window,
        "tau_absolute_M": tau,
        "edge_samples": edge_samples,
        "edge_duration_M": edge_samples * TRAINING_DTAU,
    }


def _next_power_of_two(value: float | int) -> int:
    return 1 << int(np.ceil(np.log2(max(2, int(np.ceil(value))))))


def _positive_frequency_grid(n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    signed_mf = np.fft.fftfreq(n_fft, d=TRAINING_DTAU)
    negative_indices = np.flatnonzero(signed_mf < 0.0)[::-1]
    return -signed_mf[negative_indices], negative_indices


def _time_to_positive_spectrum(
    strain: np.ndarray,
    delta_t: float,
    n_fft: int,
    negative_indices: np.ndarray,
) -> np.ndarray:
    if len(strain) > n_fft:
        raise ValueError("FFT grid is shorter than a time-domain waveform")
    spectrum = delta_t * np.fft.fft(np.asarray(strain), n=n_fft)
    return 0.5 * np.conj(spectrum[negative_indices])


def _generate_phenomd_frequency(
    parameters: dict[str, float],
    delta_f_hz: float,
    f_lower_hz: float,
    f_final_hz: float,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        from pycbc.waveform import get_fd_waveform
    except ImportError as exc:
        raise ImportError(
            "PyCBC is required; install requirements-waveform-comparison.txt."
        ) from exc

    mass1 = FIDUCIAL_TOTAL_MASS_MSUN * parameters["mass1_fraction"]
    mass2 = FIDUCIAL_TOTAL_MASS_MSUN * parameters["mass2_fraction"]
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=mass1,
        mass2=mass2,
        spin1z=parameters["chi1z"],
        spin2z=parameters["chi2z"],
        distance=FIDUCIAL_DISTANCE_MPC,
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
    nonzero = np.isfinite(output) & (np.abs(output) > 0.0)
    if np.count_nonzero(nonzero) < 32:
        raise ValueError("IMRPhenomD has insufficient common-grid support")
    return output, {
        "generation_f_lower_hz": float(f_lower_hz),
        "returned_nonzero_f_min_hz": float(frequency_hz[nonzero][0]),
        "returned_nonzero_f_max_hz": float(frequency_hz[nonzero][-1]),
    }


def _largest_true_run(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return indices
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    runs = np.split(indices, breaks)
    return max(runs, key=len)


def _maximize_time_phase_match(
    reference: np.ndarray,
    candidate: np.ndarray,
    mf: np.ndarray,
    preferred_time_shift_M: float | None,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    mf = np.asarray(mf, dtype=float)
    if not (len(reference) == len(candidate) == len(mf)):
        raise ValueError("Reference, candidate, and frequency grids must match")
    if len(mf) < 32:
        raise ValueError("At least 32 common frequency bins are required")
    delta_mf = float(np.median(np.diff(mf)))
    if not np.allclose(np.diff(mf), delta_mf, rtol=1.0e-8, atol=1.0e-14):
        raise ValueError("Time maximization requires a uniform frequency grid")

    reference_norm = np.sqrt(np.sum(np.abs(reference) ** 2) * delta_mf)
    candidate_norm = np.sqrt(np.sum(np.abs(candidate) ** 2) * delta_mf)
    if reference_norm <= 0.0 or candidate_norm <= 0.0:
        raise ValueError("Cannot match a zero-norm waveform")
    cross = reference * np.conj(candidate) * delta_mf

    def correlation(time_shift_M: float) -> complex:
        return np.sum(cross * np.exp(2j * np.pi * mf * time_shift_M))

    period_M = 1.0 / delta_mf
    correlation_size = _next_power_of_two(16 * len(cross))
    sampled = correlation_size * np.fft.ifft(cross, n=correlation_size)
    best_index = int(np.argmax(np.abs(sampled)))
    spacing_M = period_M / correlation_size
    centre_M = best_index * spacing_M
    if centre_M >= 0.5 * period_M:
        centre_M -= period_M
    refined = minimize_scalar(
        lambda shift: -abs(correlation(float(shift))),
        bounds=(centre_M - spacing_M, centre_M + spacing_M),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    time_shift_M = float(refined.x)
    if preferred_time_shift_M is not None:
        time_shift_M += round(
            (float(preferred_time_shift_M) - time_shift_M) / period_M
        ) * period_M
    best_correlation = correlation(time_shift_M)
    match = float(
        np.clip(
            abs(best_correlation) / (reference_norm * candidate_norm),
            0.0,
            1.0,
        )
    )
    return {
        "match": match,
        "mismatch": 1.0 - match,
        "time_shift_M": time_shift_M,
        "phase_shift_radians": float(np.angle(best_correlation)),
        "correlation_period_M": period_M,
    }
