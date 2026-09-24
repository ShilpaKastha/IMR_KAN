"""Compare SXS, KAN, NRSur7dq4, IMRPhenomD, and the TaylorF2/Newtonian baseline.

The only required inputs are an SXS ID and an NRSur7dq4 Planck-window turn-on
duration, chosen from 500 M, 600 M, or 700 M.  The SXS reference retains the
exact 600 M training turn-on.  NRSur7dq4 receives the selected turn-on and the
same ringdown-safe adaptive turn-off prescription used for the SXS targets.

SXS and NRSur7dq4 are transformed once from time to frequency.  KAN is
evaluated directly through ``KAN_utilities/KAN_Waveform.py`` and IMRPhenomD is
requested directly in frequency space from PyCBC.  Neither frequency-domain
model is transformed to time and back.  The TaylorF2-phase/Newtonian-amplitude
PN backbone used by KAN is also evaluated directly in frequency space.  All
mismatches use one uniform grid, the intersection of the trustworthy supports,
flat frequency weighting, and maximization over relative time and constant
phase. Use ``models`` (or ``--models`` on the command line) to select a subset;
only selected models contribute to the common band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np

# When this file is loaded by path (for example with importlib in a notebook),
# Python does not necessarily add its parent directory to ``sys.path``.  Put
# this module's own directory first so all companion imports are resolved only
# from this repository, independently of the caller's working directory.
MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from KAN_utilities import KAN_Waveform as KW
from . import _mismatch_support as base


ALLOWED_START_TAPERS_M = (500.0, 600.0, 700.0)
REFERENCE_NAME = "Training-conditioned SXS"
PN_BACKBONE_NAME = "TaylorF2 PN backbone"
CORE_CANDIDATE_NAMES = (
    "KAN",
    "NRSur7dq4",
    "IMRPhenomD",
)
CANDIDATE_NAMES = CORE_CANDIDATE_NAMES + (PN_BACKBONE_NAME,)


def _select_models(models: str | Sequence[str] | None, include_pn_backbone: bool) -> tuple[str, ...]:
    """Normalize requested names, retaining historical result dictionary keys."""
    if models is None:
        return CANDIDATE_NAMES if include_pn_backbone else CORE_CANDIDATE_NAMES
    names = [models] if isinstance(models, str) else list(models)
    aliases = {name.casefold(): name for name in CANDIDATE_NAMES}
    aliases["pn backbone"] = PN_BACKBONE_NAME
    selected = []
    for name in names:
        canonical = aliases.get(name.strip().casefold()) if isinstance(name, str) else None
        if canonical is None:
            raise ValueError(f"Unknown waveform model {name!r}; choose KAN, IMRPhenomD, NRSur7dq4, or PN Backbone")
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("models must contain at least one waveform model")
    return tuple(selected)


def _validate_start_taper(start_taper_M: float) -> float:
    value = float(start_taper_M)
    for allowed in ALLOWED_START_TAPERS_M:
        if np.isclose(value, allowed, rtol=0.0, atol=1.0e-12):
            return allowed
    choices = ", ".join(f"{value:g}M" for value in ALLOWED_START_TAPERS_M)
    raise ValueError(f"start_taper_M must be one of: {choices}")


def _parse_start_taper(text: str) -> float:
    value = str(text).strip().lower()
    if value.endswith("m"):
        value = value[:-1].strip()
    try:
        return _validate_start_taper(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_nrsur_calibration(parameters: dict[str, float]) -> None:
    """Enforce the strict NRSur7dq4 calibration used by the final audit."""

    q = float(parameters["mass_ratio"])
    spin = max(abs(parameters["chi1z"]), abs(parameters["chi2z"]))
    if q > 4.0:
        raise ValueError(
            f"q={q:.12g} is outside the strict NRSur7dq4 calibration q<=4; "
            "extrapolation is disabled."
        )
    if spin > 0.8:
        raise ValueError(
            f"max(|chi1z|, |chi2z|)={spin:.12g} exceeds the strict "
            "NRSur7dq4 calibration |chi|<=0.8; extrapolation is disabled."
        )


def _adaptive_window(
    tau: np.ndarray,
    strain: np.ndarray,
    t_peak: float,
    tau_ringdown: float,
    start_taper_M: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Training adaptive window with a selectable Planck turn-on length."""

    amplitude = np.abs(strain)
    length = len(tau)
    peak_index = int(np.argmin(np.abs(tau - t_peak)))
    peak_amplitude = float(amplitude[peak_index])

    protect_end = t_peak + base.TRAINING_PROTECT_TAU * tau_ringdown
    protect_index = int(np.argmin(np.abs(tau - protect_end)))
    candidates = np.where(
        (np.arange(length) > protect_index)
        & (amplitude <= base.TRAINING_WINDOW_OFF_START_FRAC * peak_amplitude)
    )[0]
    off_index = int(candidates[0]) if candidates.size else -1
    ok = True
    reason = "adaptive amplitude threshold"
    if off_index < 0:
        off_index = max(
            protect_index + 1,
            length
            - int(base.TRAINING_MIN_WINDOW_OFF_M / base.TRAINING_DTAU),
        )
        ok = off_index < length - 2
        reason = "record ends before threshold; turn-off pinned after protection"
    n_off = length - off_index
    if n_off * base.TRAINING_DTAU < base.TRAINING_MIN_WINDOW_OFF_M:
        ok = False
        reason = "no admissible turn-off of the minimum width"

    protect_start = t_peak - base.TRAINING_PROTECT_PRE_M
    protect_start_index = int(np.argmin(np.abs(tau - protect_start)))
    n_on = int(round(start_taper_M / base.TRAINING_DTAU))
    if n_on >= protect_start_index:
        n_on = max(0, protect_start_index - 1)
        reason += "; turn-on truncated to clear the protected interval"

    window = base._planck_window(length, n_on, n_off if ok else 0)
    protected = (tau >= protect_start) & (tau <= protect_end)
    information = {
        "ok": bool(ok),
        "reason": reason,
        "requested_turn_on_M": float(start_taper_M),
        "turn_on_samples": int(n_on),
        "turn_on_duration_M": float(n_on * base.TRAINING_DTAU),
        "turn_on_start_M": float(tau[0]),
        "turn_on_end_M": float(tau[min(n_on, length - 1)]),
        "turn_off_samples": int(n_off if ok else 0),
        "turn_off_duration_M": float(
            n_off * base.TRAINING_DTAU if ok else 0.0
        ),
        "turn_off_start_M": float(tau[off_index]) if ok else float("nan"),
        "turn_off_end_M": float(tau[-1]) if ok else float("nan"),
        "protect_start_M": float(protect_start),
        "protect_end_M": float(protect_end),
        "max_window_deviation_in_protected_region": float(
            np.max(np.abs(window[protected] - 1.0))
            if np.any(protected)
            else 0.0
        ),
    }
    return window, information


def _measured_start_mf(tau: np.ndarray, strain: np.ndarray) -> float:
    count = min(
        len(strain),
        max(16, int(200.0 / base.TRAINING_DTAU)),
    )
    frequency = base._instantaneous_mf(tau[:count], strain[:count])
    useful = frequency[np.isfinite(frequency) & (frequency > 0.0)]
    if len(useful) < 8:
        raise ValueError("Could not estimate a waveform starting frequency")
    return float(np.median(useful))


def _condition_sxs_reference(
    raw: dict[str, Any],
    parameters: dict[str, float],
    physical_strain_scale: float,
) -> dict[str, Any]:
    tau_raw = np.asarray(raw["tau"], dtype=float)
    h22_raw = np.asarray(raw["h22"], dtype=np.complex128)
    t_peak = float(raw["t_peak"])
    amplitude = np.abs(h22_raw)
    peak_amplitude = float(
        amplitude[int(np.argmin(np.abs(tau_raw - t_peak)))]
    )

    start = float(raw["reference_time"] + base.TRAINING_JUNK_PAD_M)
    late = (tau_raw > t_peak) & (
        amplitude < base.TRAINING_WINDOW_OFF_END_FRAC * peak_amplitude
    )
    stop = float(tau_raw[late][0]) if np.any(late) else float(tau_raw[-1])
    tau, h22 = base._resample_complex(
        tau_raw,
        h22_raw,
        start,
        stop,
        base.TRAINING_DTAU,
    )

    tau_ringdown = base._ringdown_damping_time(parameters)
    window, window_info = _adaptive_window(
        tau,
        h22,
        t_peak,
        tau_ringdown,
        base.TRAINING_WINDOW_ON_M,
    )
    if not window_info["ok"]:
        raise ValueError(
            "The adaptive SXS turn-off is not admissible: "
            f"{window_info['reason']}"
        )

    mf_start = _measured_start_mf(tau, h22)
    mf_ringdown = float(
        KW.mf_ringdown(
            parameters["mass1_fraction"],
            parameters["mass2_fraction"],
            parameters["chi1z"],
            parameters["chi2z"],
        )
    )
    return {
        "strain": physical_strain_scale * h22 * window,
        "tau": tau,
        "mf_start": mf_start,
        "mf_min": max(base.TRAINING_HARD_MF_MIN, base.TRAINING_K_LO * mf_start),
        "mf_max": base.TRAINING_K_HI * mf_ringdown,
        "mf_ringdown": mf_ringdown,
        "window": window_info,
    }


def _condition_nrsur(
    parameters: dict[str, float],
    delta_t: float,
    start_taper_M: float,
) -> dict[str, Any]:
    _validate_nrsur_calibration(parameters)
    try:
        from pycbc.waveform import get_td_waveform
    except ImportError as exc:
        raise ImportError(
            "PyCBC is required; install requirements-waveform-comparison.txt."
        ) from exc

    mass1 = base.FIDUCIAL_TOTAL_MASS_MSUN * parameters["mass1_fraction"]
    mass2 = base.FIDUCIAL_TOTAL_MASS_MSUN * parameters["mass2_fraction"]
    hp, hc = get_td_waveform(
        approximant="NRSur7dq4",
        mass1=mass1,
        mass2=mass2,
        spin1z=parameters["chi1z"],
        spin2z=parameters["chi2z"],
        distance=base.FIDUCIAL_DISTANCE_MPC,
        inclination=0.0,
        coa_phase=0.0,
        delta_t=delta_t,
        f_lower=0.0,
        mode_array=[(2, 2), (2, -2)],
    )
    strain = np.asarray(hp, dtype=float) - 1j * np.asarray(hc, dtype=float)
    tau = base.TRAINING_DTAU * np.arange(len(strain), dtype=float)
    peak_index = int(np.argmax(np.abs(strain)))
    t_peak = float(tau[peak_index])
    tau_ringdown = base._ringdown_damping_time(parameters)
    window, window_info = _adaptive_window(
        tau,
        strain,
        t_peak,
        tau_ringdown,
        start_taper_M,
    )
    if not window_info["ok"]:
        raise ValueError(
            "The adaptive NRSur7dq4 turn-off is not admissible: "
            f"{window_info['reason']}"
        )

    mf_start = _measured_start_mf(tau, strain)
    mf_ringdown = float(
        KW.mf_ringdown(
            parameters["mass1_fraction"],
            parameters["mass2_fraction"],
            parameters["chi1z"],
            parameters["chi2z"],
        )
    )
    return {
        "strain": strain * window,
        "tau": tau,
        "mf_start": mf_start,
        "mf_min": base.TRAINING_K_LO * mf_start,
        "mf_max": base.TRAINING_K_HI * mf_ringdown,
        "mf_ringdown": mf_ringdown,
        "window": window_info,
    }


def compute_mismatches(
    sxs_id: str,
    start_taper_M: float,
    *,
    models: str | Sequence[str] | None = None,
    include_pn_backbone: bool = True,
    download: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Compare selected models with SXS using flat PSD weighting.

    ``models`` accepts one name or a sequence: KAN, IMRPhenomD, NRSur7dq4,
    or PN Backbone (case insensitive). None preserves the historical default.
    Explicit selections override ``include_pn_backbone``. Only selected models
    are generated, and the common frequency band uses their supports plus SXS.
    Consequently, changing the selection can change the mismatch values.
    The PN result key remains "TaylorF2 PN backbone" for compatibility.
    """

    start_taper_M = _validate_start_taper(start_taper_M)
    candidate_names = _select_models(models, include_pn_backbone)

    # Keep downloaded and cached inputs inside this module's own folder.
    data_dir = Path(base.DEFAULT_DATA_DIRECTORY).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if "NRSur7dq4" in candidate_names:
        base._ensure_nrsur_data(data_dir, download=download)
    simulation = base._load_sxs(
        sxs_id,
        data_dir,
        download=download,
        progress=progress,
    )
    parameters = base._binary_parameters(simulation.metadata)
    if "NRSur7dq4" in candidate_names:
        _validate_nrsur_calibration(parameters)
    raw = base._extract_sxs_mode(simulation)

    mass_seconds = base.FIDUCIAL_TOTAL_MASS_MSUN * base.MTSUN_SI
    distance_seconds = base.FIDUCIAL_DISTANCE_MPC * base.MPC_SI / base.C_SI
    delta_t = base.TRAINING_DTAU * mass_seconds
    physical_strain_scale = (
        mass_seconds / distance_seconds
    ) * base.Y22_FACEON
    geometric_frequency_scale = mass_seconds**2 / distance_seconds

    sxs = _condition_sxs_reference(
        raw,
        parameters,
        physical_strain_scale,
    )
    nrsur = (
        _condition_nrsur(parameters, delta_t, start_taper_M)
        if "NRSur7dq4" in candidate_names else None
    )
    lower_bounds = [sxs["mf_min"]]
    upper_bounds = [sxs["mf_max"], np.nextafter(0.5 / base.TRAINING_DTAU, 0.0)]
    if nrsur is not None:
        lower_bounds.append(nrsur["mf_min"])
        upper_bounds.append(nrsur["mf_max"])
    if "KAN" in candidate_names:
        domain = KW.model_domain()
        lower_bounds.append(float(domain["Mf"][0]))
        upper_bounds.append(float(domain["Mf"][1]))
    preliminary_mf_min = max(lower_bounds)
    preliminary_mf_max = min(upper_bounds)
    if preliminary_mf_max <= preliminary_mf_min:
        raise ValueError(
            f"Empty selected-model common band Mf={preliminary_mf_min:.7g}--"
            f"{preliminary_mf_max:.7g}"
        )

    duration_samples = (
        KW.signal_duration(
            parameters["symmetric_mass_ratio"], preliminary_mf_min
        )
        + 500.0
    ) / base.TRAINING_DTAU
    n_fft = base._next_power_of_two(
        max(
            base.TRAINING_PAD_FACTOR * len(sxs["strain"]),
            base.TRAINING_PAD_FACTOR * len(nrsur["strain"]) if nrsur is not None else 0,
            1.2 * duration_samples,
        )
    )
    mf_full, negative_indices = base._positive_frequency_grid(n_fft)
    frequency_hz_full = mf_full / mass_seconds
    delta_mf = float(mf_full[1] - mf_full[0])
    delta_f_hz = delta_mf / mass_seconds

    spectra_full: dict[str, np.ndarray] = {
        REFERENCE_NAME: base._time_to_positive_spectrum(
            sxs["strain"], delta_t, n_fft, negative_indices
        ),
    }
    if nrsur is not None:
        spectra_full["NRSur7dq4"] = base._time_to_positive_spectrum(
            nrsur["strain"], delta_t, n_fft, negative_indices
        )

    phenom_support = None
    if "IMRPhenomD" in candidate_names:
        phenom, phenom_support = base._generate_phenomd_frequency(
            parameters,
            delta_f_hz,
            preliminary_mf_min / mass_seconds,
            preliminary_mf_max / mass_seconds,
            frequency_hz_full,
        )
        spectra_full["IMRPhenomD"] = phenom

    preliminary = (mf_full >= preliminary_mf_min) & (
        mf_full <= preliminary_mf_max
    )
    preliminary_indices = np.flatnonzero(preliminary)
    if "KAN" in candidate_names:
        evaluator = KW.KAN
        # Reconstruct the masses from eta after clipping only round-off above the
        # physical equal-mass boundary.  This maps values such as
        # 0.25000000000000006 to exactly 0.25 without extrapolating the model.
        eta_for_kan = min(parameters["symmetric_mass_ratio"], 0.25)
        delta_for_kan = np.sqrt(max(1.0 - 4.0 * eta_for_kan, 0.0))
        mass1_msun = 0.5 * base.FIDUCIAL_TOTAL_MASS_MSUN * (
            1.0 + delta_for_kan
        )
        mass2_msun = base.FIDUCIAL_TOTAL_MASS_MSUN - mass1_msun
        kan_values = evaluator(
            frequency_hz_full[preliminary_indices],
            mass1_msun,
            mass2_msun,
            parameters["chi1z"],
            parameters["chi2z"],
            base.FIDUCIAL_DISTANCE_MPC,
            strict_domain=True,
        )
        kan_full = np.full_like(
            mf_full,
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        kan_full[preliminary_indices] = kan_values
        spectra_full["KAN"] = kan_full

    if PN_BACKBONE_NAME in candidate_names:
        preliminary_mf = mf_full[preliminary_indices]
        pn_amplitude = geometric_frequency_scale * np.exp(
            KW.newtonian_log_amp(
                preliminary_mf,
                parameters["symmetric_mass_ratio"],
            )
        )
        pn_phase = -KW.taylorf2_phase(
            preliminary_mf,
            parameters["mass1_fraction"],
            parameters["mass2_fraction"],
            parameters["chi1z"],
            parameters["chi2z"],
        )
        pn_full = np.full_like(
            mf_full,
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        pn_full[preliminary_indices] = pn_amplitude * np.exp(1j * pn_phase)
        spectra_full[PN_BACKBONE_NAME] = pn_full

    valid = preliminary.copy()
    for spectrum in spectra_full.values():
        valid &= np.isfinite(spectrum) & (np.abs(spectrum) > 0.0)
    common_indices = base._largest_true_run(valid)
    if len(common_indices) < 32:
        raise ValueError(
            f"Only {len(common_indices)} contiguous bins are valid for all "
            f"{1 + len(candidate_names)} waveforms"
        )

    common_mf = mf_full[common_indices]
    common_frequency_hz = frequency_hz_full[common_indices]
    spectra = {
        name: np.asarray(values[common_indices], dtype=np.complex128)
        for name, values in spectra_full.items()
    }

    mismatch_results: dict[str, dict[str, float]] = {}
    for name in candidate_names:
        match = base._maximize_time_phase_match(
            spectra[REFERENCE_NAME],
            spectra[name],
            common_mf,
            preferred_time_shift_M=None,
        )
        match["time_shift_seconds"] = match["time_shift_M"] * mass_seconds
        mismatch_results[name] = match

    return {
        "sxs_id": str(sxs_id),
        "reference": REFERENCE_NAME,
        "start_taper_M": start_taper_M,
        "parameters": parameters,
        "mismatches": mismatch_results,
        "conditioning": {
            "reference_start_taper_M": base.TRAINING_WINDOW_ON_M,
            "nrsur_start_taper_M": start_taper_M if nrsur is not None else None,
            "SXS": {
                "measured_Mf_start": sxs["mf_start"],
                "trust_Mf_min": sxs["mf_min"],
                "trust_Mf_max": sxs["mf_max"],
                "window": sxs["window"],
            },
            "NRSur7dq4": {
                "measured_Mf_start": nrsur["mf_start"],
                "trust_Mf_min": nrsur["mf_min"],
                "trust_Mf_max": nrsur["mf_max"],
                "window": nrsur["window"],
            } if nrsur is not None else None,
            "common_Mf_min": float(common_mf[0]),
            "common_Mf_max": float(common_mf[-1]),
            "common_f_min_hz": float(common_frequency_hz[0]),
            "common_f_max_hz": float(common_frequency_hz[-1]),
            "frequency_samples": int(len(common_mf)),
            "delta_Mf": delta_mf,
            "delta_f_hz": delta_f_hz,
            "fft_samples": int(n_fft),
            "imrphenomd_support": phenom_support,
            "frequency_weighting": "flat PSD (uniform df)",
            "pn_backbone_source": (
                "Newtonian amplitude and TaylorF2 phase evaluated directly "
                "through KAN_utilities/KAN_Waveform.py"
            ) if PN_BACKBONE_NAME in candidate_names else None,
            "transform_policy": {
                "SXS_and_NRSur7dq4": (
                    "one time-to-frequency transform" if nrsur is not None
                    else "SXS only: one time-to-frequency transform"
                ),
                "frequency_domain_models": (
                    ", ".join(name for name in candidate_names if name != "NRSur7dq4")
                    + " evaluated directly in frequency space"
                ),
            },
        },
        "frequency_Mf": common_mf,
        "frequency_hz": common_frequency_hz,
        "spectra": spectra,
        "numerical_scaling": {
            "total_mass_msun": base.FIDUCIAL_TOTAL_MASS_MSUN,
            "distance_mpc": base.FIDUCIAL_DISTANCE_MPC,
        },
    }


def summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-serializable part of a computation result."""

    return {
        "sxs_id": result["sxs_id"],
        "reference": result["reference"],
        "start_taper_M": result["start_taper_M"],
        "parameters": result["parameters"],
        "mismatches": result["mismatches"],
        "conditioning": result["conditioning"],
        "numerical_scaling": result["numerical_scaling"],
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sxs_id", help="for example SXS:BBH:0198")
    parser.add_argument(
        "start_taper",
        type=_parse_start_taper,
        help="NRSur7dq4 Planck turn-on: 500M, 600M, or 700M",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help='Models to compare: KAN IMRPhenomD NRSur7dq4 "PN Backbone" (default: all)',
    )
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = compute_mismatches(
        args.sxs_id,
        args.start_taper,
        models=args.models,
        download=not args.no_download,
        progress=not args.no_progress,
    )
    text = json.dumps(summary(result), indent=2)
    print(text)
    if args.json_path is not None:
        destination = args.json_path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
