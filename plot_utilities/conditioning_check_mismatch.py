"""Repeat the waveform mismatches using minimally conditioned raw SXS.

The required inputs are an SXS ID and an NRSur7dq4 Planck turn-on duration,
chosen from 500 M, 600 M, or 700 M.  The reference is the full raw SXS (2, 2)
record, uniformly resampled at 0.5 M and given only a short sine-squared taper
on the first and last 0.5 percent of its samples.

The comparison deliberately retains the trustworthy frequency-band rule used
by ``consistent_conditioning_mismatch.py``.  Consequently, changing the
reference conditioning is the only substantive difference between the two
calculations.  NRSur7dq4 uses the selected start taper and the training-style
adaptive end taper.  KAN and IMRPhenomD are evaluated directly in frequency
space and are never transformed to time and back.  The
TaylorF2-phase/Newtonian-amplitude PN backbone used by KAN is likewise
evaluated directly in frequency space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Support loading this file by path from a notebook or another working
# directory while keeping all companion imports confined to this folder.
MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from KAN_utilities import KAN_Waveform as KW
from . import consistent_conditioning_mismatch as consistent
from . import _mismatch_support as base


REFERENCE_NAME = "Raw SXS (minimal)"
PN_BACKBONE_NAME = "TaylorF2 PN backbone"
CANDIDATE_NAMES = (
    "KAN",
    "NRSur7dq4",
    "IMRPhenomD",
    PN_BACKBONE_NAME,
)
MINIMAL_EDGE_FRACTION = base.DEFAULT_MINIMAL_EDGE_FRACTION


def compute_mismatches(
    sxs_id: str,
    start_taper_M: float,
    *,
    download: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Return raw-SXS-reference mismatches on the four-way common band."""

    start_taper_M = consistent._validate_start_taper(start_taper_M)

    # Inputs and downloads remain confined to this module directory.
    data_dir = Path(base.DEFAULT_DATA_DIRECTORY).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    base._ensure_nrsur_data(data_dir, download=download)
    simulation = base._load_sxs(
        sxs_id,
        data_dir,
        download=download,
        progress=progress,
    )
    parameters = base._binary_parameters(simulation.metadata)
    consistent._validate_nrsur_calibration(parameters)
    raw = base._extract_sxs_mode(simulation)

    mass_seconds = base.FIDUCIAL_TOTAL_MASS_MSUN * base.MTSUN_SI
    distance_seconds = base.FIDUCIAL_DISTANCE_MPC * base.MPC_SI / base.C_SI
    delta_t = base.TRAINING_DTAU * mass_seconds
    physical_strain_scale = (
        mass_seconds / distance_seconds
    ) * base.Y22_FACEON
    geometric_frequency_scale = mass_seconds**2 / distance_seconds

    raw_sxs = base._minimal_sxs(
        raw,
        physical_strain_scale,
        MINIMAL_EDGE_FRACTION,
    )
    nrsur = consistent._condition_nrsur(
        parameters,
        delta_t,
        start_taper_M,
    )

    # This anchor is not used as a reference waveform.  It supplies exactly
    # the same SXS trust boundaries as the primary calculation so that the
    # conditioning check is performed over the same physical band.
    support_anchor = consistent._condition_sxs_reference(
        raw,
        parameters,
        physical_strain_scale,
    )

    domain = KW.model_domain()
    preliminary_mf_min = max(
        support_anchor["mf_min"],
        nrsur["mf_min"],
        float(domain["Mf"][0]),
    )
    preliminary_mf_max = min(
        support_anchor["mf_max"],
        nrsur["mf_max"],
        float(domain["Mf"][1]),
        np.nextafter(0.5 / base.TRAINING_DTAU, 0.0),
    )
    if preliminary_mf_max <= preliminary_mf_min:
        raise ValueError(
            f"Empty four-way common band Mf={preliminary_mf_min:.7g}--"
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
            base.TRAINING_PAD_FACTOR * len(support_anchor["strain"]),
            base.TRAINING_PAD_FACTOR * len(nrsur["strain"]),
            len(raw_sxs["strain"]),
            1.2 * duration_samples,
        )
    )
    mf_full, negative_indices = base._positive_frequency_grid(n_fft)
    frequency_hz_full = mf_full / mass_seconds
    delta_mf = float(mf_full[1] - mf_full[0])
    delta_f_hz = delta_mf / mass_seconds

    spectra_full: dict[str, np.ndarray] = {
        REFERENCE_NAME: base._time_to_positive_spectrum(
            raw_sxs["strain"], delta_t, n_fft, negative_indices
        ),
        "NRSur7dq4": base._time_to_positive_spectrum(
            nrsur["strain"], delta_t, n_fft, negative_indices
        ),
    }

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
    evaluator = KW.KAN
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
            "five waveforms"
        )

    common_mf = mf_full[common_indices]
    common_frequency_hz = frequency_hz_full[common_indices]
    spectra = {
        name: np.asarray(values[common_indices], dtype=np.complex128)
        for name, values in spectra_full.items()
    }

    mismatch_results: dict[str, dict[str, float]] = {}
    for name in CANDIDATE_NAMES:
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
            "reference_policy": (
                "full raw SXS record resampled at 0.5M with only short "
                "sine-squared endpoint tapers"
            ),
            "comparison_band_policy": (
                "same trust-boundary prescription as "
                "consistent_conditioning_mismatch.py"
            ),
            "minimal_endpoint_fraction_each_side": MINIMAL_EDGE_FRACTION,
            "minimal_endpoint_samples_each_side": int(
                raw_sxs["edge_samples"]
            ),
            "minimal_endpoint_duration_M": float(
                raw_sxs["edge_duration_M"]
            ),
            "sxs_support_anchor": {
                "measured_Mf_start": support_anchor["mf_start"],
                "trust_Mf_min": support_anchor["mf_min"],
                "trust_Mf_max": support_anchor["mf_max"],
            },
            "nrsur_start_taper_M": start_taper_M,
            "NRSur7dq4": {
                "measured_Mf_start": nrsur["mf_start"],
                "trust_Mf_min": nrsur["mf_min"],
                "trust_Mf_max": nrsur["mf_max"],
                "window": nrsur["window"],
            },
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
            ),
            "transform_policy": {
                "raw_SXS_and_NRSur7dq4": "one time-to-frequency transform",
                "KAN_IMRPhenomD_and_PN_backbone": (
                    "direct frequency evaluation only"
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
        type=consistent._parse_start_taper,
        help="NRSur7dq4 Planck turn-on: 500M, 600M, or 700M",
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
