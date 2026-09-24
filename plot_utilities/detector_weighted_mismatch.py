"""Advanced-LIGO-design weighted waveform mismatches versus total mass.

The two required inputs are an SXS ID and detector-frame total mass in solar
masses.  The reference is the training-conditioned SXS waveform.  NRSur7dq4
uses the same 600 M start taper and adaptive end taper, while KAN and
IMRPhenomD are evaluated directly in frequency space by
``consistent_conditioning_mismatch.py``.

The overlap uses PyCBC's ``aLIGOZeroDetHighPower`` design PSD, starts at
20 Hz, is restricted to the common trustworthy support of every waveform,
and is maximized over relative time and constant phase.  Total mass changes
the map f=(Mf)/M and hence the detector weighting; it does not require
regenerating the intrinsic dimensionless waveforms. The same module also provides
the interpolation-test catalogue and mismatch-versus-mass plotting helpers that
were previously kept in a separate wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.optimize import minimize_scalar


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
BBH_ID_PATH = PROJECT_DIRECTORY / "BBH_ID.text"

from . import _mismatch_support as support
from . import consistent_conditioning_mismatch as consistent


PSD_NAME = "PyCBC aLIGOZeroDetHighPower"
LOW_FREQUENCY_HZ = 20.0
NRSUR_START_TAPER_M = 600.0
CANDIDATE_NAMES = ("KAN", "NRSur7dq4", "IMRPhenomD")

DEFAULT_MASSES_MSUN = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)
MODEL_ORDER = ("KAN", "NRSur7dq4", "IMRPhenomD")
MODEL_STYLE = {
    "KAN": {
        "color": "#000000",
        "label": "KAN",
        "marker": "o",
        "linestyle": "-",
        "markersize": 6.5,
    },
    "NRSur7dq4": {
        "color": "#0072B2",
        "label": "NRSur7dq4",
        "marker": "s",
        "linestyle": "-.",
        "markersize": 6.5,
    },
    "IMRPhenomD": {
        "color": "#D55E00",
        "label": "IMRPhenomD",
        "marker": "^",
        "linestyle": ":",
        "markersize": 7.5,
    },
}


def _load_bbh_id_section(
    section_name: str,
    path: str | Path = BBH_ID_PATH,
) -> tuple[str, ...]:
    """Read one SXS-ID section from the central ``BBH_ID.text`` file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Could not find the central BBH ID file at {source}")

    ids: list[str] = []
    in_section = False
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            if in_section:
                break
            in_section = header.startswith(section_name)
            continue
        if in_section and line.startswith("SXS:BBH:"):
            ids.append(line)

    if not ids:
        raise ValueError(
            f"No SXS IDs were found in the [{section_name}] section of {source}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate SXS IDs found in [{section_name}] of {source}")
    return tuple(ids)


def load_interpolation_test_ids(
    path: str | Path = BBH_ID_PATH,
) -> tuple[str, ...]:
    """Return the interpolation-test SXS IDs from ``BBH_ID.text``."""

    return _load_bbh_id_section("INTERPOLATION TEST", path)


INTERPOLATION_TEST_IDS = load_interpolation_test_ids()


def _validate_total_mass(total_mass_msun: float) -> float:
    mass = float(total_mass_msun)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("total_mass_msun must be a finite positive number")
    return mass


def _design_psd_on_grid(
    frequency_hz: np.ndarray,
    delta_f_hz: float,
) -> np.ndarray:
    """Evaluate PyCBC's aLIGO design PSD at grid-aligned frequencies."""

    try:
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError as exc:
        raise ImportError(
            "PyCBC is required; install requirements-waveform-comparison.txt."
        ) from exc

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    indices = np.rint(frequency_hz / delta_f_hz).astype(int)
    if np.any(indices < 0) or not np.allclose(
        frequency_hz,
        indices * delta_f_hz,
        rtol=1.0e-9,
        atol=max(1.0e-12, 1.0e-9 * delta_f_hz),
    ):
        raise ValueError("Detector frequencies are not aligned to delta_f")

    # PyCBC reserves the last sample as the Nyquist bin and sets it to zero.
    # Add one unused bin so the largest requested frequency remains physical.
    psd_full = aLIGOZeroDetHighPower(
        int(indices[-1]) + 2,
        float(delta_f_hz),
        LOW_FREQUENCY_HZ,
    )
    return np.asarray(psd_full, dtype=float)[indices]


def _maximize_weighted_match(
    reference: np.ndarray,
    candidate: np.ndarray,
    frequency_hz: np.ndarray,
    psd: np.ndarray,
) -> dict[str, float]:
    """PSD-weighted normalized match maximized over time and phase."""

    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if not (
        len(reference)
        == len(candidate)
        == len(frequency_hz)
        == len(psd)
    ):
        raise ValueError("Waveforms, frequencies, and PSD must have equal size")
    if len(frequency_hz) < 32:
        raise ValueError("At least 32 detector-weighted bins are required")
    delta_f_hz = float(np.median(np.diff(frequency_hz)))
    if not np.allclose(
        np.diff(frequency_hz),
        delta_f_hz,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError("Time maximization requires a uniform frequency grid")
    if np.any(~np.isfinite(psd)) or np.any(psd <= 0.0):
        raise ValueError("The selected detector band contains an invalid PSD")

    inverse_psd = 1.0 / psd
    reference_norm = np.sqrt(
        np.sum(np.abs(reference) ** 2 * inverse_psd) * delta_f_hz
    )
    candidate_norm = np.sqrt(
        np.sum(np.abs(candidate) ** 2 * inverse_psd) * delta_f_hz
    )
    if reference_norm <= 0.0 or candidate_norm <= 0.0:
        raise ValueError("Cannot match a zero-norm waveform")
    cross = reference * np.conj(candidate) * inverse_psd * delta_f_hz

    def correlation(time_shift_seconds: float) -> complex:
        return np.sum(
            cross
            * np.exp(
                2j * np.pi * frequency_hz * float(time_shift_seconds)
            )
        )

    period_seconds = 1.0 / delta_f_hz
    correlation_size = support._next_power_of_two(16 * len(cross))
    sampled = correlation_size * np.fft.ifft(cross, n=correlation_size)
    best_index = int(np.argmax(np.abs(sampled)))
    spacing_seconds = period_seconds / correlation_size
    centre_seconds = best_index * spacing_seconds
    if centre_seconds >= 0.5 * period_seconds:
        centre_seconds -= period_seconds
    refined = minimize_scalar(
        lambda shift: -abs(correlation(float(shift))),
        bounds=(
            centre_seconds - spacing_seconds,
            centre_seconds + spacing_seconds,
        ),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    time_shift_seconds = float(refined.x)
    best_correlation = correlation(time_shift_seconds)
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
        "time_shift_seconds": time_shift_seconds,
        "phase_shift_radians": float(np.angle(best_correlation)),
        "correlation_period_seconds": period_seconds,
    }


def _prepare_intrinsic_waveforms(
    sxs_id: str,
    *,
    download: bool,
    progress: bool,
) -> dict[str, Any]:
    """Prepare one common dimensionless spectrum for all requested masses."""

    return consistent.compute_mismatches(
        sxs_id,
        NRSUR_START_TAPER_M,
        include_pn_backbone=False,
        download=download,
        progress=progress,
    )


def _weight_prepared_waveforms(
    prepared: dict[str, Any],
    total_mass_msun: float,
) -> dict[str, Any]:
    total_mass_msun = _validate_total_mass(total_mass_msun)
    mass_seconds = total_mass_msun * support.MTSUN_SI
    mf = np.asarray(prepared["frequency_Mf"], dtype=float)
    frequency_hz = mf / mass_seconds
    delta_mf = float(np.median(np.diff(mf)))
    delta_f_hz = delta_mf / mass_seconds
    psd = _design_psd_on_grid(frequency_hz, delta_f_hz)

    valid = (
        (frequency_hz >= LOW_FREQUENCY_HZ)
        & np.isfinite(psd)
        & (psd > 0.0)
    )
    for spectrum in prepared["spectra"].values():
        valid &= np.isfinite(spectrum) & (np.abs(spectrum) > 0.0)
    detector_indices = support._largest_true_run(valid)
    if len(detector_indices) < 32:
        raise ValueError(
            f"Only {len(detector_indices)} bins remain above "
            f"{LOW_FREQUENCY_HZ:g} Hz for M={total_mass_msun:g} Msun"
        )

    detector_mf = mf[detector_indices]
    detector_frequency_hz = frequency_hz[detector_indices]
    detector_psd = psd[detector_indices]
    spectra = {
        name: np.asarray(values[detector_indices], dtype=np.complex128)
        for name, values in prepared["spectra"].items()
    }

    reference_name = prepared["reference"]
    mismatch_results: dict[str, dict[str, float]] = {}
    for name in CANDIDATE_NAMES:
        result = _maximize_weighted_match(
            spectra[reference_name],
            spectra[name],
            detector_frequency_hz,
            detector_psd,
        )
        result["time_shift_M"] = result["time_shift_seconds"] / mass_seconds
        mismatch_results[name] = result

    return {
        "sxs_id": prepared["sxs_id"],
        "reference": reference_name,
        "total_mass_msun": total_mass_msun,
        "mismatches": mismatch_results,
        "parameters": prepared["parameters"],
        "detector_weighting": {
            "psd": PSD_NAME,
            "low_frequency_hz": LOW_FREQUENCY_HZ,
            "effective_f_min_hz": float(detector_frequency_hz[0]),
            "effective_f_max_hz": float(detector_frequency_hz[-1]),
            "effective_Mf_min": float(detector_mf[0]),
            "effective_Mf_max": float(detector_mf[-1]),
            "frequency_samples": int(len(detector_mf)),
            "delta_f_hz": delta_f_hz,
            "nrsur_start_taper_M": NRSUR_START_TAPER_M,
            "common_support_source": (
                "consistent_conditioning_mismatch.py four-waveform "
                "trustworthy intersection"
            ),
            "transform_policy": prepared["conditioning"]["transform_policy"],
        },
        "frequency_Mf": detector_mf,
        "frequency_hz": detector_frequency_hz,
        "psd": detector_psd,
        "spectra": spectra,
    }


def compute_mismatches(
    sxs_id: str,
    total_mass_msun: float,
    *,
    download: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute detector-weighted mismatches for one SXS ID and total mass."""

    prepared = _prepare_intrinsic_waveforms(
        sxs_id,
        download=download,
        progress=progress,
    )
    return _weight_prepared_waveforms(prepared, total_mass_msun)


def compute_mass_series(
    sxs_id: str,
    total_masses_msun: Iterable[float],
    *,
    download: bool = True,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Compute several masses while preparing the SXS case only once."""

    masses = [_validate_total_mass(value) for value in total_masses_msun]
    if not masses:
        raise ValueError("total_masses_msun must contain at least one mass")
    prepared = _prepare_intrinsic_waveforms(
        sxs_id,
        download=download,
        progress=progress,
    )
    return [_weight_prepared_waveforms(prepared, mass) for mass in masses]


def compute_interpolation_catalogue(
    sxs_ids: Iterable[str] | None = None,
    masses_msun: Iterable[float] = DEFAULT_MASSES_MSUN,
    *,
    download: bool = True,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compute detector-weighted results for the interpolation-test catalogue.

    The default SXS IDs are read from ``BBH_ID.text``. Systems outside the
    strict NRSur7dq4 calibration domain are recorded in ``skipped`` and are not
    used in the three-model common-subset mass comparison.
    """

    selected_ids = (
        load_interpolation_test_ids() if sxs_ids is None else tuple(sxs_ids)
    )
    masses = tuple(_validate_total_mass(value) for value in masses_msun)
    if not selected_ids:
        raise ValueError("sxs_ids must contain at least one SXS ID")
    if not masses:
        raise ValueError("masses_msun must contain at least one mass")

    rows: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    for index, sxs_id in enumerate(selected_ids, start=1):
        if progress:
            print(f"[{index}/{len(selected_ids)}] {sxs_id}")
        try:
            mass_results = compute_mass_series(
                sxs_id,
                masses,
                download=download,
                progress=False,
            )
        except ValueError as exc:
            message = str(exc)
            if "strict NRSur7dq4 calibration" not in message:
                raise
            skipped[str(sxs_id)] = message
            if progress:
                print(f"  skipped from NRSur-common catalogue because {message}")
            continue

        for result in mass_results:
            for model, values in result["mismatches"].items():
                rows.append(
                    {
                        "sxs_id": str(sxs_id),
                        "total_mass_msun": float(result["total_mass_msun"]),
                        "model": model,
                        "mismatch": float(values["mismatch"]),
                        "match": float(values["match"]),
                        "effective_f_min_hz": float(
                            result["detector_weighting"]["effective_f_min_hz"]
                        ),
                        "effective_f_max_hz": float(
                            result["detector_weighting"]["effective_f_max_hz"]
                        ),
                    }
                )

    if not rows:
        raise RuntimeError("No SXS systems produced detector-weighted results")
    return pd.DataFrame(rows), skipped


def summarize_mismatch_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return median and 90th percentile for every model and total mass."""

    required = {"model", "total_mass_msun", "mismatch", "sxs_id"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results table is missing columns {sorted(missing)}")
    return (
        results.groupby(["model", "total_mass_msun"], as_index=False)
        .agg(
            median=("mismatch", "median"),
            p90=("mismatch", lambda values: np.quantile(values, 0.9)),
            systems=("sxs_id", "nunique"),
        )
        .sort_values(["model", "total_mass_msun"])
        .reset_index(drop=True)
    )


def plot_mismatch_vs_mass(
    results: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot median mismatch and the median-to-p90 band versus total mass."""

    table = summarize_mismatch_table(results)
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    for model in MODEL_ORDER:
        values = table.loc[table["model"] == model].sort_values(
            "total_mass_msun"
        )
        if values.empty:
            continue
        masses = values["total_mass_msun"].to_numpy(dtype=float)
        median = values["median"].to_numpy(dtype=float)
        p90 = values["p90"].to_numpy(dtype=float)
        style = MODEL_STYLE[model]
        axis.fill_between(
            masses,
            median,
            p90,
            color=style["color"],
            alpha=0.16,
            linewidth=0.0,
        )
        axis.plot(
            masses,
            median,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
            label=style["label"],
            zorder=3,
        )

    axis.axhline(1.0e-3, color="#666666", linestyle=":", linewidth=1.4)
    axis.set_yscale("log")
    axis.set_xlabel(
        r"detector-frame total mass $M\,[M_\odot]$",
        fontsize=14,
        fontweight="normal",
        color="black",
        labelpad=7,
    )
    axis.set_ylabel(
        "detector-weighted mismatch",
        fontsize=14,
        fontweight="normal",
        color="black",
        labelpad=7,
    )
    axis.set_xticks(np.sort(results["total_mass_msun"].unique().astype(float)))
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=12,
        colors="black",
        width=1.3,
        length=6,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        colors="black",
        width=1.0,
        length=3.5,
    )
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.3)
    axis.grid(True, which="both", alpha=0.25, linestyle=":")
    legend = axis.legend(
        loc="upper left",
        frameon=True,
        prop={"size": 12, "weight": "normal"},
        handlelength=1.8,
        handletextpad=0.7,
        borderpad=0.6,
        labelspacing=0.5,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(1.1)
    legend.get_frame().set_alpha(1.0)

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            figure.savefig(destination, dpi=300)
        print(f"Saved {destination}")

    return figure, axis, table


def summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-serializable portion of one mass result."""

    return {
        "sxs_id": result["sxs_id"],
        "reference": result["reference"],
        "total_mass_msun": result["total_mass_msun"],
        "parameters": result["parameters"],
        "mismatches": result["mismatches"],
        "detector_weighting": result["detector_weighting"],
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sxs_id",
        nargs="?",
        help="single-system mode, for example SXS:BBH:0198",
    )
    parser.add_argument(
        "total_mass_msun",
        nargs="?",
        type=float,
        help="single-system detector-frame total mass, for example 60",
    )
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help=(
            "run the interpolation-test catalogue from BBH_ID.text and create "
            "the detector-weighted mismatch-versus-mass summary"
        ),
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=list(DEFAULT_MASSES_MSUN),
        help="total masses used with --catalogue",
    )
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/detector_weighted_mismatch_vs_mass.pdf"),
        help="figure path used with --catalogue",
    )
    parser.add_argument(
        "--catalogue-csv",
        type=Path,
        default=Path("detector_weighted_catalogue.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("detector_weighted_summary.csv"),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)

    if args.catalogue:
        if args.sxs_id is not None or args.total_mass_msun is not None:
            raise ValueError(
                "Do not provide positional sxs_id/total_mass_msun with --catalogue"
            )
        catalogue, excluded = compute_interpolation_catalogue(
            masses_msun=args.masses,
            download=not args.no_download,
            progress=not args.no_progress,
        )
        _, _, summary_table = plot_mismatch_vs_mass(
            catalogue,
            output_path=args.output,
        )

        args.catalogue_csv.parent.mkdir(parents=True, exist_ok=True)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        catalogue.to_csv(args.catalogue_csv, index=False)
        summary_table.to_csv(args.summary_csv, index=False)
        print(summary_table.to_string(index=False))
        if excluded:
            print("Excluded outside strict NRSur7dq4 calibration")
            for sxs_id, reason in excluded.items():
                print(f"  {sxs_id}  {reason}")
        return 0

    if args.sxs_id is None or args.total_mass_msun is None:
        raise ValueError(
            "Provide sxs_id and total_mass_msun, or use --catalogue"
        )

    result = compute_mismatches(
        args.sxs_id,
        args.total_mass_msun,
        download=not args.no_download,
        progress=not args.no_progress,
    )
    output_text = json.dumps(summary(result), indent=2)
    print(output_text)
    if args.json_path is not None:
        destination = args.json_path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output_text + "\n", encoding="utf-8")
        print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
