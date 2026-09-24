"""Detector-weighted mismatch plot for the 30 stress-test simulations.

The stress-set SXS identifiers are read from the central ``BBH_ID.text`` file. The
reference is the training-conditioned SXS waveform.  KAN and IMRPhenomD are evaluated directly
in frequency space.  NRSur7dq4 is included only when the SXS parameters lie in
its strict q<=4 and |chi|<=0.8 calibration region; it is never extrapolated for
the high-q stress cases.  Each overlap uses PyCBC's Advanced-LIGO zero-detuned
high-power design PSD and is maximized over time and constant phase on the
common trustworthy interval of the waveforms actually valid for that system.

The plot consists of mass ratio on the horizontal
axis, detector-weighted mismatch on a logarithmic vertical axis, one marker
shape per approximant, and point colour set by the reduced PN spin chi_PN.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from KAN_utilities import KAN_Waveform as KW
from . import _mismatch_support as support
from . import consistent_conditioning_mismatch as consistent
from . import detector_weighted_mismatch as detector


# Central source of truth for the frozen SXS split.
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
BBH_ID_PATH = PROJECT_DIRECTORY / "BBH_ID.text"
STRESS_SECTION = "STRESS / SPARSE-COVERAGE"


def load_stress_sxs_ids(path: str | Path = BBH_ID_PATH) -> tuple[str, ...]:
    """Read the stress-set SXS IDs from the central ``BBH_ID.text`` file.

    The parser reads only the ``[STRESS / SPARSE-COVERAGE ...]`` section and
    stops at the next bracketed section header.  This keeps dataset membership
    in one place and prevents plotting utilities from carrying duplicate splits.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Could not find the central BBH ID file at {source}"
        )

    ids: list[str] = []
    in_stress_section = False
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            if in_stress_section:
                break
            in_stress_section = header.startswith(STRESS_SECTION)
            continue

        if in_stress_section and line.startswith("SXS:BBH:"):
            ids.append(line)

    if not ids:
        raise ValueError(
            f"No SXS IDs were found in the [{STRESS_SECTION}] section of {source}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate stress-set SXS IDs found in {source}")

    return tuple(ids)


STRESS_SXS_IDS = load_stress_sxs_ids()

APPROXIMANTS: tuple[str, ...] = (
    "KAN",
    "NRSur7dq4",
    "IMRPhenomD",
)


def _selected_stress_ids(sxs_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a requested subset and preserve the run ordering."""

    if sxs_ids is None:
        return STRESS_SXS_IDS
    requested = tuple(dict.fromkeys(str(value) for value in sxs_ids))
    if not requested:
        raise ValueError("sxs_ids must contain at least one stress-system ID")
    unknown = [value for value in requested if value not in STRESS_SXS_IDS]
    if unknown:
        raise ValueError(
            "Only the stress systems listed in BBH_ID.text are allowed; "
            f"unknown ID(s): {', '.join(unknown)}"
        )
    return requested


def reduced_pn_spin(parameters: dict[str, float]) -> float:
    r"""Return chi_PN = chi_eff - 38 eta (chi1z + chi2z) / 113."""

    mass1_fraction = float(parameters["mass1_fraction"])
    mass2_fraction = float(parameters["mass2_fraction"])
    eta = float(parameters["symmetric_mass_ratio"])
    chi1z = float(parameters["chi1z"])
    chi2z = float(parameters["chi2z"])
    chi_eff = mass1_fraction * chi1z + mass2_fraction * chi2z
    return float(chi_eff - (38.0 * eta / 113.0) * (chi1z + chi2z))


def _stress_binary_parameters(metadata: Any) -> dict[str, float]:
    """Read aligned-spin SXS parameters without imposing NRSur's q limit."""

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
    if min(mass1, mass2) <= 0.0:
        raise ValueError("SXS metadata contains a non-positive mass")

    total = mass1 + mass2
    fraction1 = mass1 / total
    fraction2 = mass2 / total
    chi1z = float(spin1[2])
    chi2z = float(spin2[2])
    if max(abs(chi1z), abs(chi2z)) > 1.001:
        raise ValueError("SXS metadata contains an unphysical aligned spin")
    return {
        "mass1_fraction": float(fraction1),
        "mass2_fraction": float(fraction2),
        "mass_ratio": float(fraction1 / fraction2),
        "symmetric_mass_ratio": float(fraction1 * fraction2),
        "chi1z": chi1z,
        "chi2z": chi2z,
    }


def _nrsur_validity(parameters: dict[str, float]) -> tuple[bool, str]:
    """Return strict NRSur7dq4 validity without enabling extrapolation."""

    q = float(parameters["mass_ratio"])
    spin = max(abs(parameters["chi1z"]), abs(parameters["chi2z"]))
    if q > 4.0:
        return False, f"q={q:.8g} exceeds the strict NRSur7dq4 limit q<=4"
    if spin > 0.8:
        return (
            False,
            f"max(|chi1z|,|chi2z|)={spin:.8g} exceeds the strict "
            "NRSur7dq4 limit 0.8",
        )
    return True, "inside strict NRSur7dq4 calibration"


def _prepare_stress_waveforms(
    sxs_id: str,
    *,
    download: bool,
    progress: bool,
) -> dict[str, Any]:
    """Prepare the common intrinsic spectra for one stress simulation."""

    data_directory = Path(support.DEFAULT_DATA_DIRECTORY).resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    simulation = support._load_sxs(
        sxs_id,
        data_directory,
        download=download,
        progress=progress,
    )
    parameters = _stress_binary_parameters(simulation.metadata)
    raw = support._extract_sxs_mode(simulation)

    mass_seconds = support.FIDUCIAL_TOTAL_MASS_MSUN * support.MTSUN_SI
    distance_seconds = (
        support.FIDUCIAL_DISTANCE_MPC * support.MPC_SI / support.C_SI
    )
    delta_t = support.TRAINING_DTAU * mass_seconds
    physical_strain_scale = (
        mass_seconds / distance_seconds
    ) * support.Y22_FACEON

    sxs = consistent._condition_sxs_reference(
        raw,
        parameters,
        physical_strain_scale,
    )

    nrsur_is_valid, nrsur_reason = _nrsur_validity(parameters)
    nrsur = None
    if nrsur_is_valid:
        support._ensure_nrsur_data(data_directory, download=download)
        nrsur = consistent._condition_nrsur(
            parameters,
            delta_t,
            detector.NRSUR_START_TAPER_M,
        )

    domain = KW.model_domain()
    lower_bounds = [sxs["mf_min"], float(domain["Mf"][0])]
    upper_bounds = [
        sxs["mf_max"],
        float(domain["Mf"][1]),
        np.nextafter(0.5 / support.TRAINING_DTAU, 0.0),
    ]
    if nrsur is not None:
        lower_bounds.append(nrsur["mf_min"])
        upper_bounds.append(nrsur["mf_max"])
    preliminary_mf_min = max(lower_bounds)
    preliminary_mf_max = min(upper_bounds)
    if preliminary_mf_max <= preliminary_mf_min:
        raise ValueError(
            f"Empty trustworthy band Mf={preliminary_mf_min:.7g}--"
            f"{preliminary_mf_max:.7g}"
        )

    duration_samples = (
        KW.signal_duration(
            parameters["symmetric_mass_ratio"], preliminary_mf_min
        )
        + 500.0
    ) / support.TRAINING_DTAU
    lengths = [
        support.TRAINING_PAD_FACTOR * len(sxs["strain"]),
        1.2 * duration_samples,
    ]
    if nrsur is not None:
        lengths.append(support.TRAINING_PAD_FACTOR * len(nrsur["strain"]))
    n_fft = support._next_power_of_two(max(lengths))
    mf_full, negative_indices = support._positive_frequency_grid(n_fft)
    frequency_hz_full = mf_full / mass_seconds
    delta_mf = float(mf_full[1] - mf_full[0])
    delta_f_hz = delta_mf / mass_seconds

    spectra_full: dict[str, np.ndarray] = {
        consistent.REFERENCE_NAME: support._time_to_positive_spectrum(
            sxs["strain"], delta_t, n_fft, negative_indices
        )
    }
    if nrsur is not None:
        spectra_full["NRSur7dq4"] = support._time_to_positive_spectrum(
            nrsur["strain"], delta_t, n_fft, negative_indices
        )

    phenom, _ = support._generate_phenomd_frequency(
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
    mass1_msun = 0.5 * support.FIDUCIAL_TOTAL_MASS_MSUN * (
        1.0 + delta_for_kan
    )
    mass2_msun = support.FIDUCIAL_TOTAL_MASS_MSUN - mass1_msun
    kan_values = evaluator(
        frequency_hz_full[preliminary_indices],
        mass1_msun,
        mass2_msun,
        parameters["chi1z"],
        parameters["chi2z"],
        support.FIDUCIAL_DISTANCE_MPC,
        strict_domain=True,
    )
    kan_full = np.full_like(
        mf_full,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    kan_full[preliminary_indices] = kan_values
    spectra_full["KAN"] = kan_full

    valid = preliminary.copy()
    for spectrum in spectra_full.values():
        valid &= np.isfinite(spectrum) & (np.abs(spectrum) > 0.0)
    common_indices = support._largest_true_run(valid)
    if len(common_indices) < 32:
        raise ValueError(
            f"Only {len(common_indices)} contiguous bins are valid for the "
            "stress comparison"
        )

    return {
        "sxs_id": str(sxs_id),
        "reference": consistent.REFERENCE_NAME,
        "parameters": parameters,
        "frequency_Mf": mf_full[common_indices],
        "spectra": {
            name: np.asarray(values[common_indices], dtype=np.complex128)
            for name, values in spectra_full.items()
        },
        "nrsur_status": {
            "included": bool(nrsur is not None),
            "reason": nrsur_reason,
        },
    }


def _weight_stress_waveforms(
    prepared: dict[str, Any],
    total_mass_msun: float,
) -> dict[str, Any]:
    """Apply the aLIGO design PSD to one prepared stress waveform set."""

    mass = detector._validate_total_mass(total_mass_msun)
    mass_seconds = mass * support.MTSUN_SI
    mf = np.asarray(prepared["frequency_Mf"], dtype=float)
    frequency_hz = mf / mass_seconds
    delta_mf = float(np.median(np.diff(mf)))
    delta_f_hz = delta_mf / mass_seconds
    psd = detector._design_psd_on_grid(frequency_hz, delta_f_hz)

    valid = (
        (frequency_hz >= detector.LOW_FREQUENCY_HZ)
        & np.isfinite(psd)
        & (psd > 0.0)
    )
    for spectrum in prepared["spectra"].values():
        valid &= np.isfinite(spectrum) & (np.abs(spectrum) > 0.0)
    indices = support._largest_true_run(valid)
    if len(indices) < 32:
        raise ValueError(
            f"Only {len(indices)} bins remain above "
            f"{detector.LOW_FREQUENCY_HZ:g} Hz for M={mass:g} Msun"
        )

    spectra = {
        name: np.asarray(values[indices], dtype=np.complex128)
        for name, values in prepared["spectra"].items()
    }
    frequencies = frequency_hz[indices]
    detector_psd = psd[indices]
    mismatches: dict[str, dict[str, float] | None] = {}
    for name in APPROXIMANTS:
        if name not in spectra:
            mismatches[name] = None
            continue
        result = detector._maximize_weighted_match(
            spectra[prepared["reference"]],
            spectra[name],
            frequencies,
            detector_psd,
        )
        result["time_shift_M"] = result["time_shift_seconds"] / mass_seconds
        mismatches[name] = result

    return {
        "sxs_id": prepared["sxs_id"],
        "reference": prepared["reference"],
        "total_mass_msun": mass,
        "parameters": prepared["parameters"],
        "mismatches": mismatches,
        "nrsur_status": prepared["nrsur_status"],
        "detector_weighting": {
            "effective_f_min_hz": float(frequencies[0]),
            "effective_f_max_hz": float(frequencies[-1]),
            "frequency_samples": int(len(indices)),
        },
    }


def _compute_stress_mismatches(
    sxs_id: str,
    total_mass_msun: float,
    *,
    download: bool,
    progress: bool,
) -> dict[str, Any]:
    prepared = _prepare_stress_waveforms(
        sxs_id,
        download=download,
        progress=progress,
    )
    return _weight_stress_waveforms(prepared, total_mass_msun)


def summarize_mismatches(catalogue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-model median and 90th percentile over finite available results.

    Each model uses its own available systems, so NRSur7dq4 may have a smaller
    sample. Missing and nonfinite values are excluded; zero mismatches count.
    Empty samples return None for both statistics (JSON null).
    """
    records = catalogue["records"]
    statistics = {}
    for name in APPROXIMANTS:
        values = np.asarray([
            record["mismatches"].get(name)
            for record in records
            if record["mismatches"].get(name) is not None
        ], dtype=float)
        values = values[np.isfinite(values)]
        statistics[name] = {
            "count": int(values.size),
            "excluded_count": len(records) - int(values.size),
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None,
        }
    return statistics


def print_mismatch_summary(catalogue: dict[str, Any]) -> None:
    """Print mismatch statistics, also usable on a previously saved catalogue."""
    statistics = summarize_mismatches(catalogue)
    print("\nDetector-weighted mismatch statistics (finite available systems per model)")
    print(f"{'Model':<14} {'N':>5} {'Excluded':>9} {'Median':>14} {'90th percentile':>17}")
    for name, row in statistics.items():
        median = "N/A" if row["median"] is None else f"{row['median']:.6e}"
        p90 = "N/A" if row["p90"] is None else f"{row['p90']:.6e}"
        print(f"{name:<14} {row['count']:>5} {row['excluded_count']:>9} {median:>14} {p90:>17}")
    if catalogue.get("failures"):
        print(f"Failed systems omitted from all statistics: {len(catalogue['failures'])}")


def compute_stress_catalogue(
    total_mass_msun: float = 60.0,
    *,
    sxs_ids: Sequence[str] | None = None,
    download: bool = True,
    progress: bool = True,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Compute detector-weighted mismatches for the stress set or a subset.

    By default any failed system stops the run, so a publication plot cannot
    silently omit a stress case.  Set ``continue_on_error=True`` only for a
    diagnostic run; failures are then returned in ``catalogue["failures"]``.
    Prints per-model median and 90th percentile and stores them in
    ``catalogue["statistics"]``, using each model's finite available results.
    """

    selected_ids = _selected_stress_ids(sxs_ids)
    mass = detector._validate_total_mass(total_mass_msun)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sxs_id in enumerate(selected_ids, start=1):
        if progress:
            print(f"Stress [{index:02d}/{len(selected_ids):02d}] {sxs_id}")
        try:
            result = _compute_stress_mismatches(
                sxs_id,
                mass,
                download=download,
                progress=progress,
            )
        except Exception as exc:
            if not continue_on_error:
                raise RuntimeError(
                    f"Stress calculation failed for {sxs_id}; no system was "
                    "silently omitted."
                ) from exc
            failures.append(
                {
                    "sxs_id": sxs_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        parameters = result["parameters"]
        records.append(
            {
                "sxs_id": sxs_id,
                "mass_ratio": float(parameters["mass_ratio"]),
                "chi1z": float(parameters["chi1z"]),
                "chi2z": float(parameters["chi2z"]),
                "chi_pn": reduced_pn_spin(parameters),
                "mismatches": {
                    name: (
                        None
                        if result["mismatches"][name] is None
                        else float(result["mismatches"][name]["mismatch"])
                    )
                    for name in APPROXIMANTS
                },
                "nrsur_status": result["nrsur_status"],
                "effective_f_min_hz": float(
                    result["detector_weighting"]["effective_f_min_hz"]
                ),
                "effective_f_max_hz": float(
                    result["detector_weighting"]["effective_f_max_hz"]
                ),
                "frequency_samples": int(
                    result["detector_weighting"]["frequency_samples"]
                ),
            }
        )

    if not records:
        raise RuntimeError("No stress-system mismatch was computed")
    catalogue = {
        "stress_sxs_ids": list(selected_ids),
        "total_mass_msun": mass,
        "reference": "Training-conditioned SXS",
        "psd": detector.PSD_NAME,
        "low_frequency_hz": detector.LOW_FREQUENCY_HZ,
        "approximants": list(APPROXIMANTS),
        "records": records,
        "failures": failures,
    }
    catalogue["statistics"] = summarize_mismatches(catalogue)
    print_mismatch_summary(catalogue)
    return catalogue



def save_catalogue(catalogue: dict[str, Any], path: str | Path) -> Path:
    """Save a computed catalogue without waveform arrays."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(catalogue, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_catalogue(path: str | Path) -> dict[str, Any]:
    """Load a catalogue previously written by :func:`save_catalogue`."""

    source = Path(path).expanduser().resolve()
    catalogue = json.loads(source.read_text(encoding="utf-8"))
    if not catalogue.get("records"):
        raise ValueError(f"Catalogue has no records: {source}")
    return catalogue


def plot_stress_mismatches(
    catalogue: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Plot q versus mismatch with chi_PN fills and model-specific outlines.

    Small, fixed horizontal offsets are used only to keep markers belonging to
    the same physical mass ratio from covering one another.  The catalogue and
    every mismatch calculation retain the unmodified mass ratios.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    records = catalogue.get("records", [])
    if not records:
        raise ValueError("catalogue contains no stress-system records")

    mass_ratio = np.asarray(
        [record["mass_ratio"] for record in records], dtype=float
    )
    chi_pn = np.asarray([record["chi_pn"] for record in records], dtype=float)
    mismatch = {
        name: np.asarray(
            [
                np.nan
                if record["mismatches"][name] is None
                else record["mismatches"][name]
                for record in records
            ],
            dtype=float,
        )
        for name in APPROXIMANTS
    }
    for name, values in mismatch.items():
        available = np.isfinite(values)
        if np.any(values[available] < 0.0):
            raise ValueError(f"{name} contains an invalid mismatch")

    colour_limit = float(np.max(np.abs(chi_pn)))
    if not np.isfinite(colour_limit) or colour_limit == 0.0:
        colour_limit = 1.0
    colour_norm = TwoSlopeNorm(
        vmin=-colour_limit,
        vcenter=0.0,
        vmax=colour_limit,
    )

    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    style_by_model = {
        "KAN": {
            "marker": "o",
            "edgecolor": "#000000",
            "size": 75.0,
            "q_offset": -0.035,
        },
        "NRSur7dq4": {
            "marker": "s",
            "edgecolor": "#0072B2",
            "size": 75.0,
            "q_offset": 0.0,
        },
        "IMRPhenomD": {
            "marker": "^",
            "edgecolor": "#D55E00",
            "size": 95.0,
            "q_offset": 0.035,
        },
    }
    label_by_model = {
        "KAN": f"KAN",
        "NRSur7dq4": "NRSur7dq4",
        "IMRPhenomD": "IMRPhenomD",
    }

    last_scatter = None
    positive_floor = np.finfo(float).tiny
    for name in APPROXIMANTS:
        available = np.isfinite(mismatch[name])
        if not np.any(available):
            continue
        style = style_by_model[name]
        last_scatter = axis.scatter(
            mass_ratio[available] + style["q_offset"],
            np.maximum(mismatch[name][available], positive_floor),
            c=chi_pn[available],
            cmap="RdBu_r",
            norm=colour_norm,
            marker=style["marker"],
            s=style["size"],
            linewidths=1.5,
            edgecolors=style["edgecolor"],
            alpha=0.95,
            label=label_by_model[name],
            zorder=3,
        )

    axis.axhline(1.0e-3, color="black", linestyle=":", linewidth=1.25)
    axis.set_yscale("log")
    axis.set_ylim(6e-6, 6e-2)
    axis.set_xlabel(
        r"mass ratio $q$",
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
    axis.grid(True, which="both", alpha=0.28, linestyle=":")
    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style_by_model[name]["marker"],
            markersize=np.sqrt(style_by_model[name]["size"]),
            markerfacecolor="white",
            markeredgecolor=style_by_model[name]["edgecolor"],
            markeredgewidth=1.5,
            label=label_by_model[name],
        )
        for name in APPROXIMANTS
        if np.any(np.isfinite(mismatch[name]))
    ]
    legend = axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        prop={"size": 12, "weight": "normal"},
        handlelength=1.6,
        handletextpad=0.7,
        borderpad=0.6,
        labelspacing=0.5,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.1)
    legend.get_frame().set_alpha(1.0)

    mass = float(catalogue["total_mass_msun"])
    low_frequency = float(catalogue["low_frequency_hz"])
    #axis.set_title(
    #    rf"aLIGO design PSD, $M={mass:g}\,M_\odot$, "
    #    rf"$f_{{\rm low}}={low_frequency:g}$ Hz, $N={len(records)}$",
    #    fontsize=15,
    #    fontweight="normal",
    #    color="black",
    #    pad=10,
    #)
    if last_scatter is None:
        raise ValueError("No finite mismatch is available to plot")
    colour_bar = fig.colorbar(last_scatter, ax=axis, pad=0.02)
    colour_bar.set_label(
        r"$\chi_{\rm PN}$",
        fontsize=14,
        fontweight="normal",
        color="black",
        labelpad=9,
    )
    colour_bar.ax.tick_params(
        which="major",
        labelsize=12,
        colors="black",
        width=1.2,
        length=5,
    )
    colour_bar.outline.set_edgecolor("black")
    colour_bar.outline.set_linewidth(1.2)

    nrsur_count = int(np.count_nonzero(np.isfinite(mismatch["NRSur7dq4"])))
    #if nrsur_count < len(records):
        #axis.text(
        #    0.99,
        #    0.01,
        #    rf"NRSur7dq4: {nrsur_count}/{len(records)} systems in strict domain",
        #    transform=axis.transAxes,
        #    fontsize=10.5,
        #    fontweight="medium",
        #    color="black",
        #    ha="right",
        #    va="bottom",
        #)

    if catalogue.get("failures"):
        axis.text(
            0.01,
            0.01,
            f"{len(catalogue['failures'])} failed system(s) omitted",
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="medium",
            color="black",
            va="bottom",
        )

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep a fixed 7.2 x 5.2 inch canvas so this PDF has the same physical
        # page size as the mismatch-versus-mass figure.  bbox_inches="tight"
        # is intentionally avoided because content-dependent cropping changes
        # the exported PDF dimensions.
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(destination)
    if show:
        plt.show()
    return fig, axis


def compute_and_plot(
    total_mass_msun: float = 60.0,
    *,
    sxs_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    download: bool = True,
    progress: bool = True,
    continue_on_error: bool = False,
    show: bool = False,
) -> dict[str, Any]:
    """Compute the stress catalogue, optionally save it, and make the plot."""

    catalogue = compute_stress_catalogue(
        total_mass_msun,
        sxs_ids=sxs_ids,
        download=download,
        progress=progress,
        continue_on_error=continue_on_error,
    )
    if json_path is not None:
        save_catalogue(catalogue, json_path)
    figure, axis = plot_stress_mismatches(
        catalogue,
        output_path=output_path,
        show=show,
    )
    catalogue["figure"] = figure
    catalogue["axis"] = axis
    return catalogue


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument(
        "--sxs-id",
        action="append",
        dest="sxs_ids",
        help=(
            "stress SXS ID to include; repeat for a subset.  The default is "
            "all IDs in the stress section of BBH_ID.text"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/stress_q_chi_pn_colorbar.pdf"),
    )
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="plot an existing result catalogue without recomputing",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.from_json is not None:
        if args.sxs_ids is not None:
            raise ValueError("--sxs-id cannot be combined with --from-json")
        catalogue = load_catalogue(args.from_json)
        print_mismatch_summary(catalogue)
        plot_stress_mismatches(
            catalogue,
            output_path=args.output,
            show=args.show,
        )
    else:
        catalogue = compute_stress_catalogue(
            args.total_mass,
            sxs_ids=args.sxs_ids,
            download=not args.no_download,
            progress=not args.no_progress,
            continue_on_error=args.continue_on_error,
        )
        if args.json_path is not None:
            saved_json = save_catalogue(catalogue, args.json_path)
            print(f"Saved {saved_json}")
        plot_stress_mismatches(
            catalogue,
            output_path=args.output,
            show=args.show,
        )

    print(f"Saved {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
