# Physics-factored KAN gravitational waveforms

This repository provides the exported waveform model and the lightweight analysis code used to reproduce the waveform comparisons and plotting workflow for the SXS-based KAN study.

The model keeps analytically controlled inspiral structure explicit. The Newtonian frequency-domain amplitude and aligned-spin 3.5PN TaylorF2 phase form the baseline, while a compact Kolmogorov–Arnold network supplies the logarithmic-amplitude and intrinsic phase corrections learned from SXS numerical-relativity waveforms. After export, waveform evaluation is pure NumPy and does not require Torch or the training notebooks.

The current model is calibrated to quasi-circular, non-precessing binary black holes in the dominant `(2,2)` mode over approximately `1 <= q <= 8` and `|chi1z|, |chi2z| <= 0.8`.

## Repository layout

```text
.
├── KAN_model/
│   └── KAN_sxs_numpy.json
├── KAN_utilities/
│   ├── __init__.py
│   └── KAN_Waveform.py
├── plot_utilities/
│   ├── __init__.py
│   ├── _mismatch_support.py
│   ├── conditioning_check_mismatch.py
│   ├── consistent_conditioning_mismatch.py
│   ├── detector_weighted_mismatch.py
│   ├── detector_weighted_mismatch_for_varied_mass.py
│   ├── stress_detector_weighted_mismatch.py
│   └── waveform_plotting.py
├── BBH_ID.text
├── How_to_KAN.md
├── plot_prep.ipynb
├── requirements-plot-prep.txt
└── .gitignore
```

`KAN_model/KAN_sxs_numpy.json` is the single exported coefficient set used by the evaluator. `KAN_utilities/KAN_Waveform.py` reconstructs the analytic waveform directly from those coefficients. The files in `plot_utilities/` handle SXS and NRSur downloads, waveform conditioning, mismatch calculations, and figure construction.

## Quick waveform evaluation

```python
import numpy as np
from KAN_utilities import KAN_Waveform as KW

f = np.linspace(20.0, 512.0, 4000)
h = KW.KAN(
    f,
    m1_msun=36.0,
    m2_msun=29.0,
    chi1z=0.2,
    chi2z=0.1,
    distance_mpc=400.0,
)
```

For geometric-frequency calculations

```python
mf = np.geomspace(0.006, 0.12, 800)
h_mf = KW.KAN_mf(mf, eta=0.24, chi1z=0.2, chi2z=0.1)
```

See [How_to_KAN.md](How_to_KAN.md) for the full user manual, including conventions, model support, residual access, and time-domain reconstruction.

## Reproducing the paper analysis

The main entry point is `plot_prep.ipynb`. The notebook is organized as a linear reproduction workflow and records the numerical summaries beside the plots. It covers

- representative SXS, KAN, NRSur7dq4, and IMRPhenomD waveform comparisons
- flat-noise time- and phase-maximized mismatch distributions
- the TaylorF2/Newtonian baseline comparison
- the conditioning sensitivity check
- detector-weighted mismatch as a function of total mass
- the sparse-coverage stress sample

Install the analysis dependencies with

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-plot-prep.txt
```

Then start Jupyter from the repository root and run `plot_prep.ipynb` from top to bottom.

The plotting utilities download SXS and NRSur resources when required. These files are stored under `plot_utilities/waveform_data/` and are excluded from version control by `.gitignore`.

## SXS split record

`BBH_ID.text` records the exact IDs reconstructed from the frozen split used in the accompanying paper, *Learning Inspiral–Merger–Ringdown Waveforms from a Post-Newtonian Baseline*, by **Arghya Chattopadhyay** and **Shilpa Kastha**. It contains

- 280 training systems
- 75 validation systems
- 75 interpolation-test systems
- 30 sparse-coverage stress systems
- the preprocessing exclusion
- the four interpolation-test systems omitted from strict NRSur7dq4 common-model comparisons

The interpolation test and stress sets are kept distinct because the latter probes regions with sparse NR coverage rather than the well-supported interpolation domain.

## Generated files

Running the notebook creates the `figures/` directory and may create downloaded waveform caches under `plot_utilities/waveform_data/`. These generated files are ignored by Git. The exported KAN coefficient file, source code, notebook, and ID manifest remain tracked.

## Scope

The evaluator should not be interpreted as uniformly calibrated over the entire numerical spline box. Accuracy follows the available NR coverage, and comparisons should use the intersection of the trustworthy frequency supports of the models involved.

## Citation

If this repository or the methodology is useful in your work, please cite the accompanying paper.

```bibtex
@article{ChattopadhyayKastha,
soon to be updated
}
```

---
