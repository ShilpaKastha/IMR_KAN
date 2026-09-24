# How to use the exported KAN waveform

This repository contains a pure-NumPy evaluator for the physics-factored KAN waveform. The trained network is not required at evaluation time. The complete learned representation is stored in

```text
KAN_model/KAN_sxs_numpy.json
```

and is evaluated by

```text
KAN_utilities/KAN_Waveform.py
```

The model retains the Newtonian stationary-phase amplitude and aligned-spin 3.5PN TaylorF2 phase explicitly. The exported KAN supplies the logarithmic-amplitude correction and intrinsic phase correction learned from SXS numerical-relativity waveforms.

## 1. Quick start

From the repository root

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

amplitude = np.abs(h)
phase = np.unwrap(np.angle(h))
```

`KW.KAN` returns the positive-frequency waveform in strain per Hz.

## 2. Model convention

The frequency-domain waveform is

```text
h_tilde(f) = (M_sec^2 / D_sec)
             a_N(Mf, eta)
             exp[dlnA_KAN(Mf, theta)]
             exp{i[-Psi_TF2(Mf, theta) + r_KAN(Mf, theta)
                   + 2 pi Mf tau_hat + phi0]}.
```

Here `Mf = M_sec f`, `M_sec = G M/c^3`, and `D_sec = D_L/c`. The KAN learns only `dlnA_KAN` and `r_KAN`.

The intrinsic coordinates are

```text
chi_s  = (chi1z + chi2z)/2
chi_a  = (chi1z - chi2z)/2
delta  = sqrt(1 - 4 eta)
chi_PN = (1 - 76 eta/113) chi_s + delta chi_a
zeta   = delta chi_a
```

and the two frequency coordinates are the PN velocity coordinate and the ringdown-scaled coordinate `Mf/Mf_RD`.

## 3. Mass and spin ordering

The physical API accepts the masses in either order. The evaluator internally enforces `m1 >= m2` and swaps the associated spins together with the masses.

```python
h1 = KW.KAN(f, 36.0, 29.0, 0.2, 0.1, 400.0)
h2 = KW.KAN(f, 29.0, 36.0, 0.1, 0.2, 400.0)
np.allclose(h1, h2)
```

For the dimensionless `eta`-based API, the caller must provide `chi1z` for the larger-mass component because the individual masses are not supplied to that function.

## 4. Dimensionless waveform

For comparisons with numerical relativity in geometric units use

```python
mf = np.geomspace(0.006, 0.12, 800)
h_mf = KW.KAN_mf(
    mf,
    eta=0.24,
    chi1z=0.2,
    chi2z=0.1,
)
```

`KW.KAN_mf` returns the waveform with the overall `M_sec^2/D_sec` factor removed.

To inspect the analytic and learned pieces separately

```python
components = KW.KAN_mf(
    mf,
    eta=0.24,
    chi1z=0.2,
    chi2z=0.1,
    components=True,
)

components["amplitude"]
components["phase"]
components["r"]
components["delta_lnA"]
components["psi_tf2"]
```

The learned residuals alone are also available through

```python
r, dlnA = KW.kan_residuals(mf, 0.24, 0.2, 0.1)
```

## 5. Calibration and numerical support

The model was trained on quasi-circular, non-precessing binary black holes using the dominant `(2,2)` mode. The SXS sample spans approximately

- `1 <= q <= 8`
- `|chi1z| <= 0.8`
- `|chi2z| <= 0.8`

The exported spline domain can be inspected with

```python
KW.model_domain()
```

The numerical spline bounds are not a guarantee of uniform physical accuracy. The SXS coverage is nonuniform and each numerical simulation contributes over its own trustworthy frequency interval. For comparisons against another waveform model, use the intersection of the valid supports.

Set `strict_domain=True` in `KW.KAN` or `KW.KAN_mf` when you want out-of-domain evaluations to raise rather than warn.

## 6. Time-domain reconstruction

The model is fundamentally frequency-domain. `TimeD_KAN` evaluates the analytic waveform on a uniform `Mf` grid and performs the inverse transform using the frequency convention adopted in the project.

```python
tau, h22 = KW.TimeD_KAN(
    eta=0.24,
    chi1z=0.2,
    chi2z=0.1,
    dtau=0.5,
    mf_lo=0.006,
    mf_hi=0.12,
)
```

For physical units

```python
t_seconds, h_strain = KW.TimeD_KAN_physical(
    36.0,
    29.0,
    0.2,
    0.1,
    400.0,
    dtau=0.5,
    mf_lo=0.006,
    mf_hi=0.12,
)
```

A smooth band-edge taper is applied only for the inverse transform so that a sharp spectral cutoff does not produce artificial ringing. It is not the time-domain conditioning window used to prepare the SXS training targets.

## 7. Using a model file from another location

Every evaluator accepts an optional `path` argument

```python
h = KW.KAN_mf(
    mf,
    eta=0.24,
    chi1z=0.2,
    chi2z=0.1,
    path="/path/to/KAN_sxs_numpy.json",
)
```

If `path` is omitted, the evaluator uses `KAN_model/KAN_sxs_numpy.json` from this repository.

## 8. Reproducing the paper comparisons

`plot_prep.ipynb` is the main reproducibility notebook. It uses the exported KAN model together with SXS, NRSur7dq4, IMRPhenomD, and the analytic PN backbone to recreate the waveform comparisons, mismatch distributions, detector-weighted mass study, and sparse-coverage stress study.

The exact training, validation, interpolation-test, stress, preprocessing-exclusion, and NRSur comparison ID lists are recorded in `BBH_ID.text`.

Downloaded SXS and NRSur files are placed under `plot_utilities/waveform_data/`. That directory is ignored by Git so large waveform data created during the analysis are not accidentally committed.
