"""Pure-NumPy gravitational waveform from the physics-factored KAN model.

No Torch, notebook state, or training data are required. Waveform evaluation
uses NumPy together with the exported coefficients in ``KAN_model/KAN_sxs_numpy.json``.

The model is

    h~(f) = (M_sec^2 / D_sec) a(Mf) exp[i phi(Mf)]

    a(Mf)   = a_N(Mf; eta) exp[dlnA_KAN(Mf; theta)]
    phi(Mf) = -Psi_TF2(Mf; theta) + r_KAN(Mf; theta) + 2 pi Mf tau_hat + phi_0

with the Newtonian stationary-phase amplitude and aligned-spin 3.5PN
TaylorF2 phase retained explicitly. Only ``dlnA_KAN`` and ``r_KAN`` are
learned, and the exported correction is evaluated as cubic B-spline products.

Public API
    KAN_mf              dimensionless positive-frequency waveform on an Mf grid
    KAN                 physical-frequency waveform in strain/Hz
    TimeD_KAN           band-limited inverse transform in geometric units
    TimeD_KAN_physical  time-domain waveform in physical units
    kan_residuals       learned phase and log-amplitude corrections
    model_domain        declared numerical support of the exported model
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

__all__ = ["KAN_mf", "KAN", "TimeD_KAN", "TimeD_KAN_physical",
           "band_taper", "signal_duration", "kan_residuals",
           "model_domain", "load_model"]

# ---- constants ----
MTSUN_SI = 4.925491025543576e-06      # G M_sun / c^3, seconds
MPC_SI = 3.085677581491367e+22        # megaparsec, metres
C_SI = 299792458.0                    # speed of light, m/s
EULER_GAMMA = float(np.euler_gamma)
SPIN_WEIGHTED_Y22_FACEON = float(np.sqrt(5.0 / (4.0 * np.pi)))
FACE_ON_FFT_FACTOR = 0.5

# ---- ported analytic physics (NumPy only) ----

def mass_params(m1, m2):
    """Return (M, eta, delta) from component masses (any consistent unit)."""
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    M = m1 + m2
    eta = m1 * m2 / M**2
    delta = (m1 - m2) / M
    return M, eta, delta


def chi_pn(eta, chi1, chi2):
    """Effective aligned-spin parameter used by IMRPhenomD's calibration.

    chi_PN = chi_s (1 - 76 eta / 113) + delta chi_a  is (up to normalisation)
    the spin combination entering the leading-order spin-orbit phase term, so
    it is the single most informative spin variable for the phasing.
    """
    eta = np.asarray(eta, dtype=np.float64)
    delta = np.sqrt(np.maximum(1.0 - 4.0 * eta, 0.0))
    chi_s = 0.5 * (np.asarray(chi1, dtype=np.float64) + np.asarray(chi2, dtype=np.float64))
    chi_a = 0.5 * (np.asarray(chi1, dtype=np.float64) - np.asarray(chi2, dtype=np.float64))
    return chi_s * (1.0 - 76.0 * eta / 113.0) + delta * chi_a


def chi_extra(eta, chi1, chi2):
    """Exchange-symmetric second spin coordinate: delta * chi_a."""
    eta = np.asarray(eta, dtype=np.float64)
    delta = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    return delta * 0.5 * (np.asarray(chi1, dtype=np.float64)
                          - np.asarray(chi2, dtype=np.float64))


def final_spin_estimate(m1, m2, chi1, chi2):
    """Approximate remnant Kerr spin for aligned-spin BBH."""
    M, eta, _ = mass_params(m1, m2)
    S = (np.asarray(m1) ** 2 * np.asarray(chi1) + np.asarray(m2) ** 2 * np.asarray(chi2)) / M**2
    # Nonspinning part: Tichy-Marronetti-style eta polynomial
    # (2*sqrt(3) eta from the ISCO angular momentum of a test particle).
    af0 = 2.0 * np.sqrt(3.0) * eta - 3.871 * eta**2 + 4.028 * eta**3
    # Spin part interpolates between the test-particle limit (a_f -> chi1 as
    # eta -> 0) and the NR-calibrated equal-mass behaviour.
    af = af0 + S * (1.0 - 1.5 * eta)
    return np.clip(af, -0.998, 0.998)


def radiated_energy_estimate(m1, m2, chi1, chi2):
    """Approximate radiated energy fraction E_rad/M for aligned-spin BBH."""
    M, eta, _ = mass_params(m1, m2)
    m1a = np.asarray(m1, dtype=np.float64)
    m2a = np.asarray(m2, dtype=np.float64)
    s_bar = (m1a**2 * np.asarray(chi1) + m2a**2 * np.asarray(chi2)) / (m1a**2 + m2a**2)
    # eta-polynomial with the correct test-particle limit (1 - sqrt(8)/3) eta.
    base = (0.0559745 * eta + 0.5809510 * eta**2
            - 0.9606726 * eta**3 + 3.3524112 * eta**4)
    return np.clip(base * (1.0 + 0.5 * s_bar), 0.0, 0.12)


def qnm_omega_22(af):
    """Berti et al. fit to the l=m=2, n=0 Kerr QNM frequency.

    UNITS: dimensionless, in REMNANT-mass units -- M_f * omega_R
    (omega_R = 2 pi f_RD), NOT M * omega_R with M the initial total mass.
    """
    af = np.asarray(af, dtype=np.float64)
    return 1.5251 - 1.1568 * (1.0 - af) ** 0.1292


def qnm_quality_22(af):
    """Berti et al. fit to the l=m=2, n=0 Kerr quality factor Q (dimensionless).

    The companion of `qnm_omega_22`, which gives only the REAL frequency.  The
    damping time follows as tau = 2 Q / omega_R, in remnant-mass units (tau/M_f).
    """
    af = np.asarray(af, dtype=np.float64)
    return 0.7000 + 1.4187 * (1.0 - af) ** (-0.4990)


def mf_ringdown(m1, m2, chi1, chi2):
    """Dimensionless ringdown frequency Mf_RD in units of the INITIAL total mass M.

    x_RD = M f_RD = [M_f omega_R] / [2 pi (M_f/M)]: DIVIDE the remnant-mass-unit
    QNM frequency by the remnant mass fraction, not multiply (see KAN_utils.py
    for the full derivation).
    """
    af = final_spin_estimate(m1, m2, chi1, chi2)
    mfinal_frac = 1.0 - radiated_energy_estimate(m1, m2, chi1, chi2)  # M_f / M
    return qnm_omega_22(af) / (2.0 * np.pi * mfinal_frac)


def taylorf2_phase(Mf, m1, m2, chi1, chi2):
    """Frequency-domain TaylorF2 phase at 3.5PN, aligned spins (radians).

    Convention: h~(f) = a(f) * exp(i * phi(f)) with phi = Phi_TF2 given below
    (t_c = 0, phi_c = 0).  The sign convention is fixed empirically against
    PyCBC/LALSimulation in the validation step of the training notebook.

    Coefficients follow Appendix B of the IMRPhenomD paper
    (Khan et al., arXiv:1508.07253), i.e. the standard LALSimulation TaylorF2
    phasing with spin-orbit terms through 3.5PN and spin^2 terms at 2PN.
    """
    Mf = np.asarray(Mf, dtype=np.float64)
    M, eta, delta = mass_params(m1, m2)
    chi_s = 0.5 * (np.asarray(chi1, dtype=np.float64) + np.asarray(chi2, dtype=np.float64))
    chi_a = 0.5 * (np.asarray(chi1, dtype=np.float64) - np.asarray(chi2, dtype=np.float64))

    v = (np.pi * np.maximum(Mf, 1e-12)) ** (1.0 / 3.0)
    logv = np.log(v)

    phi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    phi3 = (-16.0 * np.pi
            + 113.0 * delta * chi_a / 3.0
            + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s)
    phi4 = (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0
            + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
            - (405.0 / 4.0) * delta * chi_a * chi_s
            + (-405.0 / 8.0 + 5.0 * eta / 2.0) * chi_s**2)
    phi5c = (38645.0 * np.pi / 756.0 - 65.0 * np.pi * eta / 9.0
             - delta * (732985.0 / 2268.0 + 140.0 * eta / 9.0) * chi_a
             - (732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta**2 / 9.0) * chi_s)
    phi6 = (11583231236531.0 / 4694215680.0
            - 6848.0 * EULER_GAMMA / 21.0
            - 640.0 * np.pi**2 / 3.0
            + (-15737765635.0 / 3048192.0 + 2255.0 * np.pi**2 / 12.0) * eta
            + 76055.0 * eta**2 / 1728.0
            - 127825.0 * eta**3 / 1296.0
            - 6848.0 * np.log(4.0) / 21.0
            + np.pi * (2270.0 * delta * chi_a / 3.0 + (2270.0 / 3.0 - 520.0 * eta) * chi_s))
    phi6_log = -6848.0 / 21.0
    phi7 = (np.pi * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta**2 / 756.0)
            + delta * (-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0
                       - 1985.0 * eta**2 / 48.0) * chi_a
            + (-25150083775.0 / 3048192.0 + 10566655595.0 * eta / 762048.0
               - 1042165.0 * eta**2 / 3024.0 + 5345.0 * eta**3 / 36.0) * chi_s)

    series = (1.0
              + phi2 * v**2
              + phi3 * v**3
              + phi4 * v**4
              + phi5c * (1.0 + 3.0 * logv) * v**5
              + (phi6 + phi6_log * logv) * v**6
              + phi7 * v**7)
    return 3.0 / (128.0 * eta * v**5) * series - np.pi / 4.0


def newtonian_log_amp(Mf, eta):
    """log of the geometric Newtonian FD amplitude a_N(Mf; eta).

    a_N = sqrt(5/24) * pi^(-2/3) * sqrt(eta) * (Mf)^(-7/6),
    defined so that |h~(f)| = (M_sec^2 / D_sec) * a_N * (1 + PN corrections)
    for a face-on binary (inclination 0, plus polarisation).
    """
    Mf = np.asarray(Mf, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    return (0.5 * np.log(5.0 / 24.0) - (2.0 / 3.0) * np.log(np.pi)
            + 0.5 * np.log(eta) - (7.0 / 6.0) * np.log(np.maximum(Mf, 1e-12)))


def geometric_prefactor(mtotal_msun, distance_mpc):
    """M_sec^2 / D_sec: converts geometric amplitude to strain/Hz."""
    m_sec = mtotal_msun * MTSUN_SI
    d_sec = distance_mpc * MPC_SI / C_SI
    return m_sec**2 / d_sec


_BSPLINE_DEGREE = 3  # cubic


def _clamped_knots(n_basis, x_lo, x_hi, degree=_BSPLINE_DEGREE):
    """Clamped (open) uniform knot vector."""
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_basis must be >= degree + 1 = {degree + 1}, got {n_basis}")
    if n_interior > 0:
        interior = np.linspace(x_lo, x_hi, n_interior + 2, dtype=np.float64)[1:-1]
    else:
        interior = np.zeros(0, dtype=np.float64)
    return np.concatenate([
        np.full(degree + 1, x_lo, dtype=np.float64),
        interior,
        np.full(degree + 1, x_hi, dtype=np.float64),
    ])


def bspline_basis_np(x, n_basis, x_lo, x_hi, degree=_BSPLINE_DEGREE):
    """Clamped uniform cubic B-spline basis, numpy.  Returns [..., n_basis].

    Inputs outside the domain are clamped to the boundary before evaluation.
    """
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    xf = np.clip(x.ravel(), x_lo, x_hi)
    knots = _clamped_knots(n_basis, x_lo, x_hi, degree)
    n_deg0 = n_basis + degree

    real = [i for i in range(n_deg0) if knots[i + 1] > knots[i]]
    last_real = real[-1] if real else n_deg0 - 1
    N = np.zeros((xf.shape[0], n_deg0), dtype=np.float64)
    for i in range(n_deg0):
        lo, hi = knots[i], knots[i + 1]
        if hi <= lo:
            continue
        mask = (xf >= lo) & (xf <= hi) if i == last_real else (xf >= lo) & (xf < hi)
        N[mask, i] = 1.0

    for d in range(1, degree + 1):
        width = n_deg0 - d
        left_den = knots[d:d + width] - knots[0:width]
        right_den = knots[d + 1:d + 1 + width] - knots[1:1 + width]
        m1 = left_den > 0
        m2 = right_den > 0
        term1 = np.zeros((xf.shape[0], width))
        term2 = np.zeros((xf.shape[0], width))
        if m1.any():
            term1[:, m1] = ((xf[:, None] - knots[0:width][None, m1]) / left_den[None, m1]
                            * N[:, 0:width][:, m1])
        if m2.any():
            term2[:, m2] = ((knots[d + 1:d + 1 + width][None, m2] - xf[:, None]) / right_den[None, m2]
                            * N[:, 1:1 + width][:, m2])
        N = term1 + term2

    return N.reshape(orig_shape + (n_basis,))


def _aff(x, lo, hi):
    return 2.0 * (np.asarray(x, dtype=np.float64) - lo) / (hi - lo) - 1.0



# =============================================================================
# EXPORTED MODEL
# =============================================================================

_DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "KAN_model" / "KAN_sxs_numpy.json"
_CACHE: dict = {}


def load_model(path=None) -> dict:
    """Load the exported KAN coefficient set."""
    p = Path(path) if path is not None else _DEFAULT_MODEL
    key = str(p.resolve())
    if key in _CACHE:
        return _CACHE[key]
    if not p.exists():
        raise FileNotFoundError(
            f"exported model not found: {p}\n"
            "Expected KAN_model/KAN_sxs_numpy.json or pass path= explicitly.")
    raw = json.loads(p.read_text())

    def rebuild(branch):
        out = {"f0": np.asarray(branch["f0"], float),
               "g0": np.asarray(branch["g0"], float)}
        for k in ("u", "w"):
            b = branch[k]
            out[k] = {"rank": int(b["rank"]),
                      "active": np.asarray(b["active"], float),
                      "fx": [np.asarray(c, float) for c in b["fx"]],
                      "S": np.asarray(b["S"], float),
                      "zx": ([np.asarray(c, float) for c in b["zx"]]
                             if b.get("zx") is not None else None)}
        return out

    model = {"config": raw["config"], "phase": rebuild(raw["phase"]),
             "amp": rebuild(raw["amp"]), "domain": raw.get("domain", {})}
    _CACHE[key] = model
    return model


def model_domain(path=None) -> dict:
    """Declared support of the model in intrinsic parameters and Mf."""
    return load_model(path).get("domain", {})


def _check_domain(model, mf, eta, chi1z, chi2z, strict):
    dom = model.get("domain") or {}
    if not dom:
        return
    msgs = []
    cpn = float(chi_pn(eta, chi1z, chi2z))
    zex = float(chi_extra(eta, chi1z, chi2z))
    for nm, val in (("eta", float(eta)), ("chi_PN", cpn), ("zeta", zex)):
        lo, hi = dom.get(nm, (None, None))
        if lo is not None and not (lo <= val <= hi):
            msgs.append(f"{nm} = {val:.4f} outside the trained range [{lo:.4f}, {hi:.4f}]")
    lo, hi = dom.get("Mf", (None, None))
    if lo is not None:
        mf = np.asarray(mf, float)
        frac = float(((mf < lo) | (mf > hi)).mean())
        if frac > 0:
            msgs.append(f"{100*frac:.1f}% of the requested Mf lies outside the "
                        f"validated band [{lo:.5f}, {hi:.5f}]")
    if msgs:
        text = "KAN model evaluated outside its declared support:\n  " + "\n  ".join(msgs)
        if strict:
            raise ValueError(text)
        warnings.warn(text, RuntimeWarning, stacklevel=3)


# =============================================================================
# KAN RESIDUALS
# =============================================================================

def kan_residuals(mf, eta, chi1z, chi2z, path=None,
                  mf_rd=None, strict_domain=False):
    """(r_KAN, dlnA_KAN) on an Mf grid, for one binary.  Pure NumPy."""
    model = load_model(path)
    c = model["config"]
    mf = np.asarray(mf, dtype=np.float64)
    eta = float(eta)
    _check_domain(model, mf, eta, chi1z, chi2z, strict_domain)

    if mf_rd is None:
        m1 = 0.5 * (1.0 + np.sqrt(max(1.0 - 4.0 * eta, 0.0)))
        m2 = 1.0 - m1
        mf_rd = float(mf_ringdown(m1, m2, chi1z, chi2z))
    cpn = float(chi_pn(eta, chi1z, chi2z))
    zex = float(chi_extra(eta, chi1z, chi2z))

    Bu = bspline_basis_np(_aff((np.pi * mf) ** (1.0 / 3.0), c["v_lo"], c["v_hi"]),
                          c["n_u"], c["u_lo"], c["u_hi"])
    Bw = bspline_basis_np(mf / mf_rd, c["n_w"], c["w_lo"], c["w_hi"])
    Be = bspline_basis_np(_aff(np.array([eta]), c["eta_lo"], c["eta_hi"]),
                          c["n_theta"], c["t_lo"], c["t_hi"])[0]
    Bc = bspline_basis_np(_aff(np.array([cpn]), c["chipn_lo"], c["chipn_hi"]),
                          c["n_theta"], c["t_lo"], c["t_hi"])[0]
    Bz = bspline_basis_np(_aff(np.array([zex]), c["zex_lo"], c["zex_hi"]),
                          c["n_theta"], c["t_lo"], c["t_hi"])[0]

    def branch(br):
        out = Bu @ br["f0"] + Bw @ br["g0"]
        for key, Bx in (("u", Bu), ("w", Bw)):
            b = br[key]
            for rho in range(b["rank"]):
                if b["active"][rho] == 0.0:
                    continue
                th = float(Be @ b["S"][rho] @ Bc)
                if b["zx"] is not None:
                    th *= float(Bz @ b["zx"][rho])
                out = out + (Bx @ b["fx"][rho]) * th
        return out

    return branch(model["phase"]), branch(model["amp"])


# =============================================================================
# DIMENSIONLESS WAVEFORM
# =============================================================================

def KAN_mf(mf, eta, chi1z, chi2z, tau_hat=0.0, phi0=0.0,
           mf_rd=None, path=None, strict_domain=False, components=False):
    """Dimensionless positive-frequency waveform on an ``Mf`` grid."""
    mf = np.asarray(mf, dtype=np.float64)
    r, dlnA = kan_residuals(mf, eta, chi1z, chi2z, path=path,
                            mf_rd=mf_rd, strict_domain=strict_domain)
    m1 = 0.5 * (1.0 + np.sqrt(max(1.0 - 4.0 * float(eta), 0.0)))
    m2 = 1.0 - m1
    log_a = newtonian_log_amp(mf, float(eta)) + dlnA
    psi = taylorf2_phase(mf, m1, m2, chi1z, chi2z)
    phase = -psi + r + 2.0 * np.pi * mf * float(tau_hat) + float(phi0)
    if components:
        return dict(amplitude=np.exp(log_a), phase=phase, r=r, delta_lnA=dlnA,
                    psi_tf2=psi, mf=mf)
    return np.exp(log_a) * np.exp(1j * phase)


# =============================================================================
# PHYSICAL WAVEFORM
# =============================================================================

def _canonical_masses_spins(m1, m2, chi1, chi2):
    """Enforce m1 >= m2 convention, swapping the associated spin WITH the mass.

    The internal model always represents a binary with the canonical (larger)
    mass first; a caller may legally pass m1 < m2 (e.g. `KAN(f, m2, m1, chi2,
    chi1)` must equal `KAN(f, m1, m2, chi1, chi2)`), so mass and spin must be
    swapped TOGETHER here or the spin gets silently reassigned to the wrong body.
    """
    m1 = float(m1); m2 = float(m2); chi1 = float(chi1); chi2 = float(chi2)
    if m1 < m2:
        return m2, m1, chi2, chi1
    return m1, m2, chi1, chi2


def KAN(f_hz, m1_msun, m2_msun, chi1z, chi2z, distance_mpc,
        tau_hat=0.0, phi0=0.0, path=None, strict_domain=False,
        components=False):
    """Physical positive-frequency waveform ``h~(f)`` in strain/Hz."""
    m1_msun, m2_msun, chi1z, chi2z = _canonical_masses_spins(
        m1_msun, m2_msun, chi1z, chi2z)
    f_hz = np.asarray(f_hz, dtype=np.float64)
    M = m1_msun + m2_msun
    eta = m1_msun * m2_msun / M**2
    m_sec = M * MTSUN_SI
    d_sec = float(distance_mpc) * MPC_SI / C_SI
    mf = m_sec * f_hz
    pref = m_sec**2 / d_sec
    out = KAN_mf(mf, eta, chi1z, chi2z, tau_hat=tau_hat, phi0=phi0,
                 path=path, strict_domain=strict_domain, components=components)
    if components:
        out["amplitude"] = pref * out["amplitude"]
        out["f_hz"] = f_hz
        out["prefactor"] = pref
        return out
    return pref * out


# =============================================================================
# TIME DOMAIN
# =============================================================================
# The KAN is a FREQUENCY-domain model, so going to the time domain is an
# INVERSE transform of the analytic function -- never an inverse transform of a
# stored training array.
#
# This project's convention for the complex face-on h22 is
#
#     H(Mf) = dtau * FFT[h(tau)]                                        (1)
#     h_tilde_plus(Mf) = (1/2) conj[H(-Mf)]   for Mf > 0                (2)
#
# so essentially all of the power of the complex h22 sits on the NEGATIVE
# frequency branch.  The reconstruction must therefore build the negative
# branch, NOT assume the Hermitian spectrum of a real signal.  Inverting (2),
#
#     H(-Mf) = conj[2 h_tilde_plus(Mf)],  Mf > 0,   H(Mf >= 0) = 0
#
# and inverting (1),  h(tau) = IFFT[H] / dtau.


def band_taper(mf, mf_lo, mf_hi, frac=0.05):
    """Smooth cosine edge taper on [mf_lo, mf_hi]; zero outside.

    This exists ONLY to stop a sharp spectral cutoff from ringing in the time
    domain.  It is a reconstruction device applied to a frequency band, and is
    a completely different object from the time-domain window used to condition
    the SXS training data.  Do not conflate the two.
    """
    mf = np.asarray(mf, dtype=np.float64)
    w = np.zeros_like(mf)
    inside = (mf >= mf_lo) & (mf <= mf_hi)
    w[inside] = 1.0
    width = frac * (mf_hi - mf_lo)
    if width <= 0:
        return w
    lo = inside & (mf < mf_lo + width)
    hi = inside & (mf > mf_hi - width)
    w[lo] = 0.5 * (1.0 - np.cos(np.pi * (mf[lo] - mf_lo) / width))
    w[hi] = 0.5 * (1.0 - np.cos(np.pi * (mf_hi - mf[hi]) / width))
    return w


def signal_duration(eta, mf_lo):
    """Newtonian time from frequency `mf_lo` to merger, in units of M.

    t = (5/256) / [eta (pi Mf)^{8/3}].  Used to check that a requested time
    grid is long enough to contain the waveform; a grid shorter than this
    wraps the inspiral onto the merger and corrupts it silently.
    """
    return float((5.0 / 256.0) / (float(eta) * (np.pi * float(mf_lo)) ** (8.0 / 3.0)))


def TimeD_KAN(eta, chi1z, chi2z, *, dtau=0.5, n=None,
              tau_span=None, mf_lo=None, mf_hi=None, edge_frac=0.05,
              tau_shift=0.0, phi_shift=0.0, mf_rd=None, path=None,
              strict_domain=False, return_spectrum=False,
              center=True, check_duration=True):
    """Band-limited time-domain face-on h22 of the analytic KAN.

    The analytic KAN is EVALUATED on a uniform Mf grid and inverse transformed;
    no stored waveform is used anywhere.

    `tau_shift` and `phi_shift` are explicit extrinsic alignment handles: the
    SXS time/phase origin and the KAN affine gauge are both arbitrary, so a
    direct overlay must align before any disagreement is claimed.

    The inverse transform is periodic, so the raw result is circularly wrapped:
    the merger can land anywhere in the array.  With ``center=True`` (default)
    the series is rolled so the merger sits at one quarter of the span, with
    the inspiral contiguous before it, and ``tau`` is measured from the merger.
    Pass ``center=False`` for the raw periodic array.

    The grid must be long enough to hold the signal.  With
    ``check_duration=True`` a grid shorter than the Newtonian time from the
    lower band edge raises, because the alternative is a silently corrupted
    waveform: at half the required span the merger is wrong by ~20%.

    Returns (tau, h22) with tau in units of M, h22 the complex face-on strain.
    """
    dom = load_model(path).get("domain", {})
    if mf_lo is None:
        mf_lo = float(dom.get("Mf", (0.006, 0.12))[0])
    if mf_hi is None:
        mf_hi = float(dom.get("Mf", (0.006, 0.12))[1])

    if n is None:
        n = int(round(tau_span / dtau)) if tau_span else 1 << 16
    n = int(n)

    need = signal_duration(eta, mf_lo)
    span = n * dtau
    if check_duration and span < need:
        raise ValueError(
            f"time grid too short: span = {span:,.0f} M but the waveform lasts "
            f"~{need:,.0f} M from Mf = {mf_lo:.5f} at eta = {float(eta):.4f}. "
            f"The inspiral would wrap onto the merger and corrupt it. "
            f"Use n >= {int(2 ** np.ceil(np.log2(need / dtau)))} at dtau = {dtau}, "
            f"or raise mf_lo.")
    mfg = np.fft.fftfreq(n, d=dtau)              # dimensionless Mf grid

    H = np.zeros(n, dtype=np.complex128)
    neg = mfg < 0
    a = -mfg[neg]                                # positive |Mf| on the negative branch
    sel = (a >= mf_lo) & (a <= mf_hi)
    if sel.any():
        hp = KAN_mf(a[sel], eta, chi1z, chi2z, tau_hat=tau_shift,
                    phi0=phi_shift, path=path, mf_rd=mf_rd,
                    strict_domain=strict_domain, components=False)
        hp = hp * band_taper(a[sel], mf_lo, mf_hi, edge_frac)
        idx = np.where(neg)[0][sel]
        H[idx] = np.conj(2.0 * hp)               # inverse of h_plus = 1/2 conj H(-Mf)

    h = np.fft.ifft(H) / dtau                    # inverse of H = dtau FFT[h]
    tau = np.arange(n) * dtau
    if center:
        # Roll the merger to a quarter of the span so the inspiral is
        # contiguous ahead of it, and measure tau from the merger.
        k_pk = int(np.argmax(np.abs(h)))
        k_target = n // 4
        h = np.roll(h, k_target - k_pk)
        tau = (np.arange(n) - k_target) * dtau
    if return_spectrum:
        return tau, h, mfg, H
    return tau, h


def TimeD_KAN_physical(m1_msun, m2_msun, chi1z, chi2z, distance_mpc,
                       **kw):
    """`TimeD_KAN` in seconds and physical strain.

    Returns (t_seconds, h_strain) for the complex face-on h22.
    """
    m1_msun, m2_msun, chi1z, chi2z = _canonical_masses_spins(m1_msun, m2_msun, chi1z, chi2z)
    M = m1_msun + m2_msun
    eta = m1_msun * m2_msun / M**2
    m_sec = M * MTSUN_SI
    d_sec = float(distance_mpc) * MPC_SI / C_SI
    tau, h = TimeD_KAN(eta, chi1z, chi2z, **kw)
    return tau * m_sec, h * (m_sec / d_sec)
