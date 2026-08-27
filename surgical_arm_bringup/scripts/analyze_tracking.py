#!/usr/bin/env python3
"""analyze_tracking.py -- cartesian EE tracking-error analysis of impedance logs.

Step 0 of the impedance -> insertion.py integration: BEFORE building a setpoint
channel, establish how accurately the cartesian impedance controller actually
holds/follows an EE pose. The container inner radius is 42 mm; the 2026-07-24
characterisation ended at ~51 mm static position offset and ~15 deg orientation
offset, which would make an impedance-mode insertion physically impossible. This
script quantifies that from the existing `--log` CSVs so the fix can be measured
rather than guessed.

WHAT IT DOES
------------
The log records joint state (t, q, dq, tau, taug, ramp, engage, xerr, trim) but
NOT the commanded EE pose, so the setpoint is RECONSTRUCTED here from the run's
interaction mode, exactly as run_impedance() would have computed it:

  hold-ee / render : p_des, quat_des = FK(q at t=0)          (static hold)
  track            : p_des = track_setpoint(A, B, t, period) (A = FK(q0),
                     B = A + --track-offset); quat_des = quat(q0)

The reconstruction is then VALIDATED against the logged `xerr` column, which is
the controller's own |p_cur - p_des| against its live setpoint. If the two agree
the reconstruction is trustworthy; where they diverge, a re-anchor capture moved
the setpoint mid-run and the script re-baselines at that event (see
--reanchor-jump). The residual mismatch is always reported -- do not trust the
numbers if it is large.

Orientation error has no logged counterpart, so it is reconstruction-only and
flagged as UNVALIDATED.

Metrics are reported over a SETTLED window (the transient after engage is
excluded, as are hand-guided/yielding intervals), and the error is decomposed
into a static bias (the mean, which a stiffer spring or better gravity model
fixes) and a drift slope (creep, which stiffness cannot fix -- see the j5/j7
control-dead-zone finding in the impedance notes).

USAGE
-----
  # A hold-ee run (defaults):
  ./analyze_tracking.py /tmp/imp_cart_anchor.csv

  # A track run -- the mode and its parameters are NOT in the log, so pass the
  # ones the run actually used:
  ./analyze_tracking.py /tmp/imp_cart_test.csv \\
      --interaction-mode track --track-offset 0 0.10 0 --track-period 6

  # Several logs at once, with plots, and only the last 20 s scored:
  ./analyze_tracking.py /tmp/imp_cart_*.csv --plot --settle-from -20

Needs PyKDL + urdf_parser_py (source your ROS setup). kortex_api is NOT needed:
this script stubs it out so impedance.py can be imported off-robot.
"""

import argparse
import csv
import os
import sys
import types

import numpy as np


# --- Import impedance.py off-robot ------------------------------------------
# impedance.py imports kortex_api at module scope, which only exists inside
# ~/.venvs/kortex_impedance on the robot host. Everything this script needs
# (KinDynModel, orientation_error, track_setpoint) is pure numpy/PyKDL and
# touches no Kortex symbol at import time -- there is no module-level use of
# any *_pb2 -- so stubbing the package lets the analysis run on the laptop
# while keeping ONE FK implementation shared with the controller. Reimplementing
# FK here would risk the analysis disagreeing with the thing it is measuring.
def _import_impedance():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import impedance  # noqa: F401
        return impedance
    except ImportError as exc:
        if "kortex_api" not in str(exc):
            raise
    for name in ("kortex_api",
                 "kortex_api.RouterClient", "kortex_api.SessionManager",
                 "kortex_api.TCPTransport", "kortex_api.UDPTransport",
                 "kortex_api.autogen",
                 "kortex_api.autogen.client_stubs",
                 "kortex_api.autogen.client_stubs.ActuatorConfigClientRpc",
                 "kortex_api.autogen.client_stubs.BaseClientRpc",
                 "kortex_api.autogen.client_stubs.BaseCyclicClientRpc",
                 "kortex_api.autogen.messages"):
        mod = types.ModuleType(name)
        # Any attribute access returns a dummy: the module-level code never
        # calls into these, and a run_impedance() call would fail loudly.
        mod.__getattr__ = lambda attr: types.SimpleNamespace()  # type: ignore
        sys.modules.setdefault(name, mod)
    import impedance  # noqa: F811
    print("NOTE: kortex_api stubbed (off-robot analysis); FK/quat code is the "
          "real impedance.py implementation.")
    return impedance


imp = _import_impedance()


# --- Log loading -------------------------------------------------------------
def load_log(path):
    """Read an impedance.py --log CSV into a dict of numpy columns."""
    cols = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for c in header:
            cols[c] = []
        for row in reader:
            if not row:
                continue
            for c, v in zip(header, row):
                cols[c].append(float(v))
    for c in cols:
        cols[c] = np.array(cols[c])
    # Joint count: q1..qN, excluding the q_-prefixed names analyse_csv guards on.
    n = sum(1 for c in header if c.startswith("q")
            and not c.startswith("q_") and c[1:].isdigit())
    if n == 0:
        raise SystemExit(f"{path}: no q1..qN columns -- not an impedance log?")
    q = np.column_stack([cols[f"q{i+1}"] for i in range(n)])
    dq = np.column_stack([cols[f"dq{i+1}"] for i in range(n)])
    return {
        "path": path, "n": n, "t": cols["t"], "q": q, "dq": dq,
        # 'engage' and 'xerr' postdate the earliest logs; tolerate their absence.
        "ramp": cols.get("ramp"),
        "engage": cols.get("engage"),
        "xerr": cols.get("xerr"),
        "header": header,
    }


# --- Setpoint reconstruction -------------------------------------------------
def reconstruct_setpoints(log, model, args):
    """Rebuild the commanded (p_des, quat_des) the controller was tracking.

    Returns (p_cur, quat_cur, p_des, quat_des, events, resid) where `events` are
    the sample indices at which a re-anchor capture was inferred and `resid` is
    the per-sample |reconstructed xerr - logged xerr| (the validation signal).
    """
    t, q = log["t"], log["q"]
    m = len(t)
    p_cur = np.zeros((m, 3))
    quat_cur = np.zeros((m, 4))
    for i in range(m):
        p_cur[i], quat_cur[i] = model.fk(q[i])

    p_des = np.zeros((m, 3))
    quat_des = np.zeros((m, 4))
    events = []

    if args.interaction_mode == "track":
        # p_a is the startup EE pose; p_b = p_a + --track-offset. Re-anchor is
        # not active in track mode (the controller drives p_des itself), so no
        # capture detection is needed.
        p_a = p_cur[0].copy()
        p_b = p_a + np.array(args.track_offset, dtype=float)
        for i in range(m):
            p_des[i] = imp.track_setpoint(p_a, p_b, t[i], args.track_period)
            quat_des[i] = quat_cur[0]
    else:
        # Static hold, with re-anchor captures re-baselining the setpoint. A
        # capture sets p_des <- p_cur, which shows up in the logged xerr as an
        # abrupt drop to ~0; that scalar is all the log preserves of the event,
        # so the capture pose is recovered as the measured pose at that sample.
        cur_p, cur_quat = p_cur[0].copy(), quat_cur[0].copy()
        xerr = log["xerr"]
        for i in range(m):
            if (xerr is not None and i > 0
                    and xerr[i] < xerr[i - 1] - args.reanchor_jump
                    and xerr[i] < args.reanchor_settled):
                cur_p, cur_quat = p_cur[i].copy(), quat_cur[i].copy()
                events.append(i)
            p_des[i] = cur_p
            quat_des[i] = cur_quat

    resid = None
    if log["xerr"] is not None:
        resid = np.abs(np.linalg.norm(p_cur - p_des, axis=1) - log["xerr"])
    return p_cur, quat_cur, p_des, quat_des, events, resid


# --- Windowing ---------------------------------------------------------------
def settled_mask(log, args):
    """Samples that count toward the steady-state metrics.

    Excludes the engage transient (the torque ramp plus a settling allowance)
    and, unless --include-yield, any interval where the re-anchor controller had
    softened the task spring (engage < 1): during a hand-guide the distance from
    the setpoint is the operator's doing, not a tracking failure.
    """
    t = log["t"]
    if args.settle_from is not None:
        t0 = (t[-1] + args.settle_from) if args.settle_from < 0 else args.settle_from
    else:
        t0 = t[0] + args.settle_frac * (t[-1] - t[0])
    mask = t >= t0
    if args.interaction_mode == "track" and args.track_period > 0:
        # The track setpoint oscillates, so a least-squares slope over a partial
        # period reports the sweep itself as drift. Trim the window to a whole
        # number of periods; then the oscillation contributes ~zero slope and
        # what remains is real creep.
        span = t[-1] - t0
        whole = np.floor(span / args.track_period) * args.track_period
        if whole >= args.track_period:
            mask &= t <= (t0 + whole)
    if not args.include_yield and log["engage"] is not None:
        mask &= log["engage"] >= args.engage_min
    return mask, t0


def _rotm(quat):
    """Rotation matrix from an (x,y,z,w) quaternion."""
    x, y, z, w = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tool_tip_error(p_cur, quat_cur, p_des, quat_des, offset):
    """Position error at a tool tip rigidly offset from the controlled frame.

    The impedance controller's task frame is the KDL chain's tip link (the wrist
    flange by default), but the insertion tolerance applies at the TOOL point --
    assembly_tip sits 414 mm out. Over that lever an orientation error dominates:
    1 deg at the flange is 7 mm at the tip. Reporting only the flange error
    therefore flatters any run whose error is rotational.
    """
    off = np.asarray(offset, dtype=float)
    tip_cur = p_cur + np.einsum("nij,j->ni",
                                np.array([_rotm(q) for q in quat_cur]), off)
    tip_des = p_des + np.einsum("nij,j->ni",
                                np.array([_rotm(q) for q in quat_des]), off)
    return np.linalg.norm(tip_cur - tip_des, axis=1)


def drift_per_min(t, y):
    """Least-squares slope of y over t, expressed per minute (0 if degenerate)."""
    if len(t) < 3 or (t[-1] - t[0]) < 1e-6:
        return 0.0
    return float(np.polyfit(t, y, 1)[0] * 60.0)


# --- Reporting ---------------------------------------------------------------
def analyse(path, args):
    log = load_log(path)
    model = imp.KinDynModel(args.urdf, args.base_link, args.tip_link)
    if model.n != log["n"]:
        raise SystemExit(f"{path}: log has {log['n']} joints, model chain has "
                         f"{model.n}; check --base-link/--tip-link.")

    p_cur, quat_cur, p_des, quat_des, events, resid = reconstruct_setpoints(
        log, model, args)

    t = log["t"]
    err_vec = p_cur - p_des                       # world-frame position error
    err_pos = np.linalg.norm(err_vec, axis=1)
    err_ori = np.array([np.linalg.norm(
        imp.orientation_error(quat_des[i], quat_cur[i])) for i in range(len(t))])

    mask, t0 = settled_mask(log, args)
    if mask.sum() < 3:
        # Diagnose WHY the window is empty, in order. The time cut is checked
        # first because if it selected nothing, every later test is being asked
        # about an empty array -- and `np.all([])` is vacuously True, which used
        # to report a run as "entirely YIELDING" when the real problem was that
        # --settle-from had been given a value past the end of the run.
        in_time = t >= t0
        n_time = int(np.count_nonzero(in_time))
        span = t[-1] - t[0]
        if n_time < 3:
            raise SystemExit(
                f"{path}: settled window has {mask.sum()} samples -- the settle "
                f"cut t >= {t0:.1f} s leaves {n_time} of {len(t)} samples, and "
                f"this run is only {span:.1f} s long.\n"
                "--settle-from is an ABSOLUTE time in SECONDS (not a percentage "
                f"and not a fraction). Try --settle-from {max(1.0, 0.3 * span):.0f}, "
                "or a negative value to count back from the end "
                f"(--settle-from {-max(1.0, 0.5 * span):.0f}), or --settle-frac "
                "to give a fraction of the run.")

        why = "loosen --settle-from/--settle-frac"
        if not args.include_yield and log["engage"] is not None:
            eng = log["engage"][in_time]
            n_yield = int(np.count_nonzero(eng < args.engage_min))
            if n_yield == len(eng):
                why = ("the whole window was YIELDING (hand-guided, task spring "
                       "softened) -- this run holds nothing to score; "
                       "--include-yield forces it")
            elif n_yield:
                why = (f"{n_yield} of {n_time} samples in the window were "
                       "YIELDING (hand-guided) and were excluded, leaving too "
                       "few to score; --include-yield forces them in")
        raise SystemExit(f"{path}: settled window has {mask.sum()} samples -- "
                         f"{why}.")

    ts = t[mask]
    ep, eo, ev = err_pos[mask], err_ori[mask], err_vec[mask]

    print("=" * 74)
    print(f"EE TRACKING ANALYSIS: {path}")
    print(f"  mode        : cartesian / {args.interaction_mode}")
    print(f"  duration    : {t[-1] - t[0]:.1f} s, {len(t)} samples "
          f"({len(t) / max(t[-1] - t[0], 1e-3):.0f} Hz logged)")
    print(f"  settled from: t >= {t0:.1f} s  ({mask.sum()} samples scored"
          + ("" if args.include_yield or log["engage"] is None
             else f", yield engage<{args.engage_min} excluded") + ")")

    # --- reconstruction trust ------------------------------------------------
    if resid is None:
        print("  !! log has no 'xerr' column -- the reconstructed setpoint is "
              "UNVALIDATED (older log format).")
    else:
        r = resid[mask]
        verdict = ("OK" if r.max() < args.resid_tol
                   else "SUSPECT -- setpoint reconstruction disagrees with the "
                        "controller")
        print(f"  reconstruct : max |mine - logged xerr| = {r.max()*1000:.2f} mm "
              f"({verdict})")
    if events:
        print(f"  re-anchor   : {len(events)} capture(s) inferred at t = "
              + ", ".join(f"{t[i]:.1f}s" for i in events[:8])
              + (" ..." if len(events) > 8 else ""))

    # --- run status ----------------------------------------------------------
    # A short run, a run that spent most of its life yielding, or one that ended
    # in the collapse-guard excursion is NOT a hold-accuracy characterisation --
    # the settled-window filter would otherwise quietly score the few good
    # samples and report a PASS for a run that ended with the arm 300 mm away.
    warnings = []
    if log["engage"] is not None:
        yield_frac = float(np.mean(log["engage"] < args.engage_min))
        print(f"  yielding    : {yield_frac*100:.0f}% of the run (re-anchor "
              "softened the task spring -- hand-guided, not holding)")
        if yield_frac > args.max_yield_frac:
            warnings.append(f"{yield_frac*100:.0f}% of this run was hand-guided")
    if log["xerr"] is not None and log["xerr"][-1] > args.collapse_err:
        print(f"  !! run ENDS at xerr = {log['xerr'][-1]*1000:.0f} mm -- at or "
              "past the --max-pose-err collapse guard; this looks like an "
              "aborted/guided run, not a hold test.")
        warnings.append("run ended in a collapse-guard excursion")
    settled_span = float(ts[-1] - ts[0])
    if settled_span < args.min_settle_window:
        warnings.append(f"only {settled_span:.1f}s of settled data "
                        f"(< {args.min_settle_window:.0f}s)")

    # --- position ------------------------------------------------------------
    print("-" * 74)
    print("POSITION ERROR (settled)")
    print(f"  mean {np.mean(ep)*1000:7.2f} mm | median {np.median(ep)*1000:7.2f}"
          f" | p95 {np.percentile(ep, 95)*1000:7.2f} | max {ep.max()*1000:7.2f}")
    for k, ax in enumerate("xyz"):
        print(f"    {ax}: bias {np.mean(ev[:, k])*1000:+8.2f} mm   "
              f"sd {np.std(ev[:, k])*1000:6.2f} mm   "
              f"drift {drift_per_min(ts, ev[:, k])*1000:+8.2f} mm/min")
    drift_norm = drift_per_min(ts, ep)
    print(f"  |err| drift : {drift_norm*1000:+.2f} mm/min "
          + ("(static offset -- stiffness/gravity problem)"
             if abs(drift_norm) < args.drift_tol
             else "(CREEPING -- a position drift damping cannot null)"))

    # --- orientation ---------------------------------------------------------
    print("-" * 74)
    print("ORIENTATION ERROR (settled, UNVALIDATED -- no logged reference)")
    print(f"  mean {np.rad2deg(np.mean(eo)):7.2f} deg | p95 "
          f"{np.rad2deg(np.percentile(eo, 95)):7.2f} | max "
          f"{np.rad2deg(eo.max()):7.2f}")
    print(f"  drift       : {np.rad2deg(drift_per_min(ts, eo)):+.2f} deg/min")

    # --- joint drift (is the wrist-anchor fix holding?) ----------------------
    print("-" * 74)
    print("JOINT DRIFT (settled)  * = continuous joint (1,3,5,7): the "
          "dead-zone joints")
    qs = log["q"][mask]
    for j in range(log["n"]):
        star = "*" if (j + 1) in (1, 3, 5, 7) else " "
        d = np.rad2deg(drift_per_min(ts, np.unwrap(qs[:, j])))
        flag = "  <-- creeping" if abs(d) > args.joint_drift_tol else ""
        print(f"  joint {j+1}{star}: {d:+8.2f} deg/min{flag}")

    # --- tool tip ------------------------------------------------------------
    # When a tool offset is given the GATE moves to the tip: that is where the
    # 42 mm bore is, and a flange-only number hides orientation error behind the
    # lever arm.
    tip_p95 = None
    if any(abs(v) > 1e-9 for v in args.tool_offset):
        err_tip = tool_tip_error(p_cur, quat_cur, p_des, quat_des,
                                 args.tool_offset)[mask]
        tip_p95 = float(np.percentile(err_tip, 95))
        lever = float(np.linalg.norm(args.tool_offset))
        print("-" * 74)
        print(f"TOOL-TIP POSITION ERROR (offset {tuple(args.tool_offset)}, "
              f"lever {lever*1000:.0f} mm)")
        print(f"  mean {np.mean(err_tip)*1000:7.2f} mm | p95 "
              f"{tip_p95*1000:7.2f} | max {err_tip.max()*1000:7.2f}")
        print(f"  orientation contributes ~{lever*np.mean(eo)*1000:.1f} mm at "
              f"the mean {np.rad2deg(np.mean(eo)):.1f} deg tilt")

    # --- verdict -------------------------------------------------------------
    pos_p95 = float(np.percentile(ep, 95))
    ori_p95 = float(np.rad2deg(np.percentile(eo, 95)))
    gated_pos = pos_p95 if tip_p95 is None else tip_p95
    ok_pos = gated_pos <= args.gate_pos
    ok_ori = ori_p95 <= args.gate_ori
    ok_drift = abs(drift_norm) <= args.drift_tol
    print("-" * 74)
    print(f"INSERTION GATE (p95 <= {args.gate_pos*1000:.1f} mm / "
          f"{args.gate_ori:.1f} deg, settled)")
    print(f"  {'tool tip' if tip_p95 is not None else 'position'}  p95 "
          f"{gated_pos*1000:7.2f} mm  {'ok' if ok_pos else 'FAIL'}")
    print(f"  orient.   p95 {ori_p95:7.2f} deg {'ok' if ok_ori else 'FAIL'}")
    print(f"  drift     {drift_norm*1000:+9.2f} mm/min {'ok' if ok_drift else 'FAIL'}")
    if warnings:
        # An unsettled or hand-guided run cannot answer the question at all --
        # do not let it read as a pass.
        status = "INCONCLUSIVE"
        print("  => INCONCLUSIVE: " + "; ".join(warnings) + ".")
        print("     Re-run a dedicated hold test: no hand contact, "
              "--reanchor-time 0, --duration 60.")
    elif ok_pos and ok_ori and ok_drift:
        status = "PASS"
        print("  => PASS: accurate enough to drive the insertion phases under "
              "impedance.")
    else:
        status = "FAIL"
        print("  => FAIL: not ready for impedance-mode insertion -- the tool "
              "would miss a 42 mm bore. Fix tracking (tip inertia in the KDL "
              "model, --gravity-trim, --cart-scale) before building the "
              "setpoint channel.")
    print("=" * 74)

    if args.plot:
        make_plot(path, t, err_pos, err_vec, err_ori, log, t0, events)

    return {"path": path, "pos_p95": pos_p95, "ori_p95": ori_p95,
            "pos_mean": float(np.mean(ep)),
            "drift_mm_min": drift_norm * 1000.0,
            "status": status}


def make_plot(path, t, err_pos, err_vec, err_ori, log, t0, events):
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (plot skipped: {e})")
        return

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, err_pos * 1000, "k", lw=1.4, label="|err|")
    for k, c in zip(range(3), "rgb"):
        ax[0].plot(t, err_vec[:, k] * 1000, c, lw=0.8, alpha=0.7, label="xyz"[k])
    ax[0].set_ylabel("position error (mm)")
    ax[0].legend(loc="upper right", ncol=4, fontsize=8)

    ax[1].plot(t, np.rad2deg(err_ori), "m", lw=1.2)
    ax[1].set_ylabel("orientation error (deg)")

    if log["ramp"] is not None:
        ax[2].plot(t, log["ramp"], label="ramp")
    if log["engage"] is not None:
        ax[2].plot(t, log["engage"], label="engage (yield)")
    ax[2].set_ylabel("gain scale")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(loc="lower right", fontsize=8)

    for a in ax:
        a.grid(True, ls="--", alpha=0.5)
        a.axvspan(t[0], t0, color="grey", alpha=0.15)
        for i in events:
            a.axvline(t[i], color="orange", ls=":", lw=1.0)
    ax[0].set_title(f"EE tracking error -- {os.path.basename(path)}  "
                    "(grey = excluded transient, orange = re-anchor)")
    fig.tight_layout()
    png = (path.rsplit(".", 1)[0] if "." in path else path) + "_tracking.png"
    fig.savefig(png, dpi=150)
    print(f"  plot: {png}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="impedance.py --log CSV file(s)")

    # The log does not record the run's arguments, so the mode that produced it
    # must be supplied. Defaults match impedance.py's defaults.
    p.add_argument("--interaction-mode",
                   choices=["hold-ee", "track", "render"], default="hold-ee",
                   help="interaction mode the logged run used (default hold-ee; "
                        "'free' logs have no EE setpoint and cannot be scored)")
    p.add_argument("--track-offset", type=float, nargs=3, default=(0.0, 0.10, 0.0),
                   help="[track] A->B offset the run used (default 0 0.10 0)")
    p.add_argument("--track-period", type=float, default=6.0,
                   help="[track] sweep period in s (default 6)")

    p.add_argument("--urdf", default=imp.DEFAULT_URDF)
    p.add_argument("--base-link", default=imp.DEFAULT_BASE_LINK)
    p.add_argument("--tip-link", default=imp.DEFAULT_TIP_LINK)

    p.add_argument("--settle-from", type=float, default=None, metavar="T",
                   help="score from t=T s; NEGATIVE = last |T| seconds "
                        "(default: use --settle-frac)")
    p.add_argument("--settle-frac", type=float, default=0.75,
                   help="if --settle-from is unset, skip this fraction of the "
                        "run as transient (default 0.75 = score the last 25%%)")
    p.add_argument("--include-yield", action="store_true",
                   help="also score intervals where re-anchor softened the "
                        "spring (default: excluded -- that error is the "
                        "operator's hand, not the controller)")
    p.add_argument("--engage-min", type=float, default=0.99,
                   help="engage column below this counts as yielding (0.99)")

    p.add_argument("--gate-pos", type=float, default=0.003, metavar="M",
                   help="insertion gate on position p95, metres (default 0.003)")
    p.add_argument("--gate-ori", type=float, default=2.0, metavar="DEG",
                   help="insertion gate on orientation p95, deg (default 2)")

    p.add_argument("--reanchor-jump", type=float, default=0.01, metavar="M",
                   help="xerr drop in one sample that marks a re-anchor capture "
                        "(default 0.01 m)")
    p.add_argument("--reanchor-settled", type=float, default=0.02, metavar="M",
                   help="xerr must fall below this for a drop to count (0.02 m)")
    p.add_argument("--resid-tol", type=float, default=0.002, metavar="M",
                   help="max acceptable disagreement between the reconstructed "
                        "and logged xerr (default 0.002 m)")
    p.add_argument("--drift-tol", type=float, default=0.002, metavar="M_PER_MIN",
                   help="|err| slope above which the run is called creeping "
                        "rather than statically offset (default 0.002 m/min)")
    p.add_argument("--joint-drift-tol", type=float, default=1.0, metavar="DEG",
                   help="per-joint drift flagged above this (deg/min, default 1)")
    p.add_argument("--min-settle-window", type=float, default=10.0, metavar="S",
                   help="settled data shorter than this makes the run "
                        "INCONCLUSIVE (default 10 s)")
    p.add_argument("--max-yield-frac", type=float, default=0.25, metavar="F",
                   help="if more than this fraction of the run was hand-guided "
                        "(engage low) the run is INCONCLUSIVE (default 0.25)")
    p.add_argument("--collapse-err", type=float, default=0.25, metavar="M",
                   help="final xerr above this means the run ended in the "
                        "collapse-guard excursion (default 0.25 m)")

    p.add_argument("--tool-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="tool tip offset from the CONTROLLED frame, in that "
                        "frame. Adds a tool-tip error block and moves the "
                        "position gate there -- that is where the bore is. For "
                        "this arm: assembly_tip is -0.027 0 -0.414 from "
                        "bracelet_link. Default 0 0 0 = report the flange only.")
    p.add_argument("--plot", action="store_true", help="write a PNG per log")
    args = p.parse_args()

    results = []
    for path in args.logs:
        if not os.path.exists(path):
            print(f"skip (not found): {path}")
            continue
        try:
            results.append(analyse(path, args))
        except SystemExit as e:
            print(f"skip: {e}")
        print()

    if len(results) > 1:
        print("SUMMARY")
        print(f"{'log':<34} {'pos p95':>9} {'ori p95':>9} {'drift':>12}  gate")
        for r in results:
            print(f"{os.path.basename(r['path']):<34} "
                  f"{r['pos_p95']*1000:7.2f}mm {r['ori_p95']:7.2f}deg "
                  f"{r['drift_mm_min']:+8.2f}mm/m  {r['status']}")


if __name__ == "__main__":
    main()
