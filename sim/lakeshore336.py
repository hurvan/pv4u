#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from p4p.server import Server

from common.pva_common import PVGroup, build_provider_dict

# ----------------------------
# Config / simple plant model
# ----------------------------


@dataclass
class LS336LoopCfg:
    """Per-loop physics + controller defaults."""

    # first-order plant: dT/dt = gain*(u%) - (T - Tamb)/tau
    T_ambient: float = 295.0  # K
    tau_s: float = 25.0  # time constant (s)
    gain_kps_at_100pct: float = 0.5  # K/s at 100% (range=High)
    # PID defaults (RBV initializes from these; writable via *_S#)
    P: float = 10.0
    I: float = 1.0
    D: float = 0.0
    # range scaling for Off/Low/Med/High (fraction of 100%)
    range_scales: Tuple[float, float, float, float] = (0.0, 0.25, 0.5, 1.0)


@dataclass
class LS336Cfg:
    tick_hz: float = 10.0
    noise_sigma_K: float = 0.005
    # four measurement inputs (0..3) and four loops (1..4)
    loops: Dict[int, LS336LoopCfg] = field(
        default_factory=lambda: {
            1: LS336LoopCfg(),  # heater 1
            2: LS336LoopCfg(),  # heater 2
            3: LS336LoopCfg(
                gain_kps_at_100pct=0.0
            ),  # analog out (no plant coupling by default)
            4: LS336LoopCfg(gain_kps_at_100pct=0.0),
        }
    )
    model: str = "LS336"
    firmware: str = "2.90"
    serial: str = "SIM-336-0001"


_RANGE_CHOICES = ["Off", "Low", "Med", "High"]
_MODE_CHOICES = ["PID", "OpenLoop"]  # simplified: PID vs manual (open loop)


# ----------------------------
# Device
# ----------------------------


class LS336:
    """
    Lakeshore 336 simulator with colon-joined PV names, e.g. 'SIM:LS336:KRDG0'.
    Implements a minimal-but-useful subset for GUIs & tests.
    """

    def __init__(
        self, prefix: str, cfg: Optional[LS336Cfg] = None, *, time_fn=time.time
    ):
        self.cfg = cfg or LS336Cfg()
        self.time_fn = time_fn

        # group uses ':' so names look like SYS:DEV:PROP
        self.grp = PVGroup(
            prefix.rstrip(":"),
            default_units="",
            default_precision=3,
            time_fn=self.time_fn,
            write_cb=self._on_put,
            name_join=":",
        )

        # measurements (0..3)
        self._K: Dict[int, float] = {i: 295.0 for i in range(4)}
        self._SR: Dict[int, float] = {
            i: 0.0 for i in range(4)
        }  # raw sensor placeholder
        self._INNAME: Dict[int, str] = {i: f"IN{i}" for i in range(4)}

        # loops (1..4)
        self._setpoint_cmd: Dict[int, float] = {
            i: 300.0 for i in range(1, 5)
        }  # commanded SETP_Si
        self._setpoint_rbv: Dict[int, float] = {
            i: 300.0 for i in range(1, 5)
        }  # ramped SETPi
        self._pid_P: Dict[int, float] = {i: self.cfg.loops[i].P for i in range(1, 5)}
        self._pid_I: Dict[int, float] = {i: self.cfg.loops[i].I for i in range(1, 5)}
        self._pid_D: Dict[int, float] = {i: self.cfg.loops[i].D for i in range(1, 5)}
        self._ramp_enabled: Dict[int, bool] = {i: False for i in range(1, 5)}
        self._ramp_rate: Dict[int, float] = {i: 0.0 for i in range(1, 5)}  # K/min
        self._range_idx: Dict[int, int] = {i: 0 for i in range(1, 5)}  # 0..3 Off..High
        self._mode_idx: Dict[int, int] = {
            i: 0 for i in range(1, 5)
        }  # 0=PID, 1=OpenLoop
        self._manual_out_cmd: Dict[int, float] = {i: 0.0 for i in range(1, 5)}  # %
        self._out_pct_rbv: Dict[int, float] = {i: 0.0 for i in range(1, 5)}  # %
        self._relay_state: Dict[int, int] = {1: 0, 2: 0}

        # per-loop PID integrator / last error for D
        self._I_sum: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._e_prev: Dict[int, float] = {i: 0.0 for i in range(1, 5)}

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._declare_pvs()

    # ----------------------------
    # PV declarations
    # ----------------------------

    def _declare_pvs(self) -> None:
        g = self.grp

        # identity / info
        g.make_str("MODEL", "336", writeable=False)
        g.make_str("FIRMWARE", self.cfg.firmware, writeable=False)
        g.make_str("SERIAL", self.cfg.serial, writeable=False)
        g.make_int(
            "DISABLE_3062", 1, code="h", writeable=True
        )  # present but meaningless here

        # inputs (0..3)
        for i in range(4):
            g.make_float(
                f"KRDG{i}", self._K[i], writeable=False, units="K"
            )  # Temperature (K)
            g.make_float(
                f"SRDG{i}", self._SR[i], writeable=False, units="V"
            )  # Raw sensor
            g.make_str(f"INNAME{i}", self._INNAME[i], writeable=False)
            g.make_str(f"INNAME_S{i}", self._INNAME[i], writeable=True)  # command name

        # loops (1..4)
        for i in range(1, 5):
            # numeric IDs (handy for panels)
            g.make_int(f"OUTPUT{i}", i, code="h", writeable=False)

            # PID RBV + set PVs
            g.make_float(f"P{i}", self._pid_P[i], writeable=False)
            g.make_float(f"I{i}", self._pid_I[i], writeable=False)
            g.make_float(f"D{i}", self._pid_D[i], writeable=False)
            g.make_float(f"P_S{i}", self._pid_P[i], writeable=True)
            g.make_float(f"I_S{i}", self._pid_I[i], writeable=True)
            g.make_float(f"D_S{i}", self._pid_D[i], writeable=True)

            # setpoint w/ ramping
            g.make_float(
                f"SETP{i}", self._setpoint_rbv[i], writeable=False, units="K"
            )  # RBV
            g.make_float(
                f"SETP_S{i}", self._setpoint_cmd[i], writeable=True, units="K"
            )  # command
            g.make_float(
                f"RAMP{i}", self._ramp_rate[i], writeable=False, units="K/min"
            )  # RBV
            g.make_float(
                f"RAMP_S{i}", self._ramp_rate[i], writeable=True, units="K/min"
            )  # K/min
            g.make_enum(
                f"RAMPST{i}",
                ["Off", "On"],
                init_index=int(self._ramp_enabled[i]),
                writeable=False,
            )
            g.make_enum(
                f"RAMPST_S{i}",
                ["Off", "On"],
                init_index=int(self._ramp_enabled[i]),
                writeable=True,
            )

            # output mode + manual output
            # Use the OMM/OMI/OMP names you provided; OMM is the meaningful one here
            g.make_enum(
                f"OMM{i}",
                ["PID", "OpenLoop"],
                init_index=self._mode_idx[i],
                writeable=False,
            )
            g.make_enum(
                f"OMM_S{i}",
                ["PID", "OpenLoop"],
                init_index=self._mode_idx[i],
                writeable=True,
            )
            g.make_int(f"OMI{i}", 0, code="h", writeable=False)
            g.make_int(f"OMI_S{i}", 0, code="h", writeable=True)
            g.make_int(f"OMP{i}", 0, code="h", writeable=False)
            g.make_int(f"OMP_S{i}", 0, code="h", writeable=True)

            g.make_float(
                f"MOUT{i}", self._out_pct_rbv[i], writeable=False, units="%"
            )  # %
            g.make_float(
                f"MOUT_S{i}", self._manual_out_cmd[i], writeable=True, units="%"
            )

            # range (Off/Low/Med/High)
            g.make_enum(
                f"RANGE{i}",
                _RANGE_CHOICES,
                init_index=self._range_idx[i],
                writeable=False,
            )
            g.make_enum(
                f"RANGE_S{i}",
                _RANGE_CHOICES,
                init_index=self._range_idx[i],
                writeable=True,
            )

            # tuning mode placeholder
            g.make_int(f"TUNEMODE_S{i}", 0, code="h", writeable=True)

        # convenience aliases
        g.make_float(
            "HTR1", self._out_pct_rbv[1], writeable=False, units="%"
        )  # heater 1 %
        g.make_float(
            "HTR2", self._out_pct_rbv[2], writeable=False, units="%"
        )  # heater 2 %

        # unpowered analog outputs 3/4 in volts, 0..10 V from % (RBV only; command via MOUT_S3/4 + OpenLoop)
        g.make_float("AOUT3", 0.0, writeable=False, units="V")
        g.make_float("AOUT4", 0.0, writeable=False, units="V")

        # relays
        g.make_enum(
            "RELAY1", ["Off", "On"], init_index=self._relay_state[1], writeable=False
        )
        g.make_enum(
            "RELAY2", ["Off", "On"], init_index=self._relay_state[2], writeable=False
        )
        g.make_enum(
            "RELAYST1", ["Off", "On"], init_index=self._relay_state[1], writeable=True
        )
        g.make_enum(
            "RELAYST2", ["Off", "On"], init_index=self._relay_state[2], writeable=True
        )

    # ----------------------------
    # PUT handling
    # ----------------------------

    def _on_put(self, suffix: str, value):
        # input names
        if suffix.startswith("INNAME_S"):
            i = int(suffix.replace("INNAME_S", ""))
            self._INNAME[i] = str(value)
            self.grp.post_num(f"INNAME{i}", self._INNAME[i])  # string posts accepted
            return

        # relays
        if suffix in ("RELAYST1", "RELAYST2"):
            idx = 1 if suffix.endswith("1") else 2
            self._relay_state[idx] = 1 if int(value) else 0
            self.grp.post_num(f"RELAYST{idx}", self._relay_state[idx])
            self.grp.post_num(f"RELAY{idx}", self._relay_state[idx])
            return

        # per-loop controls
        for i in range(1, 5):
            if suffix == f"SETP_S{i}":
                self._setpoint_cmd[i] = float(value)
                # if ramp disabled => snap now
                if not self._ramp_enabled[i] or self._ramp_rate[i] <= 0.0:
                    self._setpoint_rbv[i] = self._setpoint_cmd[i]
                    self.grp.post_num(f"SETP{i}", self._setpoint_rbv[i])
                return

            if suffix == f"P_S{i}":
                self._pid_P[i] = max(0.0, float(value))
                self.grp.post_num(f"P{i}", self._pid_P[i])
                return
            if suffix == f"I_S{i}":
                self._pid_I[i] = max(0.0, float(value))
                self.grp.post_num(f"I{i}", self._pid_I[i])
                return
            if suffix == f"D_S{i}":
                self._pid_D[i] = max(0.0, float(value))
                self.grp.post_num(f"D{i}", self._pid_D[i])
                return

            if suffix == f"RAMP_S{i}":
                self._ramp_rate[i] = max(0.0, float(value))
                self.grp.post_num(f"RAMP{i}", self._ramp_rate[i])
                return
            if suffix == f"RAMPST_S{i}":
                self._ramp_enabled[i] = bool(int(value))
                self.grp.post_num(f"RAMPST{i}", int(self._ramp_enabled[i]))
                return

            if suffix == f"OMM_S{i}":
                idx = int(value)  # 0=PID, 1=OpenLoop
                idx = 0 if idx <= 0 else 1
                self._mode_idx[i] = idx
                self.grp.post_num(f"OMM{i}", idx)
                return

            if suffix == f"MOUT_S{i}":
                self._manual_out_cmd[i] = float(value)
                # clamp here; RBV follows in run()
                return

            if suffix == f"RANGE_S{i}":
                idx = int(value)
                idx = min(max(idx, 0), len(_RANGE_CHOICES) - 1)
                self._range_idx[i] = idx
                self.grp.post_num(f"RANGE{i}", idx)
                return

        # misc toggles
        if suffix == "DISABLE_3062":
            # do nothing (placeholder for a scanner option present in real HW)
            return

    # ----------------------------
    # Helpers
    # ----------------------------

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _step_setpoint_ramping(self, dt: float) -> None:
        """Move RBV setpoint toward commanded setpoint at RAMP_S (K/min) if enabled."""
        for i in range(1, 5):
            if not self._ramp_enabled[i] or self._ramp_rate[i] <= 0.0:
                self._setpoint_rbv[i] = self._setpoint_cmd[i]
                continue
            step = (self._ramp_rate[i] / 60.0) * dt
            cur = self._setpoint_rbv[i]
            tgt = self._setpoint_cmd[i]
            if math.isclose(cur, tgt, abs_tol=1e-9):
                self._setpoint_rbv[i] = tgt
            else:
                sgn = 1.0 if (tgt > cur) else -1.0
                nxt = cur + sgn * step
                if (sgn > 0 and nxt > tgt) or (sgn < 0 and nxt < tgt):
                    nxt = tgt
                self._setpoint_rbv[i] = nxt
            self.grp.post_num(f"SETP{i}", self._setpoint_rbv[i])

    def _pid_compute(self, i: int, T_meas: float, dt: float) -> float:
        """Return output percent 0..100 from PID (saturated with Range)."""
        e = self._setpoint_rbv[i] - T_meas
        P = self._pid_P[i] * e
        self._I_sum[i] += e * dt * self._pid_I[i]
        # guard integral windup a bit
        self._I_sum[i] = self._clamp(self._I_sum[i], -200.0, 200.0)
        D = 0.0
        if dt > 0.0 and self._pid_D[i] > 0.0:
            D = self._pid_D[i] * (e - self._e_prev[i]) / dt
        self._e_prev[i] = e

        u = P + self._I_sum[i] + D
        # pretend "controller counts" map to % aggressively
        u_pct = self._clamp(u, 0.0, 100.0)
        return u_pct

    def _range_scale(self, i: int) -> float:
        return self.cfg.loops[i].range_scales[self._range_idx[i]]

    def _step_outputs_and_plant(self, dt: float) -> None:
        # compute output % per loop
        for i in range(1, 5):
            # clamp manual cmd; RBV computed below
            self._manual_out_cmd[i] = self._clamp(self._manual_out_cmd[i], 0.0, 100.0)

            if (
                self._mode_idx[i] == 1
            ):  # OpenLoop => follow manual cmd (subject to range)
                u = self._manual_out_cmd[i]
            else:
                # PID mode
                sensor_index = (
                    0 if i in (1, 3) else 1
                )  # simple: loop1/3 read IN0; loop2/4 read IN1
                T_meas = self._K[sensor_index]
                u = self._pid_compute(i, T_meas, dt)

            # enforce range scaling
            u *= self._range_scale(i)
            u = self._clamp(u, 0.0, 100.0)
            self._out_pct_rbv[i] = u
            self.grp.post_num(f"MOUT{i}", u)
            if i == 1:
                self.grp.post_num("HTR1", u)
            elif i == 2:
                self.grp.post_num("HTR2", u)

        # analog outs 3/4 in 0..10 V (linear from %)
        self.grp.post_num("AOUT3", (self._out_pct_rbv[3] / 100.0) * 10.0)
        self.grp.post_num("AOUT4", (self._out_pct_rbv[4] / 100.0) * 10.0)

        # simple plant for inputs 0/1 driven by loops 1/2 respectively
        for pair in [(0, 1), (1, 2)]:
            in_idx, loop_idx = pair
            lp = self.cfg.loops[loop_idx]
            T = self._K[in_idx]
            u_frac = self._out_pct_rbv[loop_idx] / 100.0
            # dT/dt = +gain*u - (T - Tamb)/tau
            dT = (lp.gain_kps_at_100pct * u_frac) - (T - lp.T_ambient) / max(
                lp.tau_s, 1e-6
            )
            T_next = T + dT * dt
            # add a sniff of noise
            T_next += random.gauss(0.0, self.cfg.noise_sigma_K)
            self._K[in_idx] = T_next
            self.grp.post_num(f"KRDG{in_idx}", T_next)

        # SRDG raw "voltage" placeholders, just proportional to (T-ambient)
        for i in range(4):
            self._SR[i] = (self._K[i] - 295.0) * 0.01
            self.grp.post_num(f"SRDG{i}", self._SR[i])

    # ----------------------------
    # Runner
    # ----------------------------

    def _run(self):
        next_tick = self.time_fn()
        while not self._stop_evt.is_set():
            now = self.time_fn()
            dt = max(0.0, now - next_tick + (1.0 / self.cfg.tick_hz))

            self._step_setpoint_ramping(dt)
            self._step_outputs_and_plant(dt)

            next_tick += 1.0 / self.cfg.tick_hz
            time.sleep(max(0.0, next_tick - self.time_fn()))

    def start(self):
        if not self._thread.is_alive():
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


# ----------------------------
# CLI
# ----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Lakeshore 336 PVA simulator (colon-joined PV names)."
    )
    ap.add_argument("--prefix", default="SIM:LS336:", help="e.g. 'SIM:LS336:'")
    ap.add_argument("--tick-hz", type=float, default=10.0, help="Simulation tick")
    ap.add_argument("--amb", type=float, default=295.0, help="Ambient K")
    ap.add_argument("--gain1", type=float, default=0.5, help="Loop1 K/s at 100%%")
    ap.add_argument("--gain2", type=float, default=0.5, help="Loop2 K/s at 100%%")
    ap.add_argument("--tau1", type=float, default=25.0, help="Loop1 tau s")
    ap.add_argument("--tau2", type=float, default=25.0, help="Loop2 tau s")
    args = ap.parse_args()

    cfg = LS336Cfg(tick_hz=args.tick_hz)
    cfg.loops[1].T_ambient = args.amb
    cfg.loops[2].T_ambient = args.amb
    cfg.loops[3].T_ambient = args.amb
    cfg.loops[4].T_ambient = args.amb
    cfg.loops[1].gain_kps_at_100pct = args.gain1
    cfg.loops[2].gain_kps_at_100pct = args.gain2
    cfg.loops[1].tau_s = args.tau1
    cfg.loops[2].tau_s = args.tau2

    dev = LS336(args.prefix, cfg).start()
    providers = build_provider_dict({args.prefix.rstrip(":"): dev.grp})

    print(f"[sim-ls336] Serving at '{args.prefix}*'.")
    print(f"  - Inputs: KRDG0..3  (Kelvin), SRDG0..3")
    print(
        f"  - Loops:  SETP1..4 / SETP_S1..4 + PID P/I/D, RANGE, RAMP/RAMPST, MOUT/MOUT_S"
    )
    print(
        f"  - Heaters: HTR1, HTR2 (aliases of MOUT1/2). AOUT3/4 reflect MOUT3/4 as 0..10 V."
    )
    print("Ctrl+C to exit.")
    try:
        with Server(providers=[providers]) as S:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[sim-ls336] Shutting down...")
    finally:
        dev.stop()


if __name__ == "__main__":
    main()
