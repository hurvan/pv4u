#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from p4p.server import Server

from common.pva_common import PVGroup, build_provider_dict


@dataclass
class ChopperConfig:
    egu_speed: str = "Hz"
    egu_angle: str = "degrees"
    egu_delay_ns: str = "ns"
    precision: int = 3

    accel_hzps: float = 5.0  # constant accel (|dHz/ds|)
    park_vel_degps: float = 90.0  # parking slew velocity
    jitter_ns_rms: float = 200.0  # TDC jitter (RMS)
    evr_flush_hz: float = 14.0  # EVR flush rate (Hz)

    # lock/phase settings
    pll_gain: float = 0.25
    lock_thr_us: float = 5.0
    lock_acq_count: int = 3
    lock_loss_us: float = 8.0
    lock_loss_count: int = 1
    phase_err_window: int = 16

    # GUI offsets
    resolver_offset_deg: float = 0.0
    tdc_offset_deg: float = 0.0

    # delay composition
    chop_delay_ns: float = 0.0
    mech_delay_deg: float = 0.0
    beampos_delay_ns: float = 0.0

    # rotation sense (for mirrored motors in dual-disc stacks)
    rot_sense_index: int = 0  # 0=CW_Positive, 1=CCW_Positive


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def now_ns() -> int:
    return time.time_ns()


class EssChopper:
    # states and enums
    STATE_CHOICES = [
        "Parked",
        "Parking",
        "Ramping",
        "Spinning",
        "Locking",
        "Locked",
        "Coasting",
        "Fault",
    ]
    EXEC_CHOICES = ["Start", "Stop", "ClearAlarms"]
    PARK_CHOICES = ["Open", "Close"] + [f"Window {i}" for i in range(1, 11)]
    PARK_OPEN = 15.0  # degrees
    PARK_CLOSE = 195.0  # degrees
    ROT_SENSE_CHOICES = ["CW_Positive", "CCW_Positive"]
    ALARM_SUFFIXES = [
        "Comm_Alrm",
        "HW_Alrm",
        "IntLock_Alrm",
        "Lvl_Alrm",
        "Pos_Alrm",
        "Pwr_Alrm",
        "Ref_Alrm",
        "SW_Alrm",
        "Volt_Alrm",
    ]

    def __init__(
        self, prefix: str, cfg: Optional[ChopperConfig] = None, *, time_fn=time.time
    ):
        self.cfg = cfg or ChopperConfig()
        self.prefix = prefix.rstrip(":")
        self.time_fn = time_fn

        # core state
        self._spd_s_hz: float = 0.0
        self._spd_r_hz: float = 0.0
        self._spin_enable: bool = False
        self._resolver_deg: float = 0.0
        self._target_park_deg: float = 0.0
        self._park_mode_idx: int = 0
        self._state_idx: int = 0
        self._park_pos_map: Dict[int, float] = {
            i: 0.0 for i in range(len(self.PARK_CHOICES))
        }
        self._park_pos_map[0] = self.PARK_OPEN
        self._park_pos_map[1] = self.PARK_CLOSE

        # delays / EVR
        self._chop_dly_ns: float = self.cfg.chop_delay_ns
        self._mech_dly_deg: float = self.cfg.mech_delay_deg
        self._beampos_dly_ns: float = self.cfg.beampos_delay_ns
        self._phase_target_ns: float = (
            0.0  # absolute desired time offset (ns) rotor->EVR drive pulse
        )

        # EVR timing
        self._evr_period_ns: int = int(
            round(1e9 / self.cfg.evr_flush_hz)
        )  # fixed flush frame period
        self._last_flush_ns: int = now_ns()
        self._evr_drive_hz: float = 14.0  # EVR drive output (follows Spd_S)
        self._evr_phase_ns: float = float(
            now_ns()
        )  # absolute phase origin (ns) of EVR drive pulses

        # TDC state
        self._last_tdc_ns: Optional[int] = None

        # locking + history
        self._phase_err_hist: List[float] = []
        self._consec_in: int = 0
        self._consec_out: int = 0
        self._diff_ns_samples = deque(maxlen=1000)
        self._diff_publish_every = 100  # post every 100 new errors
        self._diff_since_pub = 0

        self._rot_sense_idx: int = clamp(self.cfg.rot_sense_index, 0, 1)
        self._alarms_active = {s: False for s in self.ALARM_SUFFIXES}

        # loop
        self._tick = 1.0 / 50.0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # PVs (NOTE name_join=':' so we create SYS:DEV:PROP, not SYS:DEV.PROP)
        self.grp = PVGroup(
            prefix,
            default_units="",
            default_precision=self.cfg.precision,
            write_cb=self._on_put,
            time_fn=self.time_fn,
            name_join=":",
        )
        self._declare_pvs()
        self._compute_totdly()

    # ---- PVs ----

    def _declare_pvs(self):
        g = self.grp

        # commands/state
        g.make_enum("C_Execute", self.EXEC_CHOICES, init_index=1, writeable=True)
        g.make_enum(
            "ChopState_R",
            self.STATE_CHOICES,
            init_index=self._state_idx,
            writeable=False,
        )

        # speed + accel
        g.make_float(
            "Spd_S", 0.0, writeable=True, units="Hz"
        )  # setpoint (does NOT start)
        g.make_float("Spd_R", 0.0, writeable=False, units="Hz")  # readback
        g.add_mdel_fields("Spd_R")
        g.make_float("ACCEL_HZPS", self.cfg.accel_hzps, writeable=True)

        # rotation sense + dir readback
        g.make_enum(
            "ROT_SENSE",
            self.ROT_SENSE_CHOICES,
            init_index=self._rot_sense_idx,
            writeable=True,
        )
        g.make_str("Dir_R", "CW", writeable=False)

        # parking
        g.make_enum("ParkPos_S", self.PARK_CHOICES, init_index=0, writeable=True)
        g.make_float(
            "Park_S", 0.0, writeable=True, units="degrees"
        )  # only writable in "Window N"
        g.make_float(
            "Pos_R", 0.0, writeable=False, units="degrees"
        )  # resolver + GUI offset
        g.make_float("PARK_VELO_DPS", self.cfg.park_vel_degps, writeable=True)

        # delay composition
        g.make_float("ChopDly-S", self._chop_dly_ns, writeable=True, units="ns")
        g.make_float("MechDly-S", self._mech_dly_deg, writeable=True, units="degrees")
        g.make_float("BeamPosDly-S", self._beampos_dly_ns, writeable=True, units="ns")
        g.make_float("TotDly", 0.0, writeable=False, units="ns")

        # lock/phase
        g.make_int("InPhs_R", 0, code="h", writeable=False)
        g.make_float("PHASE_ERR_US", 0.0, writeable=False)
        g.make_float("LOCK_THR_US", self.cfg.lock_thr_us, writeable=True)
        g.make_int("LOCK_ACQ_COUNT", self.cfg.lock_acq_count, code="h", writeable=True)
        g.make_float("LOCK_LOSS_US", self.cfg.lock_loss_us, writeable=True)
        g.make_int(
            "LOCK_LOSS_COUNT", self.cfg.lock_loss_count, code="h", writeable=True
        )
        g.make_float("PLL_GAIN", self.cfg.pll_gain, writeable=True)

        # jitter + GUI offsets
        g.make_float("JITTER_NS_RMS", self.cfg.jitter_ns_rms, writeable=True)
        g.make_float(
            "RESOLVER_OFFSET_DEG", self.cfg.resolver_offset_deg, writeable=True
        )
        g.make_float("TDC_OFFSET_DEG", self.cfg.tdc_offset_deg, writeable=True)

        # EVR / TDC list (absolute epoch ns, flush at fixed rate)
        g.make_float(
            "TSFlushRate_R", float(self.cfg.evr_flush_hz), writeable=False, units="Hz"
        )
        g.make_array_int64("02-TS-I", init=[], writeable=False)
        g.make_array_int64("DiffTSSamples", init=[], writeable=False)

        # alarms
        for sfx in self.ALARM_SUFFIXES:
            g.make_int(sfx, 0, code="h", writeable=False)
        g.make_enum(
            "FaultInject", ["None"] + self.ALARM_SUFFIXES, init_index=0, writeable=True
        )

    # ---- PUT handling ----

    def _on_put(self, suffix: str, value):
        if suffix == "C_Execute":
            cmd = self.EXEC_CHOICES[int(value)]
            if cmd == "Start":
                self._spin_enable = True
                if abs(self._spd_s_hz) > 0.0:
                    self._transition_state("Ramping")
            elif cmd == "Stop":
                self._spin_enable = False
                self._spd_s_hz = 0.0
                self.grp.post_num("Spd_S", 0.0)
                self._transition_state(
                    "Coasting" if abs(self._spd_r_hz) > 0.1 else "Parked"
                )
            elif cmd == "ClearAlarms":
                for k in list(self._alarms_active):
                    self._alarms_active[k] = False
                    self.grp.post_num(k, 0, severity=0, message="")
        elif suffix == "Spd_S":
            # setpoint only; does NOT auto-start
            self._spd_s_hz = float(value)
        elif suffix == "ACCEL_HZPS":
            self.cfg.accel_hzps = max(0.01, float(value))
        elif suffix == "ROT_SENSE":
            self._rot_sense_idx = int(value)
        elif suffix == "ParkPos_S":
            self._park_mode_idx = int(value)
            # set the target stored in the map
            self._target_park_deg = self._park_pos_map[self._park_mode_idx]
            if abs(self._spd_r_hz) < 0.01:
                self._transition_state("Parking")
        elif suffix == "Park_S":
            if self._park_mode_idx < 2:  # Open/Close: read-only
                self._trip_alarm("Pos_Alrm", "Park_S is read-only in Open/Close")
            else:
                self._target_park_deg = float(value) % 360.0
                self._park_pos_map[self._park_mode_idx] = self._target_park_deg
                if abs(self._spd_r_hz) < 0.01:
                    self._transition_state("Parking")
        elif suffix == "PARK_VELO_DPS":
            self.cfg.park_vel_degps = max(1.0, float(value))
        elif suffix == "ChopDly-S":
            self._chop_dly_ns = float(value)
        elif suffix == "MechDly-S":
            self._mech_dly_deg = float(value)
        elif suffix == "BeamPosDly-S":
            self._beampos_dly_ns = float(value)
        elif suffix == "LOCK_THR_US":
            self.cfg.lock_thr_us = max(0.0, float(value))
        elif suffix == "LOCK_ACQ_COUNT":
            self.cfg.lock_acq_count = max(1, int(value))
        elif suffix == "LOCK_LOSS_US":
            self.cfg.lock_loss_us = max(self.cfg.lock_thr_us, float(value))
        elif suffix == "LOCK_LOSS_COUNT":
            self.cfg.lock_loss_count = max(1, int(value))
        elif suffix == "PLL_GAIN":
            self.cfg.pll_gain = float(value)
        elif suffix == "JITTER_NS_RMS":
            self.cfg.jitter_ns_rms = max(0.0, float(value))
        elif suffix == "RESOLVER_OFFSET_DEG":
            self.cfg.resolver_offset_deg = float(value)
        elif suffix == "TDC_OFFSET_DEG":
            self.cfg.tdc_offset_deg = float(value)
        elif suffix == "FaultInject":
            idx = int(value)
            names = ["None"] + self.ALARM_SUFFIXES
            name = names[idx]
            if name == "None":
                for k in list(self._alarms_active):
                    self._alarms_active[k] = False
                    self.grp.post_num(k, 0, severity=0, message="")
            else:
                self._trip_alarm(name, "Injected fault")

        self._compute_totdly()

    # ---- helpers ----

    def _wrap_to_period(self, x_ns: float, P_ns: float) -> float:
        """Wrap signed nanoseconds to [-P/2, +P/2)."""
        return ((x_ns + 0.5 * P_ns) % P_ns) - 0.5 * P_ns

    def _transition_state(self, name: str):
        if name in self.STATE_CHOICES:
            self._state_idx = self.STATE_CHOICES.index(name)
            self.grp.post_num("ChopState_R", self._state_idx)

    def _trip_alarm(self, suffix: str, message: str, severity: int = 2):
        self._alarms_active[suffix] = True
        self.grp.post_num(suffix, 1, severity=severity, message=message)
        self._transition_state("Fault")

    def _rev_period_ns(self) -> Optional[float]:
        f = abs(self._spd_r_hz if self._spd_r_hz != 0.0 else self._spd_s_hz)
        if f <= 0.0:
            return None
        return 1e9 / f

    def _compute_totdly(self):
        """Absolute desired time offset (ns) rotor→EVR drive pulse."""
        spd_mag = abs(self._spd_s_hz)
        mech_ns = (
            0.0 if spd_mag <= 0.0 else (self._mech_dly_deg / 360.0) * (1e9 / spd_mag)
        )
        self._phase_target_ns = float(
            self._chop_dly_ns + self._beampos_dly_ns + mech_ns
        )
        self.grp.post_num("TotDly", self._phase_target_ns)

    def _update_dir_r(self):
        if self._spd_r_hz == 0.0:
            d = "CW"
        else:
            phys_cw = (self._spd_r_hz > 0.0 and self._rot_sense_idx == 0) or (
                self._spd_r_hz < 0.0 and self._rot_sense_idx == 1
            )
            d = "CW" if phys_cw else "CCW"
        # string PV; posting bare string is fine
        self.grp.pvs["Dir_R"].post(d)

    def _ramp_speed(self, dt: float, to_target: float):
        v = float(self._spd_r_hz)
        if math.isclose(v, to_target, abs_tol=1e-6):
            self._spd_r_hz = to_target
            return
        dv = self.cfg.accel_hzps * dt
        if to_target > v:
            v = min(to_target, v + dv)
        else:
            v = max(to_target, v - dv)
        self._spd_r_hz = v
        self.grp.post_num("Spd_R", v)

    def _advance_resolver(self, dt: float):
        if abs(self._spd_r_hz) > 0.01:
            self._resolver_deg = (
                self._resolver_deg + 360.0 * self._spd_r_hz * dt
            ) % 360.0
        else:
            if self._state_idx == self.STATE_CHOICES.index("Parking"):
                cur = self._resolver_deg
                tgt = self._target_park_deg
                if math.isclose(cur, tgt, abs_tol=1e-3):
                    self._resolver_deg = tgt
                    self._transition_state("Parked")
                else:
                    step = self.cfg.park_vel_degps * dt
                    diff = ((tgt - cur + 540.0) % 360.0) - 180.0
                    if abs(diff) <= step:
                        self._resolver_deg = tgt
                        self._transition_state("Parked")
                    else:
                        self._resolver_deg = (
                            cur + (step if diff > 0 else -step)
                        ) % 360.0
        disp = (self._resolver_deg + self.cfg.resolver_offset_deg) % 360.0
        # only post if speed is below 2hz
        if abs(self._spd_r_hz) < 2.0:
            self.grp.post_num("Pos_R", disp)

    def _evr_frame_t0_ns(self, t_ns: int) -> int:
        P = self._evr_period_ns
        return (t_ns // P) * P

    def _update_evr_drive_hz(self):
        """Quantize EVR drive to integer multiples of the base (>= 1x)."""
        base = float(self.cfg.evr_flush_hz)  # 14 Hz base
        hz = abs(self._spd_s_hz)
        if hz <= 0.0:
            mult = 1
        else:
            mult = max(1, int(round(hz / base)))
        self._evr_drive_hz = base * mult

    # ---- TDC generation (rotor-driven, independent of EVR) ----
    def _simulate_tdc_flush(self, frame_t0_ns: int) -> List[int]:
        """Generate rotor TDC timestamps inside [frame_t0, frame_t0+P)."""
        P = self._evr_period_ns
        start = frame_t0_ns
        end = frame_t0_ns + P
        out: List[int] = []

        f = abs(self._spd_r_hz)
        if f <= 0.0:
            return out

        rev_ns = 1e9 / f

        # continue from previous TDC, or start just before frame start
        if self._last_tdc_ns is None:
            # ensure first >= start
            t = int(start - rev_ns)
        else:
            t = int(self._last_tdc_ns)

        while t < start:
            t += int(rev_ns)

        while t < end:
            jitter = (
                int(random.gauss(0.0, self.cfg.jitter_ns_rms))
                if self.cfg.jitter_ns_rms > 0.0
                else 0
            )
            out.append(int(t + jitter))
            t += int(rev_ns)

        if out:
            self._last_tdc_ns = out[-1]
        return out

    def _maybe_publish_diff_samples(self, added: int = 0, force: bool = False):
        self._diff_since_pub += added
        if force or self._diff_since_pub >= self._diff_publish_every:
            # send a snapshot of the last 1000 errors
            self.grp.post_array("DiffTSSamples", list(self._diff_ns_samples))
            self._diff_since_pub = 0

    # ---- Lock/PLL against nearest EVR drive pulse ----
    def _update_lock_metrics(self, frame_t0_ns: int, tdcs: List[int]):
        if not tdcs:
            return

        self._update_evr_drive_hz()
        Tdrv = 1e9 / max(1e-6, abs(self._evr_drive_hz))

        rev_ns_opt = self._rev_period_ns()
        gui_ns = (
            0.0
            if rev_ns_opt is None
            else (self.cfg.tdc_offset_deg / 360.0) * rev_ns_opt
        )
        tau_ns = float(self._phase_target_ns + gui_ns)

        best_err = None
        added = 0  # <— count how many samples we appended this frame

        for ts in tdcs:
            k = int(round((ts - self._evr_phase_ns) / Tdrv))
            evr_ts = self._evr_phase_ns + k * Tdrv
            err = (
                self._wrap_to_period((ts - evr_ts) - tau_ns, Tdrv)
                + random.gauss(0.0, self.cfg.jitter_ns_rms)
                if self.cfg.jitter_ns_rms > 0.0
                else 0.0
            )

            self._phase_err_hist.append(err)
            if len(self._phase_err_hist) > self.cfg.phase_err_window:
                self._phase_err_hist.pop(0)

            self._diff_ns_samples.append(int(round(err)))
            added += 1

            if best_err is None or abs(err) < abs(best_err):
                best_err = err

        # Throttled publish
        self._maybe_publish_diff_samples(added=added)

        if best_err is None:
            return

        corr = clamp(self.cfg.pll_gain * best_err, -0.5 * Tdrv, 0.5 * Tdrv)
        self._evr_phase_ns = (self._evr_phase_ns + corr) % Tdrv

        abs_us = abs(best_err) / 1000.0
        self.grp.post_num("PHASE_ERR_US", abs_us)

        if abs_us <= self.cfg.lock_thr_us:
            self._consec_in, self._consec_out = self._consec_in + 1, 0
        elif abs_us >= self.cfg.lock_loss_us:
            self._consec_out, self._consec_in = self._consec_out + 1, 0

        locked = int(self.grp.pvs["InPhs_R"].current())
        if not locked and self._consec_in >= self.cfg.lock_acq_count:
            self.grp.post_num("InPhs_R", 1)
            self._transition_state("Locked")
        elif locked and self._consec_out >= self.cfg.lock_loss_count:
            self.grp.post_num("InPhs_R", 0)
            if abs(self._spd_r_hz) > 0.1:
                self._transition_state("Locking")

    # ---- thread ----

    def _run(self):
        next_tick = self.time_fn()
        next_flush = self._last_flush_ns + self._evr_period_ns
        while not self._stop_evt.is_set():
            now = self.time_fn()
            dt = max(0.0, now - next_tick + self._tick)

            # ramping
            if self._spin_enable:
                self._ramp_speed(dt, self._spd_s_hz)
            else:
                if not math.isclose(self._spd_r_hz, 0.0, abs_tol=1e-6):
                    self._ramp_speed(dt, 0.0)

            self._update_dir_r()

            # high-level state
            if abs(self._spd_r_hz) > 0.1:
                if self._spin_enable:
                    if math.isclose(self._spd_r_hz, self._spd_s_hz, abs_tol=0.05):
                        if self._state_idx != self.STATE_CHOICES.index("Locked"):
                            self._transition_state("Locking")
                    else:
                        self._transition_state("Ramping")
                else:
                    self._transition_state("Coasting")
            else:
                if self._state_idx in (
                    self.STATE_CHOICES.index("Ramping"),
                    self.STATE_CHOICES.index("Locking"),
                    self.STATE_CHOICES.index("Spinning"),
                    self.STATE_CHOICES.index("Coasting"),
                    self.STATE_CHOICES.index("Locked"),
                ):
                    self._transition_state(
                        "Parking"
                        if not math.isclose(
                            self._resolver_deg, self._target_park_deg, abs_tol=1e-3
                        )
                        else "Parked"
                    )

            self._advance_resolver(dt)

            # EVR flush frame
            t_ns = now_ns()
            if t_ns >= next_flush:
                frame_t0 = self._evr_frame_t0_ns(t_ns)
                tdcs = self._simulate_tdc_flush(frame_t0)
                self.grp.post_array("02-TS-I", tdcs)
                self._update_lock_metrics(frame_t0, tdcs)
                self._last_flush_ns = frame_t0
                next_flush = frame_t0 + self._evr_period_ns

            # keep alarms asserted
            for name, active in self._alarms_active.items():
                if active:
                    self.grp.post_num(name, 1, severity=2, message="Active")

            next_tick += self._tick
            time.sleep(max(0.0, next_tick - self.time_fn()))

    # ---- lifecycle ----

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


class ChicLink:
    def __init__(self, chic_prefix: str, *, time_fn=time.time):
        # CHIC uses colon-joined record names too
        self.grp = PVGroup(
            chic_prefix,
            default_units="",
            default_precision=0,
            time_fn=time_fn,
            name_join=":",
        )
        self.grp.make_str("ConnectedR", "Connected", writeable=True)


def main():
    ap = argparse.ArgumentParser(
        description="ESS chopper PVA simulator (records use colon, not fields)"
    )
    ap.add_argument("--prefix", default="SIM:CHP1:", help="e.g. 'SIM:CHP1:'")
    ap.add_argument("--chic-prefix", default="SIM:CHIC1:", help="e.g. 'SIM:CHIC1:'")
    ap.add_argument("--accel", type=float, default=5.0)
    ap.add_argument("--park-vel", type=float, default=90.0)
    ap.add_argument("--jitter-ns", type=float, default=200.0)
    ap.add_argument("--flush-hz", type=float, default=14.0)
    ap.add_argument("--lock-thr-us", type=float, default=5.0)
    ap.add_argument("--rot-sense", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    cfg = ChopperConfig(
        accel_hzps=args.accel,
        park_vel_degps=args.park_vel,
        jitter_ns_rms=args.jitter_ns,
        evr_flush_hz=args.flush_hz,
        lock_thr_us=args.lock_thr_us,
        rot_sense_index=args.rot_sense,
    )
    ch = EssChopper(args.prefix, cfg).start()
    chic = ChicLink(args.chic_prefix) if args.chic_prefix else None

    groups: Dict[str, PVGroup] = {args.prefix.rstrip(":"): ch.grp}
    if chic:
        groups[args.chic_prefix.rstrip(":")] = chic.grp

    providers = build_provider_dict(groups)
    print(
        f"[sim-chopper] Serving at '{args.prefix}*' (records with ':'). TDC list PV: '{args.prefix}02-TS-I'. Ctrl+C to exit."
    )
    try:
        with Server(providers=[providers]) as S:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[sim-chopper] Shutting down...")
    finally:
        ch.stop()


if __name__ == "__main__":
    main()
