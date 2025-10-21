#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from p4p.server import Server

from common.pva_common import PVGroup, build_provider_dict

# ----------------------------
# Helpers and constants
# ----------------------------


@dataclass
class MotorConfig:
    egu: str = "mm"
    prec: int = 3
    user_limits: Tuple[float, float] = (-1e3, 1e3)
    dial_limits: Tuple[float, float] = (-1e3, 1e3)
    velo: float = 5.0
    vbas: float = 0.0
    vmax: float = 0.0
    accl: float = 1.0
    hvel: float = 2.0
    jvel: float = 2.0
    jar: float = 0.0
    rdbd: float = 0.001
    spdb: float = 0.0
    dly: float = 0.0
    off: float = 0.0
    dir_pos: bool = True  # DIR 0=Pos, 1=Neg


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _set_bit(flags: int, bit: int, on: bool) -> int:
    mask = 1 << (bit - 1)
    return (flags | mask) if on else (flags & ~mask)


MSTA_DIRECTION = 1
MSTA_DONE = 2
MSTA_PLUS_LS = 3
MSTA_HOMELS = 4
MSTA_HOME = 8
MSTA_MOVING = 11
MSTA_MINUS_LS = 14
MSTA_HOMED = 15


# ----------------------------
# Motion model
# ----------------------------


class MotorState:
    def __init__(self, cfg: MotorConfig):
        self.cfg = cfg
        self.dir_rec = 0 if cfg.dir_pos else 1
        self.off = cfg.off

        self.dial_pos = 0.0
        self.dial_target = 0.0

        self.dmov = 1
        self.movn = 0
        self.hls = 0
        self.lls = 0
        self.lvio = 0
        self.msta = 0

        self._vel_now = 0.0
        self._moving = False
        self._last_ts = time.monotonic()
        self._dmov_ready_ts = 0.0

        self._homing = False
        self._jogging = False
        self._home_pos = 0.0

        self.set_mode = 0
        self.foff = 0
        self.cnen = 1

    def _sign(self) -> float:
        return +1.0 if self.dir_rec == 0 else -1.0

    def user_from_dial(self, d: float) -> float:
        return d * self._sign() + self.off

    def dial_from_user(self, u: float) -> float:
        return (u - self.off) * self._sign()

    def _limit_v(self, v: float) -> float:
        v = max(self.cfg.vbas, v)
        if self.cfg.vmax > 0.0:
            v = min(v, self.cfg.vmax)
        return v

    def _start_motion(
        self, d_target: float, dir_override: Optional[bool] = None
    ) -> None:
        self.lvio = 0
        self.dial_target = d_target
        self._moving = True
        self._jogging = False
        self.movn = 1
        self.dmov = 0
        self._dmov_ready_ts = 0.0
        dir_positive = (
            (self.dial_target - self.dial_pos) >= 0.0
            if dir_override is None
            else bool(dir_override)
        )
        self.msta = _set_bit(self.msta, MSTA_DIRECTION, dir_positive)
        self.msta = _set_bit(self.msta, MSTA_MOVING, True)
        self.msta = _set_bit(self.msta, MSTA_DONE, False)

    def _arm_dmov_pulse_without_motion(self) -> None:
        self._moving = False
        self._jogging = False
        self.movn = 0
        self.dmov = 0
        self._dmov_ready_ts = time.monotonic() + max(self.cfg.dly, 0.0)
        self.msta = _set_bit(self.msta, MSTA_MOVING, False)
        self.msta = _set_bit(self.msta, MSTA_DONE, False)

    def move_user(self, u: float) -> Optional[str]:
        if abs(u - self.user_from_dial(self.dial_pos)) < max(self.cfg.spdb, 0.0):
            self._arm_dmov_pulse_without_motion()
            return None
        return self.move_dial(self.dial_from_user(u))

    def move_dial(self, d: float) -> Optional[str]:
        lo_u, hi_u = self.cfg.user_limits
        lo_d, hi_d = self.cfg.dial_limits

        if abs(self.user_from_dial(d) - self.user_from_dial(self.dial_pos)) < max(
            self.cfg.spdb, 0.0
        ):
            self._arm_dmov_pulse_without_motion()
            return None

        if not (lo_d <= d <= hi_d):
            self.lvio = 1
            return f"Target {d:g} outside dial limits [{lo_d:g}, {hi_d:g}]"
        u = self.user_from_dial(d)
        if not (lo_u <= u <= hi_u):
            self.lvio = 1
            return f"Target {u:g} outside user limits [{lo_u:g}, {hi_u:g}]"

        self._start_motion(d)
        return None

    def stop(self) -> None:
        self._moving = False
        self._jogging = False
        self.movn = 0
        self._vel_now = 0.0
        self.dial_target = self.dial_pos
        self._dmov_ready_ts = time.monotonic() + max(self.cfg.dly, 0.0)
        self.dmov = 0
        self._homing = False
        self.msta = _set_bit(self.msta, MSTA_MOVING, False)
        self.msta = _set_bit(self.msta, MSTA_DONE, False)

    def home(self, forward: bool) -> None:
        if self._moving:
            self.stop()
        self._homing = True
        self._jogging = False
        self._start_motion(self._home_pos, dir_override=bool(forward))

    def jog(self, forward: bool, on: bool) -> None:
        if on:
            d = self.cfg.dial_limits[1] if forward else self.cfg.dial_limits[0]
            self._homing = False
            self._jogging = True
            self._start_motion(d, dir_override=bool(forward))
        else:
            self.stop()

    def _finish_at_target(self, now: float) -> None:
        self._moving = False
        self._jogging = False
        self.movn = 0
        self._vel_now = 0.0
        self._dmov_ready_ts = now + max(self.cfg.dly, 0.0)
        self.msta = _set_bit(self.msta, MSTA_MOVING, False)
        self.msta = _set_bit(self.msta, MSTA_DONE, False)

    def step(self, now: float) -> None:
        dt = max(0.0, now - self._last_ts)
        self._last_ts = now

        lo_d, hi_d = self.cfg.dial_limits
        self.hls = 1 if self.dial_pos >= hi_d else 0
        self.lls = 1 if self.dial_pos <= lo_d else 0
        self.msta = _set_bit(self.msta, MSTA_PLUS_LS, bool(self.hls))
        self.msta = _set_bit(self.msta, MSTA_MINUS_LS, bool(self.lls))

        at_home = abs(self.dial_pos - self._home_pos) <= max(self.cfg.rdbd, 1e-6)
        self.msta = _set_bit(self.msta, MSTA_HOME, at_home)
        self.msta = _set_bit(self.msta, MSTA_HOMELS, at_home)

        if not self._moving:
            if self._homing and at_home:
                self.msta = _set_bit(self.msta, MSTA_HOMED, True)
                self._homing = False
            if (
                (self.dmov == 0)
                and (self._dmov_ready_ts > 0)
                and (now >= self._dmov_ready_ts)
            ):
                self.dmov = 1
                self.msta = _set_bit(self.msta, MSTA_DONE, True)
            return

        target = self.dial_target
        pos = self.dial_pos
        dist = target - pos
        adist = abs(dist)

        if adist <= max(self.cfg.rdbd, 1e-12):
            self.dial_pos = target
            self._finish_at_target(now)
            if self._homing and at_home:
                self.msta = _set_bit(self.msta, MSTA_HOMED, True)
                self._homing = False
            return

        if self._homing:
            v_target = self.cfg.hvel
        elif self._jogging:
            v_target = self.cfg.jvel if self.cfg.jvel > 0.0 else self.cfg.velo
        else:
            v_target = self.cfg.velo
        v_target = self._limit_v(v_target)
        a = (
            self.cfg.jar
            if (self._jogging and self.cfg.jar > 0.0)
            else v_target / max(self.cfg.accl, 1e-6)
        )

        v = self._vel_now
        s_stop = (v * v) / (2 * max(a, 1e-9))
        if adist <= s_stop:
            v = max(0.0, v - a * dt)
        else:
            v = min(v_target, v + a * dt)

        step_mag = v * dt
        if step_mag >= adist or v <= 1e-9:
            self.dial_pos = target
            self._finish_at_target(now)
            if self._homing and at_home:
                self.msta = _set_bit(self.msta, MSTA_HOMED, True)
                self._homing = False
            self._vel_now = 0.0
            return

        self.dial_pos = pos + (step_mag if dist > 0 else -step_mag)
        self._vel_now = v

        if self.dial_pos > hi_d:
            self.dial_pos = hi_d
            self._vel_now = 0.0
            self._finish_at_target(now)
        elif self.dial_pos < lo_d:
            self.dial_pos = lo_d
            self._vel_now = 0.0
            self._finish_at_target(now)


# ----------------------------
# Motor device using PVGroup
# ----------------------------


class Motor:
    def __init__(self, prefix: str, cfg: MotorConfig, tick_hz: float = 10.0):
        self.cfg = cfg
        self.state = MotorState(cfg)
        self.lock = threading.RLock()
        self.tick = 1.0 / tick_hz
        self.stop_evt = threading.Event()

        self.pvs = PVGroup(
            prefix,
            default_units=cfg.egu,
            default_precision=cfg.prec,
            throttle_fields=("RBV", "DRBV"),
        )
        self.pvs.set_write_cb(self._on_write)
        self._build_motor_pvs()
        self.pvs.create_root_alias(
            readback_suffix="RBV",
            setpoint_suffix="VAL",
            user_limits=self.cfg.user_limits,
        )

        self.thread = threading.Thread(target=self._run, daemon=True)

    # ---------- PV declarations ----------

    def _build_motor_pvs(self) -> None:
        u_lo, u_hi = self.cfg.user_limits
        d_lo, d_hi = self.cfg.dial_limits

        # Drives (device echoes; don't double-echo on put)
        self.pvs.make_float(
            "VAL",
            0.0,
            writeable=True,
            display_limits=(u_lo, u_hi),
            control_limits=(u_lo, u_hi),
            echo_on_put=False,
        )
        self.pvs.make_float(
            "DVAL",
            0.0,
            writeable=True,
            display_limits=(d_lo, d_hi),
            control_limits=(d_lo, d_hi),
            echo_on_put=False,
        )
        self.pvs.make_float(
            "RVAL",
            0.0,
            writeable=True,
            display_limits=(d_lo, d_hi),
            control_limits=(d_lo, d_hi),
            echo_on_put=False,
        )
        self.pvs.make_float(
            "RLV",
            0.0,
            writeable=True,
            display_limits=(u_lo, u_hi),
            control_limits=(u_lo, u_hi),
        )

        # Readbacks/status
        self.pvs.make_float(
            "RBV",
            0.0,
            writeable=False,
            display_limits=(u_lo, u_hi),
            control_limits=(u_lo, u_hi),
        )
        self.pvs.make_float(
            "DRBV",
            0.0,
            writeable=False,
            display_limits=(d_lo, d_hi),
            control_limits=(d_lo, d_hi),
        )
        self.pvs.make_int("DMOV", 1, code="h", writeable=False)
        self.pvs.make_int("MOVN", 0, code="h", writeable=False)
        self.pvs.make_int("MSTA", 0, code="I", writeable=False)
        self.pvs.make_int("MISS", 0, code="h", writeable=False)

        # Motion params
        self.pvs.make_float("VELO", self.cfg.velo)
        self.pvs.make_float("VBAS", self.cfg.vbas)
        self.pvs.make_float("VMAX", self.cfg.vmax)
        self.pvs.make_float("ACCL", self.cfg.accl)
        self.pvs.make_float("HVEL", self.cfg.hvel)
        self.pvs.make_float("JVEL", self.cfg.jvel)
        self.pvs.make_float("JAR", self.cfg.jar)
        self.pvs.make_float("RDBD", self.cfg.rdbd)
        self.pvs.make_float("SPDB", self.cfg.spdb)
        self.pvs.make_float("DLY", self.cfg.dly)

        # Mapping / calibration / enable
        self.pvs.make_enum(
            "DIR", ["Pos", "Neg"], init_index=0 if self.cfg.dir_pos else 1
        )
        self.pvs.make_float("OFF", self.cfg.off)
        self.pvs.make_int("SET", 0, code="h")
        self.pvs.make_int("FOFF", 0, code="h")
        self.pvs.make_int("CNEN", 1, code="h")

        # Limits mirrors/flags
        self.pvs.make_float("HLM", self.cfg.user_limits[1])
        self.pvs.make_float("LLM", self.cfg.user_limits[0])
        self.pvs.make_float("DHLM", self.cfg.dial_limits[1])
        self.pvs.make_float("DLLM", self.cfg.dial_limits[0])
        self.pvs.make_int("LVIO", 0, code="h", writeable=False)
        self.pvs.make_int("HLS", 0, code="h", writeable=False)
        self.pvs.make_int("LLS", 0, code="h", writeable=False)

        # Commands
        self.pvs.make_int("STOP", 0, code="h")
        self.pvs.make_int("HOMF", 0, code="h")
        self.pvs.make_int("HOMR", 0, code="h")
        self.pvs.make_int("JOGF", 0, code="h")
        self.pvs.make_int("JOGR", 0, code="h")

        # Misc
        self.pvs.make_float("MDEL", 0.0)  # device will set initial MDEL later
        self.pvs.make_str("DESC", "")

        # Formatting/units
        self.pvs.make_str("EGU", self.cfg.egu)
        self.pvs.make_int("PREC", int(self.cfg.prec), code="h")

    # ---------- posting helpers specific to motor ----------

    def _post_status_snapshot(self) -> None:
        s = self.state
        sev = 2 if (s.hls or s.lls) else 0
        msg = "at hardware limit" if sev else ""

        prev_movn = self.pvs._last_sent.get("MOVN", (None, None, None))[0]
        prev_dmov = self.pvs._last_sent.get("DMOV", (None, None, None))[0]
        force_final = (prev_movn == 1 and int(s.movn) == 0) or (
            prev_dmov == 0 and int(s.dmov) == 1
        )

        rbv = s.user_from_dial(s.dial_pos)
        self.pvs.post_num(
            "DRBV", s.dial_pos, severity=sev, message=msg, force=force_final
        )
        self.pvs.post_num("RBV", rbv, severity=sev, message=msg, force=force_final)
        self.pvs.post_root_num(rbv, severity=sev, message=msg, force=force_final)

        self.pvs.post_num("DMOV", int(s.dmov))
        self.pvs.post_num("MOVN", int(s.movn))
        self.pvs.post_num("LVIO", int(s.lvio))
        self.pvs.post_num("HLS", int(s.hls))
        self.pvs.post_num("LLS", int(s.lls))
        self.pvs.post_num("MSTA", int(s.msta))
        self.pvs.post_num("HOMF", int(s._homing))
        self.pvs.post_num("HOMR", int(s._homing))

    def _sync_drive_echo_to_target(self) -> None:
        s = self.state
        self.pvs.post_num("DVAL", s.dial_target)
        self.pvs.post_num("RVAL", s.dial_target)
        self.pvs.post_num("VAL", s.user_from_dial(s.dial_target))

    def _sync_drive_echo_to_position(self) -> None:
        s = self.state
        self.pvs.post_num("DVAL", s.dial_pos)
        self.pvs.post_num("RVAL", s.dial_pos)
        self.pvs.post_num("VAL", s.user_from_dial(s.dial_pos))
        self.pvs.post_root_num(s.user_from_dial(s.dial_pos), force=True)

    def _repost_limits_meta(self) -> None:
        u_lo, u_hi = self.cfg.user_limits
        d_lo, d_hi = self.cfg.dial_limits
        up_u = {
            "display.limitLow": u_lo,
            "display.limitHigh": u_hi,
            "control.limitLow": u_lo,
            "control.limitHigh": u_hi,
            "valueAlarm.lowAlarmLimit": u_lo,
            "valueAlarm.highAlarmLimit": u_hi,
        }
        up_d = {
            "display.limitLow": d_lo,
            "display.limitHigh": d_hi,
            "control.limitLow": d_lo,
            "control.limitHigh": d_hi,
            "valueAlarm.lowAlarmLimit": d_lo,
            "valueAlarm.highAlarmLimit": d_hi,
        }
        self.pvs.post_meta("VAL", up_u)
        self.pvs.post_meta("RBV", up_u)
        self.pvs.post_meta("DVAL", up_d)
        self.pvs.post_meta("DRBV", up_d)
        self.pvs.post_root_meta(up_u)
        self.pvs.post_num("LLM", u_lo)
        self.pvs.post_num("HLM", u_hi)
        self.pvs.post_num("DLLM", d_lo)
        self.pvs.post_num("DHLM", d_hi)

    def _sync_user_from_dial_limits(self) -> None:
        d_lo, d_hi = self.cfg.dial_limits
        lo_u = self.state.user_from_dial(d_lo)
        hi_u = self.state.user_from_dial(d_hi)
        if lo_u > hi_u:
            lo_u, hi_u = hi_u, lo_u
        self.cfg.user_limits = (lo_u, hi_u)
        self._repost_limits_meta()

    def _sync_dial_from_user_limits(self) -> None:
        u_lo, u_hi = self.cfg.user_limits
        lo_d = self.state.dial_from_user(u_lo)
        hi_d = self.state.dial_from_user(u_hi)
        if lo_d > hi_d:
            lo_d, hi_d = hi_d, lo_d
        self.cfg.dial_limits = (lo_d, hi_d)
        self._repost_limits_meta()

    # ---------- write handling ----------

    def _on_write(self, suffix: str, value):
        s = self.state
        c = self.cfg

        def clamp_vel(v):
            v = max(0.0, float(v))
            if c.vmax > 0.0:
                v = min(v, c.vmax)
            return max(c.vbas, v)

        def _handle_drive_put(kind: str, fvalue: float):
            if s.set_mode == 1 and s.foff == 0 and kind == "VAL":
                new_off = float(fvalue) - (s.dial_pos * s._sign())
                delta = new_off - s.off
                s.off = new_off
                self.pvs.post_num("OFF", s.off)
                u_lo, u_hi = c.user_limits
                c.user_limits = (u_lo + delta, u_hi + delta)
                self._sync_dial_from_user_limits()
                self.pvs.post_num("VAL", float(fvalue))
                self._post_status_snapshot()
                s._arm_dmov_pulse_without_motion()
                return

            if s.set_mode == 1 and s.foff == 1:
                if kind == "VAL":
                    d = s.dial_from_user(float(fvalue))
                else:
                    d = float(fvalue)
                d = _clamp(d, c.dial_limits[0], c.dial_limits[1])
                s.dial_pos = d
                s.dial_target = d
                s._moving = False
                s.movn = 0
                s._vel_now = 0.0
                s._homing = False
                s._jogging = False
                s._arm_dmov_pulse_without_motion()
                self._sync_drive_echo_to_position()
                self._post_status_snapshot()
                return

            if kind == "VAL":
                s.move_user(float(fvalue))
                self._sync_drive_echo_to_target()
            else:
                s.move_dial(float(fvalue))
                self._sync_drive_echo_to_target()

        if suffix in ("VAL", "DVAL", "RVAL"):
            _handle_drive_put(suffix, float(value))

        elif suffix == "RLV":
            u = s.user_from_dial(s.dial_pos) + float(value)
            _handle_drive_put("VAL", u)
            self.pvs.post_num("RLV", 0)

        elif suffix == "STOP":
            s.stop()
            self._sync_drive_echo_to_position()

        elif suffix == "HOMF":
            if int(value) == 1:
                s.home(forward=True)
            self.pvs.post_num("HOMF", 0)
            self._sync_drive_echo_to_target()

        elif suffix == "HOMR":
            if int(value) == 1:
                s.home(forward=False)
            self.pvs.post_num("HOMR", 0)
            self._sync_drive_echo_to_target()

        elif suffix == "JOGF":
            s.jog(forward=True, on=bool(int(value)))
            if not int(value):
                self._sync_drive_echo_to_position()

        elif suffix == "JOGR":
            s.jog(forward=False, on=bool(int(value)))
            if not int(value):
                self._sync_drive_echo_to_position()

        elif suffix == "DIR":
            prev_dir = s.dir_rec
            s.dir_rec = 0 if int(value) == 0 else 1
            self.pvs.post_num("VAL", s.user_from_dial(s.dial_pos))
            if s.dir_rec != prev_dir:
                self._sync_user_from_dial_limits()

        elif suffix == "OFF":
            old_off = s.off
            s.off = float(value)
            self.pvs.post_num("VAL", s.user_from_dial(s.dial_pos))
            delta = s.off - old_off
            u_lo, u_hi = c.user_limits
            c.user_limits = (u_lo + delta, u_hi + delta)
            self._sync_dial_from_user_limits()

        elif suffix == "SET":
            s.set_mode = 1 if int(value) else 0
            self.pvs.post_num("SET", int(s.set_mode))

        elif suffix == "FOFF":
            s.foff = 1 if int(value) else 0
            self.pvs.post_num("FOFF", int(s.foff))

        elif suffix == "CNEN":
            s.cnen = 1 if int(value) else 0
            self.pvs.post_num("CNEN", int(s.cnen))

        elif suffix == "EGU":
            self.cfg.egu = str(value)
            self.pvs.default_units = self.cfg.egu
            self.pvs.update_float_meta_all(units=self.cfg.egu)

        elif suffix == "PREC":
            self.cfg.prec = int(value)
            self.pvs.default_precision = self.cfg.prec
            self.pvs.update_float_meta_all(precision=self.cfg.prec)

        elif suffix == "VELO":
            self.cfg.velo = clamp_vel(value)

        elif suffix == "JVEL":
            self.cfg.jvel = clamp_vel(value)

        elif suffix == "HVEL":
            self.cfg.hvel = clamp_vel(value)

        elif suffix == "VBAS":
            self.cfg.vbas = max(0.0, float(value))
            self.cfg.velo = max(self.cfg.velo, self.cfg.vbas)
            self.cfg.jvel = max(self.cfg.jvel, self.cfg.vbas)
            self.cfg.hvel = max(self.cfg.hvel, self.cfg.vbas)

        elif suffix == "VMAX":
            self.cfg.vmax = max(0.0, float(value))
            if self.cfg.vmax > 0.0:
                self.cfg.velo = min(self.cfg.velo, self.cfg.vmax)
                self.cfg.jvel = min(self.cfg.jvel, self.cfg.vmax)
                self.cfg.hvel = min(self.cfg.hvel, self.cfg.vmax)

        elif suffix == "ACCL":
            self.cfg.accl = max(1e-3, float(value))

        elif suffix == "JAR":
            self.cfg.jar = max(0.0, float(value))

        elif suffix == "RDBD":
            self.cfg.rdbd = max(0.0, float(value))

        elif suffix == "SPDB":
            self.cfg.spdb = max(0.0, float(value))

        elif suffix == "DLY":
            self.cfg.dly = max(0.0, float(value))

        elif suffix == "MDEL":
            self.pvs.set_mdel(float(value))

        elif suffix in ("HLM", "LLM", "DHLM", "DLLM"):
            v = float(value)
            if suffix == "HLM":
                lo_u, _ = c.user_limits
                c.user_limits = (lo_u, v)
                self._sync_dial_from_user_limits()
            elif suffix == "LLM":
                _, hi_u = c.user_limits
                c.user_limits = (v, hi_u)
                self._sync_dial_from_user_limits()
            elif suffix == "DHLM":
                lo_d, _ = c.dial_limits
                c.dial_limits = (lo_d, v)
                self._sync_user_from_dial_limits()
            elif suffix == "DLLM":
                _, hi_d = c.dial_limits
                c.dial_limits = (v, hi_d)
                self._sync_user_from_dial_limits()

            u = s.user_from_dial(s.dial_target)
            s.lvio = int(
                not (c.user_limits[0] <= u <= c.user_limits[1])
                or not (c.dial_limits[0] <= s.dial_target <= c.dial_limits[1])
            )

        self._post_status_snapshot()

    # ---------- runner ----------

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self.stop_evt.is_set():
            with self.lock:
                self.state.step(time.monotonic())
                self._post_status_snapshot()
            next_tick += self.tick
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def step_once(self) -> None:
        """Deterministic single-step (handy for unit tests)."""
        with self.lock:
            self.state.step(time.monotonic())
            self._post_status_snapshot()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_evt.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass


# ----------------------------
# Factory & CLI
# ----------------------------


def create_motor(
    prefix: str, cfg: Optional[MotorConfig] = None, tick_hz: float = 50.0
) -> Motor:
    return Motor(prefix, cfg or MotorConfig(), tick_hz=tick_hz)


def main():
    ap = argparse.ArgumentParser(
        description="Simulated EPICS motor record (p4p/PVA) using common PV core with metadata + MDEL + VAL/RBV alias."
    )
    ap.add_argument("--prefix", default="SIM:M1", help="PV prefix (e.g., 'SIM:M1')")
    ap.add_argument(
        "--tick-hz", type=float, default=10.0, help="Simulation tick frequency"
    )
    ap.add_argument("--egu", default="mm", help="Engineering units string")
    ap.add_argument("--prec", type=int, default=3, help="Display precision")
    ap.add_argument(
        "--llm", type=float, default=-1000.0, help="User low limit (VAL/RBV)"
    )
    ap.add_argument(
        "--hlm", type=float, default=+1000.0, help="User high limit (VAL/RBV)"
    )
    ap.add_argument(
        "--dllm", type=float, default=-1000.0, help="Dial low limit (DVAL/DRBV)"
    )
    ap.add_argument(
        "--dhlm", type=float, default=+1000.0, help="Dial high limit (DVAL/DRBV)"
    )
    ap.add_argument("--velo", type=float, default=5.0, help="Motion VELO (EGU/s)")
    ap.add_argument("--vbas", type=float, default=0.0, help="Motion VBAS (EGU/s)")
    ap.add_argument(
        "--vmax", type=float, default=0.0, help="Max velocity VMAX (EGU/s, 0=unlimited)"
    )
    ap.add_argument("--accl", type=float, default=1.0, help="ACCL (seconds to VELO)")
    ap.add_argument("--hvel", type=float, default=2.0, help="Homing velocity")
    ap.add_argument("--jvel", type=float, default=2.0, help="Jog velocity")
    ap.add_argument(
        "--jar", type=float, default=0.0, help="Jog acceleration (EGU/s^2); 0=>use ACCL"
    )
    ap.add_argument(
        "--rdbd", type=float, default=0.001, help="In-position deadband (EGU)"
    )
    ap.add_argument("--spdb", type=float, default=0.0, help="Set point deadband (EGU)")
    ap.add_argument("--dly", type=float, default=0.0, help="DMOV settle delay (s)")
    ap.add_argument("--off", type=float, default=0.0, help="User offset OFF")
    ap.add_argument("--dir", type=int, default=0, help="DIR (0=Pos, 1=Neg)")
    ap.add_argument(
        "--mdel",
        type=float,
        default=0.1,
        help="Initial MDEL deadband (EGU) for RBV/DRBV",
    )
    args = ap.parse_args()

    cfg = MotorConfig(
        egu=args.egu,
        prec=args.prec,
        user_limits=(args.llm, args.hlm),
        dial_limits=(args.dllm, args.dhlm),
        velo=args.velo,
        vbas=args.vbas,
        vmax=args.vmax,
        accl=args.accl,
        hvel=args.hvel,
        jvel=args.jvel,
        jar=args.jar,
        rdbd=args.rdbd,
        spdb=args.spdb,
        dly=args.dly,
        off=args.off,
        dir_pos=(args.dir == 0),
    )

    motor = create_motor(args.prefix, cfg, tick_hz=args.tick_hz).start()
    provider = build_provider_dict({args.prefix: motor.pvs})

    # initialize MDEL after PVs exist
    motor.pvs.set_mdel(args.mdel)

    print(
        f"[sim-motor] Serving PVA motor at prefix '{args.prefix}'. "
        f"Alias '{args.prefix}' maps GET/MONITOR->RBV and PUT->VAL. Press Ctrl+C to exit."
    )
    try:
        with Server(providers=[provider]) as S:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[sim-motor] Shutting down...")
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
