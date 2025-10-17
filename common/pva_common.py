#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Sequence

from p4p.nt import NTEnum, NTScalar
from p4p.server.thread import SharedPV


class PVGroup:
    """
    Reusable helper for creating p4p SharedPVs with good NT metadata,
    safe put handling, change coalescing, and RBV/VAL-style aliasing.

    Device simulators use:
      - make_float / make_int / make_enum / make_str / make_array_* to declare PVs
      - set_write_cb(fn) to receive puts
      - post_num / post_meta / post_array for updates
      - create_root_alias(readback_suffix='RBV', setpoint_suffix='VAL')
      - post_root_num / post_root_meta
      - set_mdel(deadband_egu) to throttle updates for selected PVs
      - update_float_meta_all(units=..., precision=...) to keep float PVs consistent
    """

    def __init__(
        self,
        prefix: str,
        *,
        default_units: str = "",
        default_precision: int = 3,
        write_cb: Optional[Callable[[str, Any], None]] = None,
        throttle_fields: Iterable[str] = ("RBV", "DRBV"),  # fields that obey MDEL
        time_fn: Callable[[], float] = time.time,
        name_join: str = ".",  # <--- NEW: per-group joiner ('.' for fields by default)
    ):
        self.prefix = prefix.rstrip(":")
        self.default_units = default_units
        self.default_precision = int(default_precision)
        self._write_cb = write_cb
        self.pvs: Dict[str, SharedPV] = {}
        self.nts: Dict[str, Any] = {}
        self._last_sent: Dict[str, Tuple[Any, int, str]] = {}
        self._mdel: float = 0.0
        self._mdel_fields = set(throttle_fields)
        self._time_fn = time_fn
        self.name_join = name_join  # '.' for record.fields, ':' for record:names

        # Root alias (optional; created with create_root_alias())
        self.root_nt: Optional[NTScalar] = None
        self.root_pv: Optional[SharedPV] = None
        self._last_root: Optional[Tuple[Any, int, str]] = None
        self._alias_readback: Optional[str] = None
        self._alias_setpoint: Optional[str] = None

    # ---------- configuration / DI ----------

    def set_write_cb(self, cb: Callable[[str, Any], None]) -> None:
        self._write_cb = cb

    def set_mdel(self, deadband_egu: float) -> None:
        """Set monitor deadband (EGU) used to throttle posts of 'throttle_fields' and root alias."""
        self._mdel = max(0.0, float(deadband_egu))
        if "MDEL" in self.pvs:
            self.pvs["MDEL"].post(self._mdel, timestamp=self._time_fn())

    def add_mdel_fields(self, *suffixes: str) -> None:
        self._mdel_fields.update(suffixes)

    # ---------- PV creation ----------

    def make_float(
        self,
        suffix: str,
        init: float,
        *,
        writeable: bool = True,
        display_limits: Optional[Tuple[float, float]] = None,
        control_limits: Optional[Tuple[float, float]] = None,
        with_alarm: bool = True,
        echo_on_put: bool = True,
    ) -> None:
        """
        Create NTScalar('d') PV with standard display/control/valueAlarm meta.
        If echo_on_put=False the value isn't echoed by the generic layer (device will post it later).
        """
        nt = NTScalar("d", display=True, control=True, valueAlarm=with_alarm, form=True)
        self.nts[suffix] = nt
        pv = SharedPV(nt=nt)
        self.pvs[suffix] = pv

        V = nt.wrap(init, timestamp=self._time_fn())
        V["display.units"] = self.default_units
        V["display.precision"] = self.default_precision
        if display_limits is not None:
            V["display.limitLow"], V["display.limitHigh"] = display_limits
        if control_limits is not None:
            V["control.limitLow"], V["control.limitHigh"] = control_limits
            if with_alarm:
                (
                    V["valueAlarm.lowAlarmLimit"],
                    V["valueAlarm.highAlarmLimit"],
                ) = control_limits
        V["alarm.severity"] = 0
        V["alarm.status"] = 0
        V["alarm.message"] = "NO_ALARM"
        pv.open(V)

        if writeable:
            @pv.put
            def _on_put(pv_obj, op, _suffix=suffix):
                raw = op.value()
                try:
                    new = float(raw["value"])
                except Exception as e:
                    try:
                        new = float(raw)
                    except Exception:
                        op.done(error=f"Bad put payload for {self.prefix}.{_suffix}: {e}")
                        return
                if self._write_cb:
                    self._write_cb(_suffix, new)
                if echo_on_put:
                    pv_obj.post(new, timestamp=self._time_fn())
                    self._last_sent[_suffix] = (new, 0, "")
                op.done()

    def make_int(self, suffix: str, init: int, *, code: str = "h", writeable=True) -> None:
        nt = NTScalar(code, display=True, control=True, form=True)
        self.nts[suffix] = nt
        pv = SharedPV(nt=nt)
        self.pvs[suffix] = pv

        V = nt.wrap(int(init), timestamp=self._time_fn())
        V["alarm.severity"] = 0
        V["alarm.status"] = 0
        V["alarm.message"] = "NO_ALARM"
        pv.open(V)

        if writeable:
            @pv.put
            def _on_put(pv_obj, op, _suffix=suffix):
                raw = op.value()
                try:
                    new = int(raw["value"])
                except Exception as e:
                    try:
                        new = int(raw)
                    except Exception:
                        op.done(error=f"Bad put payload for {self.prefix}.{_suffix}: {e}")
                        return
                if self._write_cb:
                    self._write_cb(_suffix, new)
                pv_obj.post(new, timestamp=self._time_fn())
                self._last_sent[_suffix] = (new, 0, "")
                op.done()

    def make_str(self, suffix: str, init: str, *, writeable=True) -> None:
        nt = NTScalar("s", display=True, form=True)
        self.nts[suffix] = nt
        pv = SharedPV(nt=nt)
        self.pvs[suffix] = pv

        V = nt.wrap(str(init), timestamp=self._time_fn())
        V["alarm.severity"] = 0
        V["alarm.status"] = 0
        V["alarm.message"] = "NO_ALARM"
        pv.open(V)

        if writeable:
            @pv.put
            def _on_put(pv_obj, op, _suffix=suffix):
                raw = op.value()
                try:
                    new = str(raw["value"])
                except Exception as e:
                    try:
                        new = str(raw)
                    except Exception:
                        op.done(error=f"Bad put payload for {self.prefix}.{_suffix}: {e}")
                        return
                if self._write_cb:
                    self._write_cb(_suffix, new)
                pv_obj.post(new, timestamp=self._time_fn())
                self._last_sent[_suffix] = (new, 0, "")
                op.done()

    def make_enum(self, suffix: str, choices, *, init_index: int = 0, writeable=True) -> None:
        choices = list(choices)
        nt = NTEnum(display=True, control=True)
        self.nts[suffix] = nt

        idx = int(init_index)
        if not (0 <= idx < len(choices)):
            raise ValueError(f"enum index {idx} out of range 0..{len(choices) - 1}")

        # IMPORTANT: create PV with nt=nt AND open with a plain index (not a Value, not a dict)
        pv = SharedPV(nt=nt)
        self.pvs[suffix] = pv
        pv.open(idx, choices=list(choices), timestamp=self._time_fn())

        def _parse_enum_put(payload) -> int:
            # accept ints, floats, strings (name or "2"), dicts, and p4p.Value-shapes
            def _str_to_idx(s: str) -> int:
                s = s.strip()
                if s.lstrip("-").isdigit():
                    return int(s)
                try:
                    return choices.index(s)
                except ValueError:
                    lower = [c.lower() for c in choices]
                    return lower.index(s.lower())

            # p4p.Value-like
            try:
                if hasattr(payload, "get") and callable(getattr(payload, "get")):
                    v = payload.get("value.index", None)
                    if v is not None:
                        return int(v)
                    v = payload.get("value", None)
                    if v is not None:
                        if hasattr(v, "get") and callable(getattr(v, "get")):
                            ii = v.get("index", None)
                            if ii is not None:
                                return int(ii)
                            vv = v.get("value", None)
                            if vv is not None:
                                if isinstance(vv, (int, float)):
                                    return int(vv)
                                if isinstance(vv, str):
                                    return _str_to_idx(vv)
                        else:
                            if isinstance(v, (int, float)):
                                return int(v)
                            if isinstance(v, str):
                                return _str_to_idx(v)
                    ii = payload.get("index", None)
                    if ii is not None:
                        return int(ii)
            except Exception:
                pass

            # dict
            if isinstance(payload, dict):
                v = payload.get("value", payload)
                if isinstance(v, dict):
                    if "index" in v:
                        return int(v["index"])
                    if "value" in v:
                        v = v["value"]
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str):
                    return _str_to_idx(v)
                return int(v)

            # scalars
            if isinstance(payload, (int, float)):
                return int(payload)
            if isinstance(payload, str):
                return _str_to_idx(payload)

            return int(payload)

        if writeable:
            @pv.put
            def _on_put(pv_obj, op, _suffix=suffix, _choices=choices):
                raw = op.value()
                try:
                    new_idx = _parse_enum_put(raw)
                    if not (0 <= new_idx < len(_choices)):
                        raise ValueError(f"enum index {new_idx} out of range 0..{len(_choices) - 1}")
                except Exception as e:
                    op.done(error=f"Bad put payload for {self.prefix}.{_suffix}: {e}")
                    return

                if self._write_cb:
                    self._write_cb(_suffix, new_idx)

                # Post just the index. Do NOT post a Value or metadata dict for enums.
                pv_obj.post(new_idx, timestamp=self._time_fn())
                self._last_sent[_suffix] = (new_idx, 0, "")
                op.done()

    # ---------- Array PVs (added) ----------

    def make_array_int64(self, suffix: str, init: Sequence[int] | None = None, *, writeable=False) -> None:
        """
        Create NTScalar('al') PV for array of int64 (e.g., lists of epoch-ns timestamps).
        """
        nt = NTScalar("al", display=True, form=True)
        self.nts[suffix] = nt
        pv = SharedPV(nt=nt)
        self.pvs[suffix] = pv

        initial = list(init) if init is not None else []
        V = nt.wrap(initial, timestamp=self._time_fn())
        V["alarm.severity"] = 0
        V["alarm.status"] = 0
        V["alarm.message"] = "NO_ALARM"
        pv.open(V)

        if writeable:
            @pv.put
            def _on_put(pv_obj, op, _suffix=suffix):
                raw = op.value()
                try:
                    seq = raw["value"] if isinstance(raw, dict) else raw
                    new = [int(x) for x in seq]
                except Exception as e:
                    op.done(error=f"Bad put payload for {self.prefix}.{_suffix}: {e}")
                    return
                if self._write_cb:
                    self._write_cb(_suffix, new)
                pv_obj.post(new, timestamp=self._time_fn())
                self._last_sent[_suffix] = ("len", len(new), "")
                op.done()

    def post_array(
        self,
        suffix: str,
        values: Sequence[int] | Sequence[float],
        severity: int = 0,
        message: str = "",
    ) -> None:
        """
        Post a new array value (uses the same timestamp/severity/message style as scalars).
        """
        self.pvs[suffix].post(values, timestamp=self._time_fn(), severity=severity, message=message)
        self._last_sent[suffix] = ("len", len(values), "")

    # ---------- alias PV ('<prefix>') ----------

    def create_root_alias(
        self,
        *,
        readback_suffix: str,
        setpoint_suffix: str,
        user_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Create '<prefix>' alias PV:
          - GET/MONITOR reads the device readback (RBV-like)
          - PUT forwards to the device setpoint (VAL-like)
        """
        nt = NTScalar("d", display=True, control=True, valueAlarm=True, form=True)
        self.root_nt = nt
        pv = SharedPV(nt=nt)
        self.root_pv = pv
        self._alias_readback = readback_suffix
        self._alias_setpoint = setpoint_suffix

        V = nt.wrap(0.0, timestamp=self._time_fn())
        V["display.units"] = self.default_units
        V["display.precision"] = self.default_precision

        if user_limits is not None:
            lo, hi = user_limits
            V["display.limitLow"] = lo
            V["display.limitHigh"] = hi
            V["control.limitLow"] = lo
            V["control.limitHigh"] = hi
            V["valueAlarm.lowAlarmLimit"] = lo
            V["valueAlarm.highAlarmLimit"] = hi

        V["alarm.severity"] = 0
        V["alarm.status"] = 0
        V["alarm.message"] = "NO_ALARM"
        pv.open(V)

        @pv.put
        def _alias_put(pv_obj, op):
            raw = op.value()
            try:
                new = float(raw["value"])
            except Exception as e:
                try:
                    new = float(raw)
                except Exception:
                    op.done(error=f"Bad put payload for {self.prefix}: {e}")
                    return
            if self._write_cb and self._alias_setpoint:
                self._write_cb(self._alias_setpoint, new)
            op.done()

    # ---------- posting helpers ----------

    def post_meta(self, suffix: str, updates: Dict[str, Any]) -> None:
        pv = self.pvs[suffix]
        nt = self.nts[suffix]
        cur = pv.current()
        try:
            cur_val = cur["value"]
        except Exception:
            cur_val = cur if not isinstance(cur, dict) else 0
        V = nt.wrap(cur_val, timestamp=self._time_fn())
        try:
            if nt.code() == "d":
                V["display.units"] = self.default_units
                V["display.precision"] = self.default_precision
        except Exception:
            pass
        for k, v in updates.items():
            try:
                V[k] = v
            except Exception:
                pass
        pv.post(V)

    def _should_throttle(self, suffix: str, value, severity: int, message: str, prev):
        if suffix not in self._mdel_fields:
            return False
        if not isinstance(value, (int, float)):
            return False
        if prev is None or not isinstance(prev[0], (int, float)):
            return False
        if self._mdel <= 0.0:
            return False
        return (
            abs(float(value) - float(prev[0])) < self._mdel
            and severity == prev[1]
            and message == prev[2]
        )

    def post_num(
        self,
        suffix: str,
        value: float | int,
        severity: int = 0,
        message: str = "",
        *,
        force: bool = False,
    ) -> None:
        prev = self._last_sent.get(suffix)
        if not force and self._should_throttle(suffix, value, severity, message, prev):
            return
        if not force and prev is not None and prev == (value, severity, message):
            return
        self.pvs[suffix].post(value, timestamp=self._time_fn(), severity=severity, message=message)
        self._last_sent[suffix] = (value, severity, message)

    # root alias posts (obey same MDEL as throttled fields)
    def post_root_num(
        self,
        value: float | int,
        severity: int = 0,
        message: str = "",
        *,
        force: bool = False,
    ) -> None:
        if self.root_pv is None:
            return
        prev = self._last_root
        if (
            not force
            and isinstance(value, (int, float))
            and prev is not None
            and isinstance(prev[0], (int, float))
            and self._mdel > 0.0
            and abs(float(value) - float(prev[0])) < self._mdel
            and severity == prev[1]
            and message == prev[2]
        ):
            return
        if not force and prev is not None and prev == (value, severity, message):
            return
        self.root_pv.post(value, timestamp=self._time_fn(), severity=severity, message=message)
        self._last_root = (value, severity, message)

    def post_root_meta(self, updates: Dict[str, Any]) -> None:
        if self.root_pv is None or self.root_nt is None:
            return
        cur = self.root_pv.current()
        try:
            cur_val = cur["value"]
        except Exception:
            cur_val = cur if not isinstance(cur, dict) else 0.0
        V = self.root_nt.wrap(cur_val, timestamp=self._time_fn())
        V["display.units"] = self.default_units
        V["display.precision"] = self.default_precision
        for k, v in updates.items():
            try:
                V[k] = v
            except Exception:
                pass
        self.root_pv.post(V)

    # ---------- bulk meta helpers ----------

    def update_float_meta_all(
        self,
        *,
        units: Optional[str] = None,
        precision: Optional[int] = None,
    ) -> None:
        for suffix, nt in self.nts.items():
            try:
                if nt.code() != "d":
                    continue
            except Exception:
                continue
            updates = {}
            if units is not None:
                updates["display.units"] = units
            if precision is not None:
                updates["display.precision"] = int(precision)
            if updates:
                self.post_meta(suffix, updates)

        alias_updates = {}
        if units is not None:
            alias_updates["display.units"] = units
        if precision is not None:
            alias_updates["display.precision"] = int(precision)
        if alias_updates:
            self.post_root_meta(alias_updates)


def build_provider_dict(groups: Dict[str, PVGroup]) -> Dict[str, SharedPV]:
    """
    Turn { '<prefix>': PVGroup } into the provider dict for p4p.Server.
    Exposes all PVs as '<prefix><join><suffix>' and the alias at '<prefix>' if present.

    - Motor-style records with fields: join='.' (default)
      e.g. 'SIM:M1.RBV', 'SIM:M1.VAL'
    - Non-field records (like chopper props): join=':'
      e.g. 'SIM:CHP1:Spd_S', 'SIM:CHP1:TotDly'
    """
    out: Dict[str, SharedPV] = {}
    for grp in groups.values():
        join = grp.name_join or "."
        out.update({f"{grp.prefix}{join}{k}": pv for k, pv in grp.pvs.items()})
        if grp.root_pv is not None:
            out[grp.prefix] = grp.root_pv
    return out
