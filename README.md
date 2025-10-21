# README

## Run with uv (no install)

Requirements: Python 3.9+ and [uv](https://docs.astral.sh/uv/).

From the `pv4u/` folder (the one containing `common/` and `sim/`):

```bash
# Motor
uv run -m sim.motor_record --help
uv run -m sim.motor_record --prefix SIM:M1 --tick-hz 20 --mdel 0.1

# Chopper
uv run -m sim.chopper --help
uv run -m sim.chopper --prefix SIM:CHP1: --chic-prefix SIM:CHIC1:
````

Ctrl+C to stop.

## Install the package (then use entry points)

```bash
uv pip install .
pv4u-motor --help
pv4u-motor --prefix SIM:M1
pv4u-chopper --help
pv4u-chopper --prefix SIM:CHP1:
```

