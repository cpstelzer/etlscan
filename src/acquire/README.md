# etlscan acquisition (`src/acquire`)

Hardware acquisition for depth-resolved imaging with electronically tunable lenses:
camera control (Basler pylon), focal scanning with the ETL (Optotune SDK) and the
optional XY stage for well plates (Zaber).

- Overview, entry points (`scripts/run_acquire*.py`) and scan settings: see the
  [main README](../../README.md#acquisition-workflows-srcacquire).
- Setting up the environment (`wellscan24_env`, including pypylon, `zaber.serial` and
  the Optotune Python SDK, which is obtained from Optotune and not distributed here): see
  [`wellscan24_env_files/howto_build_wellscan24_env.txt`](../../wellscan24_env_files/howto_build_wellscan24_env.txt).
