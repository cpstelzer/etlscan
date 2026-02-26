import argparse
import logging
import os
import sys
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.realtime import load_realtime_config, run_realtime_session


class _SuppressNoisyMeasurementWarnings(logging.Filter):
    """Hide repetitive non-finite orientation/eccentricity warnings in realtime CLI output."""

    _suppressed_fragments = (
        "Measurement 'orientation' is non-finite",
        "Measurement 'eccentricity' is non-finite",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        message = record.getMessage()
        return not any(fragment in message for fragment in self._suppressed_fragments)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime acquisition + processing mode.")
    parser.add_argument(
        "--ws-config",
        default=os.path.join("src", "acquire", "ws_config.yaml"),
        help="Path to ws_config.yaml (runtime/acquisition settings).",
    )
    parser.add_argument(
        "--pipeline-config",
        default=os.path.join("config", "pipeline_config.yml"),
        help="Path to pipeline_config.yml (shared analysis settings).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Deprecated alias for --ws-config.",
    )
    args = parser.parse_args()

    ws_config_path = args.ws_config
    if args.config:
        warnings.warn(
            "--config is deprecated and will be removed in a future version; use --ws-config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        ws_config_path = args.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(_SuppressNoisyMeasurementWarnings())

    cfg = load_realtime_config(ws_config_path, pipeline_config_path=args.pipeline_config)
    if not cfg.enabled:
        raise ValueError("realtime_mode.enabled is false in the provided config")

    plate = int(input("Enter plate number: ").strip())
    well = int(input("Enter well number: ").strip())

    output_dir = run_realtime_session(cfg, plate=plate, well=well)
    logging.info("Realtime run completed. Outputs saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
