"""config.yml is the only source of parameter values.

Two static checks keep it that way:

1. every key in ``config/config.yml`` is read somewhere in ``src/`` or ``scripts/`` (no
   dead parameters that look tunable but change nothing);
2. no code in ``src/`` reads a config key with ``.get(...)``: a missing key must fail
   (``require_config_keys`` / direct indexing), never fall back to a value in code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest
import yaml

from src.utils.common_exceptions import ConfigError
from src.utils.config_loader import require_config_keys, require_config_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yml"

# Sections whose children are data handed on as a whole, not keys the code names one by one.
_OPAQUE_SECTIONS = (
    # hardware profiles are deep-merged into `acquisition`; their keys are checked there
    ("hardware_profiles",),
    # plate geometry per plate type ("24", "96")
    ("acquisition", "plate", "definitions"),
    # parameter grid passed to GridSearchCV as it is
    ("stage_03_filtering_and_classification", "training_params", "rf_param_grid"),
)

# Keys in config.yml that also name a column / field of a data row or result dict, where
# `.get` is legitimate. Each entry lists the files allowed to use `.get('<key>')` on such data.
_DATA_KEYS_ALLOWED_IN = {
    # fields stored in the EfficientNet checkpoint file at training time
    "positive_label": ("src/stages/stage_03_filtering_and_classification.py",),
    "input_size": ("src/stages/stage_03_filtering_and_classification.py",),
}

# The loader itself: its transforms run on whatever sections a config holds (tests load
# partial configs), and it is the one place that assembles sections rather than reading values.
_LOADER = "src/utils/config_loader.py"


def _config_key_paths(node: Dict, path: Tuple[str, ...] = ()) -> Iterator[Tuple[str, ...]]:
    """Yield the path of every key in the config tree, stopping at opaque sections."""
    for key, value in node.items():
        key_path = path + (str(key),)
        yield key_path
        if isinstance(value, dict) and key_path not in _OPAQUE_SECTIONS:
            yield from _config_key_paths(value, key_path)


def _python_sources(*top_level_dirs: str) -> Dict[str, str]:
    sources = {}
    for top_level_dir in top_level_dirs:
        for source_path in sorted((REPO_ROOT / top_level_dir).rglob("*.py")):
            sources[source_path.relative_to(REPO_ROOT).as_posix()] = source_path.read_text(encoding="utf-8")
    return sources


@pytest.fixture(scope="module")
def config_key_paths() -> List[Tuple[str, ...]]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return list(_config_key_paths(yaml.safe_load(handle)))


def test_every_config_key_is_read_by_the_code(config_key_paths):
    code = "\n".join(_python_sources("src", "scripts").values())
    # Children of an opaque section are data; the section key itself must still be read.
    unread = [
        ".".join(key_path)
        for key_path in config_key_paths
        if not any(key_path[:len(opaque)] == opaque and len(key_path) > len(opaque) for opaque in _OPAQUE_SECTIONS)
        and not re.search(r"""['"]%s['"]""" % re.escape(key_path[-1]), code)
    ]
    assert not unread, (
        "config.yml keys that no code in src/ or scripts/ reads (remove them, or read them): "
        + ", ".join(unread)
    )


def test_no_config_key_is_read_with_a_fallback(config_key_paths):
    key_names = sorted({key_path[-1] for key_path in config_key_paths}, key=len, reverse=True)
    get_call = re.compile(r"""\.get\(\s*['"](%s)['"]""" % "|".join(re.escape(name) for name in key_names))

    offenders = []
    for source_name, source_text in _python_sources("src").items():
        if source_name == _LOADER:
            continue
        for line_number, line in enumerate(source_text.splitlines(), start=1):
            for match in get_call.finditer(line):
                if source_name in _DATA_KEYS_ALLOWED_IN.get(match.group(1), ()):
                    continue
                offenders.append(f"{source_name}:{line_number}: {line.strip()}")
    assert not offenders, (
        "Config keys must be read with require_config_keys / direct indexing, not .get():\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# The helpers the code uses instead of fallbacks.
# ---------------------------------------------------------------------------
def test_require_config_keys_names_each_missing_key_with_its_path():
    section = {"search_range_mm": 1.0}
    with pytest.raises(ConfigError) as error:
        require_config_keys(section, ("search_range_mm", "memory_frames", "min_track_length"),
                            "stage_01_initial_detection.tracking")
    message = str(error.value)
    assert "stage_01_initial_detection.tracking.memory_frames" in message
    assert "stage_01_initial_detection.tracking.min_track_length" in message
    assert "search_range_mm" not in message


def test_require_config_keys_returns_the_section_when_complete():
    section = {"a": 0, "b": None}   # falsy values are values, not missing keys
    assert require_config_keys(section, ("a", "b"), "some.section") is section


def test_require_config_keys_rejects_a_non_mapping_section():
    with pytest.raises(ConfigError, match="some.section"):
        require_config_keys(None, ("a",), "some.section")


def test_require_config_sections_walks_nested_sections():
    config = {"outer": {"inner": {"a": 1}}, "flat": 2}
    require_config_sections(config, {"": ("outer", "flat"), "outer.inner": ("a",)}, "root")
    with pytest.raises(ConfigError, match="root.outer.inner.b"):
        require_config_sections(config, {"outer.inner": ("a", "b")}, "root")
    with pytest.raises(ConfigError, match="root.outer.missing"):
        require_config_sections(config, {"outer.missing": ("a",)}, "root")


def test_stage1_fails_before_processing_when_a_key_is_missing():
    from src.stages.stage_01_initial_detection import validate_stage1_config
    from src.utils import config_loader

    config_loader.load_config_once.cache_clear()
    shipped_config = config_loader.load_config_once(str(CONFIG_PATH))
    validate_stage1_config(shipped_config)                          # the shipped config is complete
    validate_stage1_config(shipped_config, external_background=True)

    incomplete = dict(shipped_config)
    incomplete["stage_01_initial_detection"] = {
        name: section for name, section in shipped_config["stage_01_initial_detection"].items()
        if name != "tracking"
    }
    with pytest.raises(ConfigError, match="stage_01_initial_detection.tracking"):
        validate_stage1_config(incomplete)
