# scripts/run_pipeline.py
import os
import argparse
import logging
import sys
import time
import shutil
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils import (
    setup_logging, get_config, load_config,
    save_dataframe_to_csv, get_project_base_dir,
    load_dataframe_from_csv
)
# --- MODIFIED: Added MLModelError to the import list ---
from src.utils.common_exceptions import ConfigError, ProcessingError, DataValidationError, MLModelError
from src.utils.path_helpers import get_valid_sample_subdirectories

# Import stage modules
from src.stages.stage_01_initial_detection import run_stage_01_initial_detection
from src.stages.stage_02_characterization import run_stage_02_characterization
from src.stages.stage_03_filtering_and_classification import run_stage_03_filtering_and_classification
from src.stages.stage_04_biological_analysis import run_stage_04_biological_analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    batch_dir: str
    batch_dir_name: str
    batch_output_parent_dir: str
    sample_input_dir: str
    sample_output_base_dir: str
    stage_dirs: Dict[str, str]
    stage_outputs: Dict[str, str]


def prepare_sample_paths(config: dict, batch_dir: str, sample_id: str) -> SamplePaths:
    """Build all per-sample paths without side effects."""
    batch_dir_name = os.path.basename(os.path.normpath(batch_dir))
    if not batch_dir_name:
        sanitized_batch = batch_dir.strip().replace(os.sep, "_")
        batch_dir_name = sanitized_batch if sanitized_batch else "batch_final_particles"

    project_base_dir = get_project_base_dir()
    batch_output_parent_dir = os.path.join(project_base_dir, "data", "output", batch_dir_name)
    sample_input_dir = os.path.join(batch_dir, sample_id)
    sample_output_base_dir = os.path.join(batch_output_parent_dir, sample_id)

    stage_dirs = {
        "1": os.path.join(sample_output_base_dir, "stage1_detection"),
        "2": os.path.join(sample_output_base_dir, "stage2_characterization"),
        "3": os.path.join(sample_output_base_dir, "stage3_filtering_classification"),
        "4": os.path.join(sample_output_base_dir, "stage4_biological_analysis"),
    }

    stage_outputs = {
        "s1_final_csv_path": os.path.join(
            stage_dirs["1"], f"{sample_id}_stage1_tracked_particles.csv"
        ),
        "s2_final_csv_path": os.path.join(
            stage_dirs["2"], f"{sample_id}_stage2_all_particle_features.csv"
        ),
        "s2_final_particles_csv_path": os.path.join(stage_dirs["2"], "final_particles.csv"),
        "s2_final_particles_crops_dir": os.path.join(stage_dirs["2"], "final_particles"),
        "s3_final_csv_path": os.path.join(stage_dirs["3"], "filtered_good.csv"),
        "s3_legacy_csv_path": os.path.join(stage_dirs["3"], "focussed_and_filtered.csv"),
        "s3_filtered_crops_dir": os.path.join(stage_dirs["3"], "filtered_good"),
        "s3_legacy_crops_dir": os.path.join(stage_dirs["3"], "focussed_and_filtered"),
        "s2_all_crops_dir_path": os.path.join(stage_dirs["2"], "stage2_all_particle_crops"),
        "s4_final_csv_path": os.path.join(stage_dirs["4"], "stage4_final_analysis.csv"),
    }

    return SamplePaths(
        sample_id=sample_id,
        batch_dir=batch_dir,
        batch_dir_name=batch_dir_name,
        batch_output_parent_dir=batch_output_parent_dir,
        sample_input_dir=sample_input_dir,
        sample_output_base_dir=sample_output_base_dir,
        stage_dirs=stage_dirs,
        stage_outputs=stage_outputs,
    )


def _write_batch_aggregation_csv(
    aggregated_frames: List[pd.DataFrame],
    destination_dir: str,
    filenames: Union[str, List[str]],
) -> List[Optional[str]]:
    """Persist any aggregated particle records that have been collected."""
    if not aggregated_frames:
        return []

    if isinstance(filenames, str):
        filenames_to_write = [filenames]
    else:
        filenames_to_write = list(dict.fromkeys(filenames))

    combined_df = pd.concat(aggregated_frames, ignore_index=True)
    saved_paths: List[Optional[str]] = []
    for filename in filenames_to_write:
        saved_paths.append(save_dataframe_to_csv(combined_df, destination_dir, filename))

    return saved_paths


def _clean_output_directory(output_dir: str, stage_label: str) -> None:
    """Remove an existing stage output directory so it can be recreated cleanly."""
    if os.path.isdir(output_dir):
        logger.info(f"Cleaning existing {stage_label} output at: {output_dir}")
        shutil.rmtree(output_dir)
    else:
        logger.debug(f"No existing {stage_label} output to clean at: {output_dir}")


def accumulate_stage_outputs(
    csv_path: str,
    aggregated_frames: List[pd.DataFrame],
    sample_id: str,
    label: str,
) -> Optional[pd.DataFrame]:
    """Load and append a stage output to the aggregated frames list."""
    if not os.path.exists(csv_path):
        logger.debug("%s CSV not found for %s at %s. Skipping aggregation.", label, sample_id, csv_path)
        return None

    dataframe = load_dataframe_from_csv(csv_path)
    if dataframe is None or dataframe.empty:
        logger.debug("%s CSV for %s is empty. Skipping aggregation for this sample.", label, sample_id)
        return None

    dataframe_copy = dataframe.copy()
    if "sample_id" in dataframe_copy.columns:
        dataframe_copy["sample_id"] = dataframe_copy["sample_id"].fillna(sample_id)
    else:
        dataframe_copy.insert(0, "sample_id", sample_id)
    aggregated_frames.append(dataframe_copy)
    logger.info("Collected %d %s records for batch aggregation from %s.", len(dataframe_copy), label, sample_id)
    return dataframe_copy


def write_batch_rollups(
    aggregated_frames: List[pd.DataFrame],
    destination_dir: str,
    filenames: Union[str, List[str]],
    label: str,
) -> None:
    """Persist aggregated frames to batch rollup files and log their locations."""
    saved_paths = _write_batch_aggregation_csv(aggregated_frames, destination_dir, filenames)
    for saved_path in saved_paths:
        if saved_path:
            logger.info("Batch-level %s updated at %s.", label, saved_path)


def run_stage(stage_id: str, sample_paths: SamplePaths, config: dict) -> Optional[str]:
    """Execute a stage based on its identifier using prepared sample paths."""
    stage_outputs = sample_paths.stage_outputs
    sample_id = sample_paths.sample_id
    sample_input_dir = sample_paths.sample_input_dir
    stage_dirs = sample_paths.stage_dirs

    if stage_id == "1":
        logger.info(f">>> Starting Stage 1 for {sample_id} <<<")
        result_path = run_stage_01_initial_detection(
            image_dir_path=sample_input_dir,
            output_dir_stage1=stage_dirs["1"],
            sample_id=sample_id
        )
        if not result_path:
            raise ProcessingError(f"Stage 1 failed for {sample_id}, cannot proceed.")
        logger.info(f"<<< Stage 1 for {sample_id} COMPLETED. >>>")
        return result_path

    if stage_id == "2":
        logger.info(f">>> Starting Stage 2 for {sample_id} <<<")
        if not os.path.exists(stage_outputs["s1_final_csv_path"]):
            raise FileNotFoundError(
                f"Stage 1 output missing: {stage_outputs['s1_final_csv_path']}. Run Stage 1 first."
            )
        result_path = run_stage_02_characterization(
            input_s1_csv_path=stage_outputs["s1_final_csv_path"],
            image_dir_path=sample_input_dir,
            output_dir_stage2=stage_dirs["2"],
            sample_id=sample_id
        )
        if not result_path:
            raise ProcessingError(f"Stage 2 failed for {sample_id}, cannot proceed.")
        logger.info(f"<<< Stage 2 for {sample_id} COMPLETED. >>>")
        return result_path

    if stage_id == "3":
        logger.info(f">>> Starting Stage 3 for {sample_id} <<<")
        if not os.path.exists(stage_outputs["s2_final_particles_csv_path"]):
            raise FileNotFoundError(
                f"Stage 2 final particles output missing: {stage_outputs['s2_final_particles_csv_path']}."
            )
        if not os.path.isdir(stage_outputs["s2_final_particles_crops_dir"]):
            raise FileNotFoundError(
                f"Stage 2 final particle crops directory missing: {stage_outputs['s2_final_particles_crops_dir']}."
            )

        result_path = run_stage_03_filtering_and_classification(
            input_final_particles_csv_path=stage_outputs["s2_final_particles_csv_path"],
            output_dir_stage3=stage_dirs["3"],
            final_particles_crops_dir=stage_outputs["s2_final_particles_crops_dir"],
            sample_id=sample_id
        )
        if not result_path:
            raise ProcessingError(f"Stage 3 failed for {sample_id}, cannot proceed.")
        logger.info(f"<<< Stage 3 for {sample_id} COMPLETED. >>>")
        return result_path

    if stage_id == "4":
        logger.info(f">>> Starting Stage 4 for {sample_id} <<<")
        stage3_csv_for_stage4 = (
            stage_outputs["s3_final_csv_path"]
            if os.path.exists(stage_outputs["s3_final_csv_path"])
            else stage_outputs["s3_legacy_csv_path"]
        )
        if not os.path.exists(stage3_csv_for_stage4):
            raise FileNotFoundError(
                "Stage 3 output missing: "
                f"{stage_outputs['s3_final_csv_path']} (preferred) / {stage_outputs['s3_legacy_csv_path']} (legacy). Run Stage 3 first."
            )

        stage3_crops_for_stage4 = (
            stage_outputs["s3_filtered_crops_dir"]
            if os.path.isdir(stage_outputs["s3_filtered_crops_dir"])
            else stage_outputs["s3_legacy_crops_dir"]
        )
        if not os.path.isdir(stage3_crops_for_stage4):
            raise FileNotFoundError(
                "Stage 3 filtered crops directory missing: "
                f"{stage_outputs['s3_filtered_crops_dir']} (preferred) / {stage_outputs['s3_legacy_crops_dir']} (legacy)."
            )

        result_path = run_stage_04_biological_analysis(
            input_s3_csv_path=stage3_csv_for_stage4,
            stage3_crops_dir=stage3_crops_for_stage4,
            output_dir_stage4=stage_dirs["4"],
            sample_id=sample_id
        )
        if not result_path:
            logger.error(f"Stage 4 for {sample_id} FAILED or produced no output CSV.")
        else:
            logger.info(f"<<< Stage 4 for {sample_id} COMPLETED. >>>")
        return result_path

    raise ValueError(f"Unknown stage identifier '{stage_id}'")

def main():
    parser = argparse.ArgumentParser(description="Run the Well Plate Analyzer pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(project_root, "config", "pipeline_config.yml"),
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--input_batch_dir",
        type=str,
        required=True,
        help="Path to the parent directory containing sample subdirectories."
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated list of stages to run (e.g., '1,2,3,4' or 'all')."
    )
    # Note: ML training is now handled by a dedicated 'scripts/train_model.py' script

    args = parser.parse_args()

    # --- Setup ---
    config = {}

    try:
        load_config(args.config)
        setup_logging()
        config = get_config()
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Pipeline run initiated for Input Batch Directory: {args.input_batch_dir}")
        logger.info(f"Stages to run: {args.stages}")
    except Exception as e:
        print(f"FATAL: Error during setup: {e}")
        sys.exit(1)

    # Select stages to run
    stages_to_run_str = args.stages.lower()
    run_s1, run_s2, run_s3, run_s4 = False, False, False, False

    if stages_to_run_str == "all":
        run_s1, run_s2, run_s3, run_s4 = True, True, True, True
    else:
        selected_stages = [s.strip() for s in stages_to_run_str.split(',') if s.strip()]
        if "1" in selected_stages: run_s1 = True
        if "2" in selected_stages: run_s2 = True
        if "3" in selected_stages: run_s3 = True
        if "4" in selected_stages: run_s4 = True

    # Identify samples and derive batch-level paths
    valid_sample_ids = get_valid_sample_subdirectories(args.input_batch_dir)
    if not valid_sample_ids:
        logger.error(f"No valid sample subdirectories found in '{args.input_batch_dir}'. Aborting.")
        sys.exit(1)

    sample_paths_template = prepare_sample_paths(config, args.input_batch_dir, valid_sample_ids[0])
    project_base_dir = get_project_base_dir()
    batch_output_csv_dir = project_base_dir
    batch_output_parent_dir = sample_paths_template.batch_output_parent_dir
    os.makedirs(batch_output_csv_dir, exist_ok=True)
    os.makedirs(batch_output_parent_dir, exist_ok=True)

    batch_dir_name = sample_paths_template.batch_dir_name

    aggregated_final_particles: List[pd.DataFrame] = []
    aggregated_stage3_focussed_particles: List[pd.DataFrame] = []
    aggregated_stage3_filtered_good_particles: List[pd.DataFrame] = []
    aggregated_stage4_analysis: List[pd.DataFrame] = []
    combined_filename = f"{batch_dir_name}.csv"
    stage3_combined_filename = f"{batch_dir_name}_stage3_focussed_and_filtered.csv"
    stage3_filtered_good_combined_filename = f"{batch_dir_name}_stage3_filtered_good.csv"
    stage4_combined_filename = f"{batch_dir_name}_stage4_final_analysis.csv"

    total_pipeline_start_time = time.time()
    for current_sample_id in valid_sample_ids:
        logger.info(f"vvv Processing Sample: {current_sample_id} vvv")
        sample_specific_start_time = time.time()

        # Prepare paths and directories
        sample_paths = prepare_sample_paths(config, args.input_batch_dir, current_sample_id)
        os.makedirs(sample_paths.sample_output_base_dir, exist_ok=True)

        stage_execution_plan = [
            {"id": "1", "label": "Stage 1", "should_run": run_s1},
            {"id": "2", "label": "Stage 2", "should_run": run_s2},
            {"id": "3", "label": "Stage 3", "should_run": run_s3},
            {"id": "4", "label": "Stage 4", "should_run": run_s4},
        ]

        for stage in stage_execution_plan:
            if stage["should_run"]:
                _clean_output_directory(sample_paths.stage_dirs[stage["id"]], stage["label"])
                os.makedirs(sample_paths.stage_dirs[stage["id"]], exist_ok=True)

        try:
            # Run stages in a data-driven loop
            for stage in stage_execution_plan:
                if not stage["should_run"]:
                    continue
                run_stage(stage["id"], sample_paths, config)

            sample_specific_duration = time.time() - sample_specific_start_time
            logger.info(f"^^^ Sample {current_sample_id} processing FINISHED in {sample_specific_duration:.2f}s ^^^")

        except (FileNotFoundError, DataValidationError, ProcessingError, ConfigError, MLModelError) as e:
            logger.error(f"HALTED processing for {current_sample_id}: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error HALTED processing for {current_sample_id}: {e}")
        finally:
            # Aggregate outputs for batch rollups
            aggregation_jobs = [
                {
                    "csv_path": sample_paths.stage_outputs["s2_final_particles_csv_path"],
                    "aggregated_list": aggregated_final_particles,
                    "filename": combined_filename,
                    "label": "Stage 2 final particles",
                },
                {
                    "csv_path": sample_paths.stage_outputs["s3_final_csv_path"],
                    "aggregated_list": aggregated_stage3_filtered_good_particles,
                    "filename": stage3_filtered_good_combined_filename,
                    "label": "Stage 3 filtered_good particles",
                },
                {
                    "csv_path": sample_paths.stage_outputs["s3_legacy_csv_path"],
                    "aggregated_list": aggregated_stage3_focussed_particles,
                    "filename": stage3_combined_filename,
                    "label": "Stage 3 focussed_and_filtered particles",
                },
                {
                    "csv_path": sample_paths.stage_outputs["s4_final_csv_path"],
                    "aggregated_list": aggregated_stage4_analysis,
                    "filename": stage4_combined_filename,
                    "label": "Stage 4 final analysis records",
                },
            ]

            for job in aggregation_jobs:
                aggregated_df = accumulate_stage_outputs(
                    job["csv_path"],
                    job["aggregated_list"],
                    current_sample_id,
                    job["label"],
                )
                if aggregated_df is not None:
                    write_batch_rollups(
                        job["aggregated_list"],
                        batch_output_csv_dir,
                        job["filename"],
                        job["label"],
                    )

    total_pipeline_duration = time.time() - total_pipeline_start_time
    logger.info(f"--- Batch pipeline execution FINISHED in {total_pipeline_duration:.2f} seconds ---")

    # Persist final batch rollups
    if aggregated_final_particles:
        write_batch_rollups(
            aggregated_final_particles,
            batch_output_csv_dir,
            combined_filename,
            "Stage 2 final particles",
        )
    else:
        logger.info("No Stage 2 final particles were collected; skipping batch-level aggregation.")

    if aggregated_stage3_filtered_good_particles:
        write_batch_rollups(
            aggregated_stage3_filtered_good_particles,
            batch_output_csv_dir,
            stage3_filtered_good_combined_filename,
            "Stage 3 filtered_good particles",
        )
    else:
        logger.info("No Stage 3 filtered_good particles were collected; skipping batch-level aggregation.")

    if aggregated_stage3_focussed_particles:
        write_batch_rollups(
            aggregated_stage3_focussed_particles,
            batch_output_csv_dir,
            stage3_combined_filename,
            "Stage 3 focussed particles",
        )
    else:
        logger.info("No Stage 3 focussed particles were collected; skipping batch-level aggregation.")

    if aggregated_stage4_analysis:
        write_batch_rollups(
            aggregated_stage4_analysis,
            batch_output_csv_dir,
            stage4_combined_filename,
            "Stage 4 final analysis records",
        )
    else:
        logger.info("No Stage 4 final analysis records were collected; skipping batch-level aggregation.")

    logging.shutdown()

if __name__ == "__main__":
    main()
