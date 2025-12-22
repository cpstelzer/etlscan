# scripts/run_pipeline.py
import os
import argparse
import logging
import sys
import time
import shutil
from typing import Optional, List, Union
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils import (
    setup_logging, load_config_once,
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
    try:
        config = load_config_once(args.config)
        setup_logging(config)
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Pipeline run initiated for Input Batch Directory: {args.input_batch_dir}")
        logger.info(f"Stages to run: {args.stages}")
    except Exception as e:
        print(f"FATAL: Error during setup: {e}")
        sys.exit(1)

    project_base_dir = get_project_base_dir(config)
    batch_output_csv_dir = project_base_dir
    os.makedirs(batch_output_csv_dir, exist_ok=True)
        
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

    valid_sample_ids = get_valid_sample_subdirectories(args.input_batch_dir)
    if not valid_sample_ids:
        logger.error(f"No valid sample subdirectories found in '{args.input_batch_dir}'. Aborting.")
        sys.exit(1)

    batch_dir_name = os.path.basename(os.path.normpath(args.input_batch_dir))
    if not batch_dir_name:
        sanitized_batch = args.input_batch_dir.strip().replace(os.sep, '_')
        batch_dir_name = sanitized_batch if sanitized_batch else 'batch_final_particles'

    aggregated_final_particles: List[pd.DataFrame] = []
    aggregated_stage3_focussed_particles: List[pd.DataFrame] = []
    aggregated_stage3_filtered_good_particles: List[pd.DataFrame] = []
    aggregated_stage4_analysis: List[pd.DataFrame] = []
    combined_filename = f"{batch_dir_name}.csv"
    stage3_combined_filename = f"{batch_dir_name}_stage3_focussed_and_filtered.csv"
    stage3_filtered_good_combined_filename = f"{batch_dir_name}_stage3_filtered_good.csv"
    batch_output_parent_dir = os.path.join(project_base_dir, "data", "output", batch_dir_name)
    os.makedirs(batch_output_parent_dir, exist_ok=True)
    stage4_combined_filename = f"{batch_dir_name}_stage4_final_analysis.csv"

    total_pipeline_start_time = time.time()
    for current_sample_id in valid_sample_ids:
        logger.info(f"vvv Processing Sample: {current_sample_id} vvv")
        sample_specific_start_time = time.time()

        # Define all paths upfront for clarity and independent checks
        sample_input_dir = os.path.join(args.input_batch_dir, current_sample_id)
        sample_output_base_dir = os.path.join(batch_output_parent_dir, current_sample_id)
        os.makedirs(sample_output_base_dir, exist_ok=True)

        # Stage directories
        s1_output_dir = os.path.join(sample_output_base_dir, "stage1_detection")
        s2_output_dir = os.path.join(sample_output_base_dir, "stage2_characterization")
        s3_output_dir = os.path.join(sample_output_base_dir, "stage3_filtering_classification")
        s4_output_dir = os.path.join(sample_output_base_dir, "stage4_biological_analysis")

        # Clean existing outputs for the stages being rerun to ensure fresh outputs
        if run_s1:
            _clean_output_directory(s1_output_dir, "Stage 1")
        if run_s2:
            _clean_output_directory(s2_output_dir, "Stage 2")
        if run_s3:
            _clean_output_directory(s3_output_dir, "Stage 3")
        if run_s4:
            _clean_output_directory(s4_output_dir, "Stage 4")

        # Ensure directories exist for the stages being run
        if run_s1: os.makedirs(s1_output_dir, exist_ok=True)
        if run_s2: os.makedirs(s2_output_dir, exist_ok=True)
        if run_s3: os.makedirs(s3_output_dir, exist_ok=True)
        if run_s4: os.makedirs(s4_output_dir, exist_ok=True)

        # Expected output file paths
        s1_final_csv_path = os.path.join(s1_output_dir, f"{current_sample_id}_stage1_tracked_particles.csv")
        s2_final_csv_path = os.path.join(s2_output_dir, f"{current_sample_id}_stage2_all_particle_features.csv")
        s2_final_particles_csv_path = os.path.join(s2_output_dir, 'final_particles.csv')
        s2_final_particles_crops_dir = os.path.join(s2_output_dir, 'final_particles')
        s3_final_csv_path = os.path.join(s3_output_dir, 'filtered_good.csv')
        s3_legacy_csv_path = os.path.join(s3_output_dir, 'focussed_and_filtered.csv')
        s3_filtered_crops_dir = os.path.join(s3_output_dir, 'filtered_good')
        s3_legacy_crops_dir = os.path.join(s3_output_dir, 'focussed_and_filtered')
        s2_all_crops_dir_path = os.path.join(s2_output_dir, "stage2_all_particle_crops")
        s4_final_csv_path = os.path.join(s4_output_dir, 'stage4_final_analysis.csv')

        try:
            # --- Stage 1: Detection & Pre-Filtering ---
            if run_s1:
                logger.info(f">>> Starting Stage 1 for {current_sample_id} <<<")
                s1_result_path = run_stage_01_initial_detection(
                    image_dir_path=sample_input_dir,
                    output_dir_stage1=s1_output_dir,
                    sample_id=current_sample_id,
                    config=config,
                )
                if not s1_result_path:
                    raise ProcessingError(f"Stage 1 failed for {current_sample_id}, cannot proceed.")
                logger.info(f"<<< Stage 1 for {current_sample_id} COMPLETED. >>>")

            # --- Stage 2: Universal Characterization ---
            if run_s2:
                logger.info(f">>> Starting Stage 2 for {current_sample_id} <<<")
                if not os.path.exists(s1_final_csv_path):
                    raise FileNotFoundError(f"Stage 1 output missing: {s1_final_csv_path}. Run Stage 1 first.")
                
                s2_result_path = run_stage_02_characterization(
                    input_s1_csv_path=s1_final_csv_path,
                    image_dir_path=sample_input_dir,
                    output_dir_stage2=s2_output_dir,
                    sample_id=current_sample_id,
                    config=config,
                )
                if not s2_result_path:
                    raise ProcessingError(f"Stage 2 failed for {current_sample_id}, cannot proceed.")
                logger.info(f"<<< Stage 2 for {current_sample_id} COMPLETED. >>>")

            # --- Stage 3: Filtering & Classification ---
            if run_s3:
                logger.info(f">>> Starting Stage 3 for {current_sample_id} <<<")
                if not os.path.exists(s2_final_particles_csv_path):
                    raise FileNotFoundError(
                        f"Stage 2 final particles output missing: {s2_final_particles_csv_path}."
                    )
                if not os.path.isdir(s2_final_particles_crops_dir):
                    raise FileNotFoundError(
                        f"Stage 2 final particle crops directory missing: {s2_final_particles_crops_dir}."
                    )

                s3_result_path = run_stage_03_filtering_and_classification(
                    input_final_particles_csv_path=s2_final_particles_csv_path,
                    output_dir_stage3=s3_output_dir,
                    final_particles_crops_dir=s2_final_particles_crops_dir,
                    sample_id=current_sample_id,
                    config=config,
                )
                if not s3_result_path:
                    raise ProcessingError(f"Stage 3 failed for {current_sample_id}, cannot proceed.")
                logger.info(f"<<< Stage 3 for {current_sample_id} COMPLETED. >>>")

            # --- Stage 4: Biological Analysis ---
            if run_s4:
                logger.info(f">>> Starting Stage 4 for {current_sample_id} <<<")
                stage3_csv_for_stage4 = (
                    s3_final_csv_path
                    if os.path.exists(s3_final_csv_path)
                    else s3_legacy_csv_path
                )
                if not os.path.exists(stage3_csv_for_stage4):
                    raise FileNotFoundError(
                        "Stage 3 output missing: "
                        f"{s3_final_csv_path} (preferred) / {s3_legacy_csv_path} (legacy). Run Stage 3 first."
                    )

                stage3_crops_for_stage4 = (
                    s3_filtered_crops_dir
                    if os.path.isdir(s3_filtered_crops_dir)
                    else s3_legacy_crops_dir
                )
                if not os.path.isdir(stage3_crops_for_stage4):
                    raise FileNotFoundError(
                        "Stage 3 filtered crops directory missing: "
                        f"{s3_filtered_crops_dir} (preferred) / {s3_legacy_crops_dir} (legacy)."
                    )

                s4_result_path = run_stage_04_biological_analysis(
                    input_s3_csv_path=stage3_csv_for_stage4,
                    stage3_crops_dir=stage3_crops_for_stage4,
                    output_dir_stage4=s4_output_dir,
                    sample_id=current_sample_id,
                    config=config,
                )
                if not s4_result_path:
                    logger.error(f"Stage 4 for {current_sample_id} FAILED or produced no output CSV.")
                else:
                    logger.info(f"<<< Stage 4 for {current_sample_id} COMPLETED. >>>")

            sample_specific_duration = time.time() - sample_specific_start_time
            logger.info(f"^^^ Sample {current_sample_id} processing FINISHED in {sample_specific_duration:.2f}s ^^^")

        except (FileNotFoundError, DataValidationError, ProcessingError, ConfigError, MLModelError) as e:
            logger.error(f"HALTED processing for {current_sample_id}: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error HALTED processing for {current_sample_id}: {e}")
        finally:
            final_particles_csv_path = os.path.join(s2_output_dir, 'final_particles.csv')
            if os.path.exists(final_particles_csv_path):
                final_particles_df = load_dataframe_from_csv(final_particles_csv_path)
                if final_particles_df is not None and not final_particles_df.empty:
                    final_particles_copy = final_particles_df.copy()
                    if 'sample_id' in final_particles_copy.columns:
                        final_particles_copy['sample_id'] = final_particles_copy['sample_id'].fillna(current_sample_id)
                    else:
                        final_particles_copy.insert(0, 'sample_id', current_sample_id)
                    aggregated_final_particles.append(final_particles_copy)
                    logger.info(
                        f"Collected {len(final_particles_copy)} Stage 2 final particles for batch aggregation from {current_sample_id}."
                    )

                    saved_batch_csv_paths = _write_batch_aggregation_csv(
                        aggregated_final_particles,
                        batch_output_csv_dir,
                        combined_filename
                    )
                    for saved_batch_csv_path in saved_batch_csv_paths:
                        if saved_batch_csv_path:
                            logger.info(
                                f"Batch-level Stage 2 final particles updated at {saved_batch_csv_path}."
                            )
                else:
                    logger.debug(
                        f"Stage 2 final particles CSV for {current_sample_id} is empty. Skipping aggregation for this sample."
                    )
            else:
                logger.debug(
                    f"Stage 2 final particles CSV not found for {current_sample_id}. Skipping aggregation for this sample."
                )

            if os.path.exists(s3_final_csv_path):
                stage3_filtered_df = load_dataframe_from_csv(s3_final_csv_path)
                if stage3_filtered_df is not None and not stage3_filtered_df.empty:
                    stage3_filtered_copy = stage3_filtered_df.copy()
                    if 'sample_id' in stage3_filtered_copy.columns:
                        stage3_filtered_copy['sample_id'] = stage3_filtered_copy['sample_id'].fillna(current_sample_id)
                    else:
                        stage3_filtered_copy.insert(0, 'sample_id', current_sample_id)
                    aggregated_stage3_filtered_good_particles.append(stage3_filtered_copy)
                    logger.info(
                        f"Collected {len(stage3_filtered_copy)} Stage 3 ML-accepted particles for batch aggregation from {current_sample_id}."
                    )

                    saved_stage3_filtered_batch_paths = _write_batch_aggregation_csv(
                        aggregated_stage3_filtered_good_particles,
                        batch_output_csv_dir,
                        stage3_filtered_good_combined_filename
                    )
                    for saved_stage3_batch_path in saved_stage3_filtered_batch_paths:
                        if saved_stage3_batch_path:
                            logger.info(
                                f"Batch-level Stage 3 filtered_good particles updated at {saved_stage3_batch_path}."
                            )
                else:
                    logger.debug(
                        f"Stage 3 filtered_good CSV for {current_sample_id} is empty. Skipping aggregation for this sample."
                    )
            else:
                logger.debug(
                    "Stage 3 filtered_good CSV not found for %s at %s. Skipping aggregation for this sample.",
                    current_sample_id,
                    s3_final_csv_path,
                )

            if os.path.exists(s3_legacy_csv_path):
                stage3_focussed_df = load_dataframe_from_csv(s3_legacy_csv_path)
                if stage3_focussed_df is not None and not stage3_focussed_df.empty:
                    stage3_focussed_copy = stage3_focussed_df.copy()
                    if 'sample_id' in stage3_focussed_copy.columns:
                        stage3_focussed_copy['sample_id'] = stage3_focussed_copy['sample_id'].fillna(current_sample_id)
                    else:
                        stage3_focussed_copy.insert(0, 'sample_id', current_sample_id)
                    aggregated_stage3_focussed_particles.append(stage3_focussed_copy)
                    logger.info(
                        f"Collected {len(stage3_focussed_copy)} Stage 3 pre-ML particles for batch aggregation from {current_sample_id}."
                    )

                    saved_stage3_focussed_batch_paths = _write_batch_aggregation_csv(
                        aggregated_stage3_focussed_particles,
                        batch_output_csv_dir,
                        stage3_combined_filename
                    )
                    for saved_stage3_batch_path in saved_stage3_focussed_batch_paths:
                        if saved_stage3_batch_path:
                            logger.info(
                                f"Batch-level Stage 3 focussed_and_filtered particles updated at {saved_stage3_batch_path}."
                            )
                else:
                    logger.debug(
                        f"Stage 3 focussed_and_filtered CSV for {current_sample_id} is empty. Skipping aggregation for this sample."
                    )
            else:
                logger.debug(
                    "Stage 3 focussed_and_filtered CSV not found for %s at %s. Skipping aggregation for this sample.",
                    current_sample_id,
                    s3_legacy_csv_path,
                )

            if os.path.exists(s4_final_csv_path):
                stage4_analysis_df = load_dataframe_from_csv(s4_final_csv_path)
                if stage4_analysis_df is not None and not stage4_analysis_df.empty:
                    stage4_analysis_copy = stage4_analysis_df.copy()
                    if 'sample_id' in stage4_analysis_copy.columns:
                        stage4_analysis_copy['sample_id'] = stage4_analysis_copy['sample_id'].fillna(current_sample_id)
                    else:
                        stage4_analysis_copy.insert(0, 'sample_id', current_sample_id)
                    aggregated_stage4_analysis.append(stage4_analysis_copy)
                    logger.info(
                        f"Collected {len(stage4_analysis_copy)} Stage 4 biological analysis records for batch aggregation from {current_sample_id}."
                    )

                    saved_stage4_batch_paths = _write_batch_aggregation_csv(
                        aggregated_stage4_analysis,
                        batch_output_csv_dir,
                        stage4_combined_filename
                    )
                    for saved_stage4_batch_path in saved_stage4_batch_paths:
                        if saved_stage4_batch_path:
                            logger.info(
                                f"Batch-level Stage 4 final analysis records updated at {saved_stage4_batch_path}."
                            )
                else:
                    logger.debug(
                        f"Stage 4 final analysis CSV for {current_sample_id} is empty. Skipping aggregation for this sample."
                    )
            else:
                logger.debug(
                    "Stage 4 final analysis CSV not found for %s at %s. Skipping aggregation for this sample.",
                    current_sample_id,
                    s4_final_csv_path,
                )

    total_pipeline_duration = time.time() - total_pipeline_start_time
    logger.info(f"--- Batch pipeline execution FINISHED in {total_pipeline_duration:.2f} seconds ---")

    if aggregated_final_particles:
        saved_batch_csv_paths = _write_batch_aggregation_csv(
            aggregated_final_particles,
            batch_output_csv_dir,
            combined_filename
        )
        for saved_batch_csv_path in saved_batch_csv_paths:
            if saved_batch_csv_path:
                logger.info(
                    f"Batch-level Stage 2 final particles saved to {saved_batch_csv_path}."
                )
    else:
        logger.info("No Stage 2 final particles were collected; skipping batch-level aggregation.")

    if aggregated_stage3_filtered_good_particles:
        saved_stage3_filtered_batch_paths = _write_batch_aggregation_csv(
            aggregated_stage3_filtered_good_particles,
            batch_output_csv_dir,
            stage3_filtered_good_combined_filename
        )
        for saved_stage3_batch_path in saved_stage3_filtered_batch_paths:
            if saved_stage3_batch_path:
                logger.info(
                    f"Batch-level Stage 3 filtered_good particles saved to {saved_stage3_batch_path}."
                )
    else:
        logger.info("No Stage 3 filtered_good particles were collected; skipping batch-level aggregation.")

    if aggregated_stage3_focussed_particles:
        saved_stage3_focussed_batch_paths = _write_batch_aggregation_csv(
            aggregated_stage3_focussed_particles,
            batch_output_csv_dir,
            stage3_combined_filename
        )
        for saved_stage3_batch_path in saved_stage3_focussed_batch_paths:
            if saved_stage3_batch_path:
                logger.info(
                    f"Batch-level Stage 3 focussed particles saved to {saved_stage3_batch_path}."
                )
    else:
        logger.info("No Stage 3 focussed particles were collected; skipping batch-level aggregation.")

    if aggregated_stage4_analysis:
        saved_stage4_batch_paths = _write_batch_aggregation_csv(
            aggregated_stage4_analysis,
            batch_output_csv_dir,
            stage4_combined_filename
        )
        for saved_stage4_batch_path in saved_stage4_batch_paths:
            if saved_stage4_batch_path:
                logger.info(
                    f"Batch-level Stage 4 final analysis records saved to {saved_stage4_batch_path}."
                )
    else:
        logger.info("No Stage 4 final analysis records were collected; skipping batch-level aggregation.")

    logging.shutdown()

if __name__ == "__main__":
    main()
