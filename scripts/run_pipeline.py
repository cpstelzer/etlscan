# scripts/run_pipeline.py
import os
import argparse
import logging
import sys
import time
from typing import Optional, List
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


def _resolve_pipeline_log_directory(config: dict) -> str:
    """Return the directory where the pipeline run log is stored."""
    logging_config = config.get('logging', {}) if config else {}
    log_file_setting = logging_config.get('log_file')
    project_base_dir = get_project_base_dir()

    log_file_path: Optional[str] = None
    if log_file_setting and str(log_file_setting).lower() != 'null':
        if os.path.isabs(log_file_setting):
            log_file_path = log_file_setting
        else:
            log_file_path = os.path.join(project_base_dir, log_file_setting)
    else:
        log_file_path = os.path.join(project_base_dir, 'pipeline_run.log')

    log_dir = os.path.dirname(log_file_path)
    return log_dir if log_dir else project_base_dir


def _write_batch_aggregation_csv(
    aggregated_final_particles: List[pd.DataFrame],
    destination_dir: str,
    filename: str
) -> Optional[str]:
    """Persist the aggregated Stage 2 particles if any have been collected."""
    if not aggregated_final_particles:
        return None

    combined_final_particles_df = pd.concat(aggregated_final_particles, ignore_index=True)
    return save_dataframe_to_csv(combined_final_particles_df, destination_dir, filename)

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

    pipeline_log_dir = _resolve_pipeline_log_directory(config)
        
    stages_to_run_str = args.stages.lower()
    run_s1, run_s2, run_s3, run_s4 = False, False, False, False

    if stages_to_run_str == "all":
        run_s1, run_s2, run_s3, run_s4 = True, True, True, True
    else:
        selected_stages = [s.strip() for s in stages_to_run_str.split(',')]
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
    combined_filename = f"{batch_dir_name}.csv"
    batch_output_parent_dir = os.path.join(get_project_base_dir(), "data", "output", batch_dir_name)
    os.makedirs(batch_output_parent_dir, exist_ok=True)

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
        
        # Ensure directories exist for the stages being run
        if run_s1: os.makedirs(s1_output_dir, exist_ok=True)
        if run_s2: os.makedirs(s2_output_dir, exist_ok=True)
        if run_s3: os.makedirs(s3_output_dir, exist_ok=True)
        if run_s4: os.makedirs(s4_output_dir, exist_ok=True)

        # Expected output file paths
        s1_final_csv_path = os.path.join(s1_output_dir, f"{current_sample_id}_stage1_tracked_particles.csv")
        s2_final_csv_path = os.path.join(s2_output_dir, f"{current_sample_id}_stage2_all_particle_features.csv")
        s3_final_csv_path = os.path.join(s3_output_dir, f"{current_sample_id}_stage3_classified_good_particles.csv")
        s2_all_crops_dir_path = os.path.join(s2_output_dir, "stage2_all_particle_crops")

        try:
            # --- Stage 1: Detection & Pre-Filtering ---
            if run_s1:
                logger.info(f">>> Starting Stage 1 for {current_sample_id} <<<")
                s1_result_path = run_stage_01_initial_detection(
                    image_dir_path=sample_input_dir,
                    output_dir_stage1=s1_output_dir,
                    sample_id=current_sample_id
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
                    sample_id=current_sample_id
                )
                if not s2_result_path:
                    raise ProcessingError(f"Stage 2 failed for {current_sample_id}, cannot proceed.")
                logger.info(f"<<< Stage 2 for {current_sample_id} COMPLETED. >>>")

            # --- Stage 3: Filtering & Classification ---
            if run_s3:
                logger.info(f">>> Starting Stage 3 for {current_sample_id} <<<")
                if not os.path.exists(s2_final_csv_path):
                    raise FileNotFoundError(f"Stage 2 output missing: {s2_final_csv_path}. Run Stage 2 first.")
                
                s3_result_path = run_stage_03_filtering_and_classification(
                    input_s2_csv_path=s2_final_csv_path,
                    output_dir_stage3=s3_output_dir,
                    s2_crops_dir=s2_all_crops_dir_path,
                    sample_id=current_sample_id
                )
                if not s3_result_path:
                    raise ProcessingError(f"Stage 3 failed for {current_sample_id}, cannot proceed.")
                logger.info(f"<<< Stage 3 for {current_sample_id} COMPLETED. >>>")

            # --- Stage 4: Biological Analysis ---
            if run_s4:
                logger.info(f">>> Starting Stage 4 for {current_sample_id} <<<")
                if not os.path.exists(s3_final_csv_path):
                    raise FileNotFoundError(f"Stage 3 output missing: {s3_final_csv_path}. Run Stage 3 first.")
                if not os.path.isdir(s2_all_crops_dir_path):
                    raise FileNotFoundError(f"Stage 2 particle crops directory missing: {s2_all_crops_dir_path}. Run Stage 2 first.")

                s4_result_path = run_stage_04_biological_analysis(
                    input_s3_csv_path=s3_final_csv_path,
                    s2_crops_dir=s2_all_crops_dir_path,
                    output_dir_stage4=s4_output_dir,
                    sample_id=current_sample_id
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

                    saved_batch_csv_path = _write_batch_aggregation_csv(
                        aggregated_final_particles,
                        pipeline_log_dir,
                        combined_filename
                    )
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

    total_pipeline_duration = time.time() - total_pipeline_start_time
    logger.info(f"--- Batch pipeline execution FINISHED in {total_pipeline_duration:.2f} seconds ---")

    if aggregated_final_particles:
        saved_batch_csv_path = _write_batch_aggregation_csv(
            aggregated_final_particles,
            pipeline_log_dir,
            combined_filename
        )
        if saved_batch_csv_path:
            logger.info(
                f"Batch-level Stage 2 final particles saved to {saved_batch_csv_path}."
            )
    else:
        logger.info("No Stage 2 final particles were collected; skipping batch-level aggregation.")

    logging.shutdown()

if __name__ == "__main__":
    main()