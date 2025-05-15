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
    save_dataframe_to_csv, get_project_base_dir 
)
from src.utils.path_helpers import get_valid_sample_subdirectories 
from src.utils.common_exceptions import ConfigError, ProcessingError, DataValidationError

from src.stages.stage_01_initial_detection import run_stage_01_initial_detection
from src.stages.stage_02_quality_filter import run_stage_02_quality_filter
from src.stages.stage_03_ml_classification import run_stage_03_ml_classification

logger = logging.getLogger(__name__) 

def main():
    parser = argparse.ArgumentParser(description="Run the Well Plate Analyzer pipeline in batch or for training.")
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
        help="Path to the parent directory containing one or more sample subdirectories "
             "(e.g., 'data/input/MyExperiment'). Each subdirectory should be named like 'YYYYMMDD_plateX_wellY'."
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all", 
        help="Comma-separated list of stages to run (e.g., '1,2,3' or 'all'). "
             "Use 'train_ml' to only train the ML model from Stage 3."
    )
    parser.add_argument(
        "--force_retrain_ml",
        action="store_true",
        help="If specified with Stage 3 or 'train_ml', forces retraining of the ML model "
             "even if a saved model exists."
    )

    args = parser.parse_args()

    try:
        loaded_main_config = load_config(args.config)
        # Setup logging early. If file logging is configured, it starts here.
        setup_logging() 
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Pipeline run initiated for Input Batch Directory: {args.input_batch_dir}")
        logger.info(f"Stages to run: {args.stages}")
        if args.stages in ["all", "3"] or "3" in args.stages.split(',') or args.stages == "train_ml":
             logger.info(f"Force ML model retraining: {args.force_retrain_ml}")

    except FileNotFoundError as e:
        # Logging might not be fully set up if config load fails before setup_logging,
        # so print as a fallback.
        print(f"FATAL: Configuration file not found: {e}. Pipeline cannot start.")
        sys.exit(1)
    except ConfigError as e:
        print(f"FATAL: Error in configuration: {e}. Pipeline cannot start.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Unexpected error during setup: {e}")
        sys.exit(1)

    stages_to_run_str = args.stages.lower()
    run_s1 = False
    run_s2 = False
    run_s3_predict = False
    run_s3_train = False

    if stages_to_run_str == "all":
        run_s1, run_s2, run_s3_predict = True, True, True
        if args.force_retrain_ml:
            run_s3_train = True
    elif stages_to_run_str == "train_ml":
        run_s3_train = True
    else:
        selected_stages = [s.strip() for s in stages_to_run_str.split(',')]
        if "1" in selected_stages: run_s1 = True
        if "2" in selected_stages: run_s2 = True
        if "3" in selected_stages:
            run_s3_predict = True
            if args.force_retrain_ml:
                 run_s3_train = True
    
    # Handle ML training mode first if specified
    if run_s3_train and not run_s1 and not run_s2 and not run_s3_predict: # Only 'train_ml'
        logger.info(">>> Running in ML Training ONLY mode <<<")
        try:
            context_id_for_training = os.path.basename(os.path.normpath(args.input_batch_dir))
            run_stage_03_ml_classification(
                input_dir_or_csv_path="", 
                output_dir_stage3="",    
                sample_id=context_id_for_training, 
                mode="train",
                force_model_retrain=args.force_retrain_ml
            )
            logger.info("--- ML Training ONLY mode COMPLETED ---")
        except Exception as e:
            logger.exception(f"Error during ML Training ONLY mode: {e}")
        finally:
            # Flush logs at the end of training-only mode
            for handler in logging.root.handlers:
                handler.flush()
            logging.shutdown()
            sys.exit(0) 

    # --- Batch Processing for Stages 1, 2, 3 Prediction ---
    valid_sample_ids = get_valid_sample_subdirectories(args.input_batch_dir)
    if not valid_sample_ids:
        logger.error(f"No valid sample subdirectories found in '{args.input_batch_dir}'. Aborting batch processing.")
        # Flush logs before exiting
        for handler in logging.root.handlers:
            handler.flush()
        sys.exit(1)

    batch_dir_name = os.path.basename(os.path.normpath(args.input_batch_dir))
    batch_output_parent_dir = os.path.join(get_project_base_dir(), "data", "output", batch_dir_name)
    os.makedirs(batch_output_parent_dir, exist_ok=True)
    logger.info(f"Batch output will be stored under: {batch_output_parent_dir}")

    # Define batch summary CSV path (for Stage 3 results)
    batch_s3_summary_filename = f"{batch_dir_name}_batch_stage3_ml_results.csv"
    batch_s3_summary_csv_path = os.path.join(batch_output_parent_dir, batch_s3_summary_filename)
    
    # If the batch summary CSV exists from a previous partial run, it will be appended to.
    # The header is handled by checking file existence before writing.
    if os.path.exists(batch_s3_summary_csv_path):
        logger.info(f"Batch Stage 3 summary CSV '{batch_s3_summary_csv_path}' exists. Will append new results.")
    else:
        logger.info(f"Batch Stage 3 summary CSV '{batch_s3_summary_csv_path}' will be created.")

    total_pipeline_start_time = time.time()

    for current_sample_id in valid_sample_ids:
        logger.info(f"vvv Processing Sample: {current_sample_id} vvv")
        sample_specific_start_time = time.time()

        s1_input_image_dir_for_sample = os.path.join(args.input_batch_dir, current_sample_id)
        
        sample_output_base_dir = os.path.join(batch_output_parent_dir, current_sample_id)
        os.makedirs(sample_output_base_dir, exist_ok=True)

        s1_output_dir = os.path.join(sample_output_base_dir, "stage1_initial_detection")
        s2_output_dir = os.path.join(sample_output_base_dir, "stage2_quality_filter")
        s3_output_dir = os.path.join(sample_output_base_dir, "stage3_ml_classification")
        
        os.makedirs(s1_output_dir, exist_ok=True)
        os.makedirs(s2_output_dir, exist_ok=True)
        os.makedirs(s3_output_dir, exist_ok=True)

        s1_output_csv_path: Optional[str] = None
        s2_analyzed_particles_csv_path: Optional[str] = None
        s2_sharp_objects_dir: Optional[str] = None

        try:
            # --- Stage 1: Initial Detection ---
            if run_s1:
                logger.info(f">>> Starting Stage 1 for {current_sample_id} <<<")
                s1_start_time = time.time()
                s1_output_csv_path = run_stage_01_initial_detection(
                    image_dir_path=s1_input_image_dir_for_sample,
                    output_dir_stage1=s1_output_dir,
                    sample_id=current_sample_id
                )
                s1_duration = time.time() - s1_start_time
                if not s1_output_csv_path:
                    logger.error(f"Stage 1 failed for {current_sample_id}.")
                    if run_s2: raise ProcessingError(f"S1 failed for {current_sample_id}, cannot proceed to S2.")
                else:
                    logger.info(f"<<< Stage 1 for {current_sample_id} COMPLETED in {s1_duration:.2f}s. Output: {s1_output_csv_path} >>>")
            elif run_s2 or run_s3_predict: 
                s1_output_csv_path = os.path.join(s1_output_dir, f"{current_sample_id}_stage1_particles.csv")
                if not os.path.exists(s1_output_csv_path):
                    raise FileNotFoundError(f"S1 output CSV '{s1_output_csv_path}' for {current_sample_id} not found.")
                logger.info(f"Using existing S1 output for {current_sample_id}: {s1_output_csv_path}")

            # --- Stage 2: Quality Filter ---
            if run_s2:
                if not s1_output_csv_path:
                    raise ProcessingError(f"S1 output CSV for {current_sample_id} not available for S2.")
                
                logger.info(f">>> Starting Stage 2 for {current_sample_id} <<<")
                s2_start_time = time.time()
                s2_analyzed_particles_csv_path = run_stage_02_quality_filter(
                    input_particles_csv_path=s1_output_csv_path,
                    image_dir_path=s1_input_image_dir_for_sample, 
                    output_dir_stage2=s2_output_dir,
                    sample_id=current_sample_id
                )
                s2_duration = time.time() - s2_start_time
                if not s2_analyzed_particles_csv_path:
                    logger.error(f"Stage 2 failed for {current_sample_id}.")
                    if run_s3_predict: raise ProcessingError(f"S2 failed for {current_sample_id}, cannot proceed to S3.")
                else:
                    logger.info(f"<<< Stage 2 for {current_sample_id} COMPLETED in {s2_duration:.2f}s. Analyzed CSV: {s2_analyzed_particles_csv_path} >>>")
                    s2_sharp_objects_dir = os.path.join(s2_output_dir, "sharp_object_final_crops")
            elif run_s3_predict: 
                s2_analyzed_particles_csv_path = os.path.join(s2_output_dir, f"{current_sample_id}_stage2_analyzed_particles.csv")
                s2_sharp_objects_dir = os.path.join(s2_output_dir, "sharp_object_final_crops")
                if not os.path.exists(s2_analyzed_particles_csv_path) or not os.path.isdir(s2_sharp_objects_dir):
                    raise FileNotFoundError(f"S2 outputs (CSV or sharp crops dir) for {current_sample_id} not found.")
                logger.info(f"Using existing S2 outputs for {current_sample_id}: CSV='{s2_analyzed_particles_csv_path}', Crops Dir='{s2_sharp_objects_dir}'")

            # --- Stage 3: ML Classification ---
            s3_mode_for_run = "predict" if run_s3_predict else None
            if run_s3_train and run_s3_predict: s3_mode_for_run = "train_and_predict"
            elif run_s3_train: s3_mode_for_run = "train" 

            if s3_mode_for_run: 
                if s3_mode_for_run in ["predict", "train_and_predict"]:
                    if not s2_sharp_objects_dir:
                        raise ProcessingError(f"S2 sharp objects dir for {current_sample_id} not available for S3 prediction.")
                    if not s2_analyzed_particles_csv_path:
                        raise ProcessingError(f"S2 analyzed particles CSV for {current_sample_id} not available for S3 filters.")
                
                logger.info(f">>> Starting Stage 3 for {current_sample_id} (Mode: {s3_mode_for_run}) <<<")
                s3_start_time = time.time()
                
                s3_output_final_csv_path_sample, s3_final_df_for_sample = run_stage_03_ml_classification(
                    input_dir_or_csv_path=s2_sharp_objects_dir if s2_sharp_objects_dir else "",
                    output_dir_stage3=s3_output_dir,
                    sample_id=current_sample_id,
                    stage2_analyzed_csv_path=s2_analyzed_particles_csv_path,
                    mode=s3_mode_for_run,
                    force_model_retrain=args.force_retrain_ml 
                )
                s3_duration = time.time() - s3_start_time
                if s3_output_final_csv_path_sample:
                    logger.info(f"<<< Stage 3 for {current_sample_id} (Mode: {s3_mode_for_run}) COMPLETED in {s3_duration:.2f}s. Sample CSV: {s3_output_final_csv_path_sample} >>>")
                    
                    # Append to batch summary CSV immediately
                    if run_s3_predict and s3_final_df_for_sample is not None and not s3_final_df_for_sample.empty:
                        try:
                            file_exists = os.path.exists(batch_s3_summary_csv_path)
                            s3_final_df_for_sample.to_csv(
                                batch_s3_summary_csv_path, 
                                mode='a', 
                                header=not file_exists, 
                                index=False,
                                float_format='%.4f' # Consistent float formatting
                            )
                            logger.info(f"Appended {len(s3_final_df_for_sample)} rows for sample {current_sample_id} to batch summary: {batch_s3_summary_csv_path}")
                        except Exception as e_append:
                            logger.exception(f"Error appending sample {current_sample_id} results to batch CSV: {e_append}")
                    elif run_s3_predict and s3_final_df_for_sample is not None and s3_final_df_for_sample.empty:
                         logger.info(f"Sample {current_sample_id} had no Stage 3 results to append to batch summary.")


                else:
                    logger.error(f"Stage 3 for {current_sample_id} (Mode: {s3_mode_for_run}) FAILED or produced no sample CSV.")
            
            sample_specific_duration = time.time() - sample_specific_start_time
            logger.info(f"^^^ Sample {current_sample_id} processing FINISHED in {sample_specific_duration:.2f}s ^^^")

        except FileNotFoundError as e:
            logger.error(f"Error processing {current_sample_id}: Required file/dir not found: {e}")
        except DataValidationError as e:
            logger.error(f"Error processing {current_sample_id}: Data validation error: {e}")
        except ProcessingError as e:
            logger.error(f"Error processing {current_sample_id}: Stage processing error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing {current_sample_id}: {e}")
        
        # Flush logs after processing each sample
        logger.info(f"Flushing logs after processing sample {current_sample_id}...")
        for handler in logging.root.handlers:
            handler.flush()
        
        logger.info("-" * 80) 

    # Batch processing loop finished
    total_pipeline_duration = time.time() - total_pipeline_start_time
    logger.info(f"--- Batch pipeline execution for '{args.input_batch_dir}' FINISHED in {total_pipeline_duration:.2f} seconds ---")
    
    if run_s3_predict:
        if os.path.exists(batch_s3_summary_csv_path):
             logger.info(f"Final batch Stage 3 summary CSV is available at: {batch_s3_summary_csv_path}")
        else:
             logger.warning(f"Batch Stage 3 summary CSV was expected but not found at: {batch_s3_summary_csv_path}. This might happen if no samples produced S3 results.")

    # Final log flush and shutdown
    for handler in logging.root.handlers:
        handler.flush()
    logging.shutdown()

if __name__ == "__main__":
    main()