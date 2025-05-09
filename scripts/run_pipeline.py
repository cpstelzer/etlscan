# scripts/run_pipeline.py
import os
import argparse
import logging # Logging will be set up by utils
import sys
import time
from typing import Optional 


# Ensure the src directory is in the Python path
# This allows the script to be run from the 'scripts' directory or project root
# and still find the 'src' package.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming scripts is one level down from project root
sys.path.insert(0, project_root)

from src.utils import (
    setup_logging, get_config, load_config,
    get_sample_dir, get_output_dir_for_sample # For constructing default paths
)
from src.utils.common_exceptions import ConfigError, ProcessingError, DataValidationError

# Import stage functions
from src.stages.stage_01_initial_detection import run_stage_01_initial_detection
from src.stages.stage_02_quality_filter import run_stage_02_quality_filter
from src.stages.stage_03_ml_classification import run_stage_03_ml_classification

# Initialize logger early, but setup_logging will configure it based on the loaded config
logger = logging.getLogger(__name__) # Get a logger for this script

def main():
    parser = argparse.ArgumentParser(description="Run the Well Plate Analyzer pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(project_root, "config", "pipeline_config.yml"), # Default config path
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        required=True, # Make sample_id mandatory for running the full data pipeline
        help="Identifier for the sample to process (e.g., '20250410_plate4_well4'). "
             "This corresponds to a subdirectory name in 'data/input/'."
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all", # "all", "1", "2", "3", "1,2", "2,3", etc. or "train_ml"
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

    # --- 1. Load Configuration and Setup Logging ---
    try:
        # Load config using the provided path or default
        # The load_config function in utils ensures it's loaded once globally
        loaded_main_config = load_config(args.config)
        setup_logging() # Now setup logging using the loaded configuration
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Pipeline run initiated for Sample ID: {args.sample_id}")
        logger.info(f"Stages to run: {args.stages}")
        if args.stages in ["all", "3"] or "3" in args.stages.split(',') or args.stages == "train_ml":
             logger.info(f"Force ML model retraining: {args.force_retrain_ml}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Pipeline cannot start.")
        sys.exit(1)
    except ConfigError as e:
        logger.error(f"Error in configuration: {e}. Pipeline cannot start.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        sys.exit(1)

    # --- 2. Determine Stages to Run ---
    stages_to_run_str = args.stages.lower()
    run_s1 = False
    run_s2 = False
    run_s3_predict = False
    run_s3_train = False

    if stages_to_run_str == "all":
        run_s1, run_s2, run_s3_predict = True, True, True
        # By default, 'all' implies predict with existing model unless --force_retrain_ml
        if args.force_retrain_ml:
            run_s3_train = True
    elif stages_to_run_str == "train_ml":
        run_s3_train = True
    else:
        selected_stages = [s.strip() for s in stages_to_run_str.split(',')]
        if "1" in selected_stages: run_s1 = True
        if "2" in selected_stages: run_s2 = True
        if "3" in selected_stages:
            run_s3_predict = True # Default for stage 3 is predict
            if args.force_retrain_ml: # If specified with stage 3, also train
                 run_s3_train = True

    # --- 3. Define Paths (based on config and sample_id) ---
    # Input directory for Stage 1 (raw images for the sample)
    # get_sample_dir constructs data/input/{sample_id}
    s1_input_image_dir = get_sample_dir(args.sample_id)

    # Base output directory for the current sample
    # get_output_dir_for_sample constructs data/output/{sample_id}
    sample_output_base_dir = get_output_dir_for_sample(args.sample_id)

    # Specific output directories for each stage
    s1_output_dir = os.path.join(sample_output_base_dir, "stage1_initial_detection")
    s2_output_dir = os.path.join(sample_output_base_dir, "stage2_quality_filter")
    s3_output_dir = os.path.join(sample_output_base_dir, "stage3_ml_classification") # Base for S3 outputs
    
    os.makedirs(s1_output_dir, exist_ok=True)
    os.makedirs(s2_output_dir, exist_ok=True)
    os.makedirs(s3_output_dir, exist_ok=True)


    # Paths for data flowing between stages
    s1_output_csv_path: Optional[str] = None
    s2_output_csv_path: Optional[str] = None # CSV of all analyzed particles from S2
    s2_sharp_objects_dir: Optional[str] = None # Dir of sharp object crops from S2

    # --- 4. Execute Pipeline Stages ---
    pipeline_start_time = time.time()

    try:
        # --- Stage 1: Initial Detection ---
        if run_s1:
            logger.info(">>> Starting Stage 1: Initial Particle Detection <<<")
            s1_start_time = time.time()
            s1_output_csv_path = run_stage_01_initial_detection(
                image_dir_path=s1_input_image_dir,
                output_dir_stage1=s1_output_dir,
                sample_id=args.sample_id
            )
            s1_duration = time.time() - s1_start_time
            if not s1_output_csv_path:
                logger.error("Stage 1 failed to produce an output CSV. Subsequent stages depending on it may fail.")
                # Decide if pipeline should stop or attempt to proceed if possible
                # For now, let's stop if S1 output is crucial for S2
                if run_s2: raise ProcessingError("Stage 1 failed, cannot proceed to Stage 2.")
            else:
                logger.info(f"<<< Stage 1 COMPLETED in {s1_duration:.2f} seconds. Output CSV: {s1_output_csv_path} >>>")
        elif run_s2 or run_s3_predict: # If running S2 or S3_predict but not S1, need S1 output path
            s1_output_csv_path = os.path.join(s1_output_dir, f"{args.sample_id}_stage1_particles.csv")
            if not os.path.exists(s1_output_csv_path):
                logger.error(f"Stage 1 output CSV '{s1_output_csv_path}' not found, but required for subsequent stages.")
                raise FileNotFoundError(f"Missing Stage 1 output: {s1_output_csv_path}")
            logger.info(f"Using existing Stage 1 output: {s1_output_csv_path}")


        # --- Stage 2: Quality Filter ---
        if run_s2:
            if not s1_output_csv_path: # Should have been caught above
                logger.error("Stage 1 output CSV path is not available. Cannot run Stage 2.")
                raise ProcessingError("Missing S1 output for S2.")
            
            logger.info(">>> Starting Stage 2: Quality Filtering & Sharp Object Extraction <<<")
            s2_start_time = time.time()
            s2_output_csv_path = run_stage_02_quality_filter(
                input_particles_csv_path=s1_output_csv_path,
                image_dir_path=s1_input_image_dir, # S2 needs original images for final sharp crops
                output_dir_stage2=s2_output_dir,
                sample_id=args.sample_id
            )
            s2_duration = time.time() - s2_start_time
            if not s2_output_csv_path:
                logger.error("Stage 2 failed to produce an output CSV. Subsequent stages depending on it may fail.")
                if run_s3_predict: raise ProcessingError("Stage 2 failed, cannot proceed to Stage 3 prediction.")
            else:
                logger.info(f"<<< Stage 2 COMPLETED in {s2_duration:.2f} seconds. Output CSV: {s2_output_csv_path} >>>")
                # The input for S3 prediction is the directory of sharp object crops from S2
                s2_sharp_objects_dir = os.path.join(s2_output_dir, "sharp_object_final_crops")

        elif run_s3_predict: # If running S3_predict but not S2, need S2 output dir for sharp crops
            s2_output_csv_path = os.path.join(s2_output_dir, f"{args.sample_id}_stage2_analyzed_particles.csv") # General S2 output
            s2_sharp_objects_dir = os.path.join(s2_output_dir, "sharp_object_final_crops")
            if not os.path.isdir(s2_sharp_objects_dir):
                logger.error(f"Stage 2 output directory for sharp objects '{s2_sharp_objects_dir}' not found, "
                             "but required for Stage 3 prediction.")
                raise FileNotFoundError(f"Missing Stage 2 sharp objects directory: {s2_sharp_objects_dir}")
            logger.info(f"Using existing Stage 2 sharp objects from: {s2_sharp_objects_dir}")


        # --- Stage 3: ML Classification ---
        # Handle training mode
        if run_s3_train:
            logger.info(">>> Starting Stage 3: ML Model Training <<<")
            s3_train_start_time = time.time()
            # Stage 3 training doesn't depend on S1/S2 outputs for this sample,
            # it uses the general training_data specified in config.
            # The `input_dir_or_csv_path` argument is for prediction mode.
            model_or_features_path = run_stage_03_ml_classification(
                input_dir_or_csv_path="", # Not used for training-only mode
                output_dir_stage3=s3_output_dir, # Base for S3, though train saves to global models_dir
                sample_id=args.sample_id, # For logging context primarily
                mode="train",
                force_model_retrain=args.force_retrain_ml
            )
            s3_train_duration = time.time() - s3_train_start_time
            if model_or_features_path:
                 logger.info(f"<<< Stage 3 ML Training COMPLETED in {s3_train_duration:.2f} seconds. Model/artifacts related to: {model_or_features_path} >>>")
            else:
                 logger.error("Stage 3 ML Training FAILED.")
                 # If train_and_predict, this failure should halt prediction
                 if run_s3_predict: raise ProcessingError("ML Training failed, cannot proceed to prediction.")
        
        # Handle prediction mode
        if run_s3_predict:
            if not s2_sharp_objects_dir: # Should have been set if S2 ran or was skipped
                logger.error("Stage 2 sharp objects directory path is not available. Cannot run Stage 3 prediction.")
                raise ProcessingError("Missing S2 sharp objects dir for S3 prediction.")

            logger.info(">>> Starting Stage 3: ML Prediction for Ideal FEMs <<<")
            s3_predict_start_time = time.time()
            s3_mode_for_run = "predict"
            if run_s3_train: # If training was also selected for this run
                s3_mode_for_run = "train_and_predict" # Model should already be trained or loaded by this point if run_s3_train was True
            
            s3_output_features_csv = run_stage_03_ml_classification(
                input_dir_or_csv_path=s2_sharp_objects_dir,
                output_dir_stage3=s3_output_dir, # Base for S3 outputs for this sample
                sample_id=args.sample_id,
                mode=s3_mode_for_run, # Pass appropriate mode
                force_model_retrain=args.force_retrain_ml if s3_mode_for_run == "train_and_predict" else False # Retrain only if mode includes train
            )
            s3_predict_duration = time.time() - s3_predict_start_time
            if s3_output_features_csv:
                logger.info(f"<<< Stage 3 ML Prediction COMPLETED in {s3_predict_duration:.2f} seconds. Accepted FEMs CSV: {s3_output_features_csv} >>>")
            else:
                logger.error("Stage 3 ML Prediction FAILED or produced no accepted FEMs CSV.")
        
        pipeline_duration = time.time() - pipeline_start_time
        logger.info(f"--- Pipeline execution for Sample ID: {args.sample_id} FINISHED in {pipeline_duration:.2f} seconds ---")

    except FileNotFoundError as e:
        logger.error(f"A required file or directory was not found: {e}")
        logger.error("Pipeline run aborted.")
    except DataValidationError as e:
        logger.error(f"Data validation error: {e}")
        logger.error("Pipeline run aborted.")
    except ProcessingError as e:
        logger.error(f"A processing error occurred in one of the stages: {e}")
        logger.error("Pipeline run aborted.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during pipeline execution: {e}") # Includes stack trace
        logger.error("Pipeline run aborted due to unexpected error.")
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()