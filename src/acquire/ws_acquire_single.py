# Import module for image acquisition

'''
Script for recording images without controlling Zaber XY-table
'''

from wellscanner_acquisition_module import run_acquisition, load_config


plate_no = 0
well_no = 0 
start_dpt = -2.0

ACQUISITION_CONFIG_PATH = "ws_config.yaml" # default: ws_config.yaml
acq_conf = load_config(ACQUISITION_CONFIG_PATH)
run_acquisition(plate_no, well_no, start_dpt, acq_conf)
