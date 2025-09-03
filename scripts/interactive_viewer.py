# interactive_viewer.py (to be placed in well_plate_analyzer/scripts/)

import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skimage.io
import skimage.util
import skimage.color
import skimage.measure 
import numpy as np
import tempfile
import logging
import cv2 
from sklearn.neighbors import BallTree 
import shutil 
import re
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import trackpy as tp


try:
    from src.utils.config_loader import get_config, load_config, get_unified_segmentation_params 
    from src.core.ml_feature_extractor import extract_ml_features, _calculate_and_attach_posterior_attributes
    from src.visualization.feature_visualization import visualize_ml_features
    from src.core.detailed_morphology_calculator import reconstruct_main_body_masks, calculate_detailed_body_morphology 
    from src.visualization.stage_04_visualization import visualize_detailed_morphology 
    from src.utils.common_exceptions import ConfigError, DataValidationError 
    from src.stages.stage_01_initial_detection import (
        _parse_image_filenames, 
        _filter_image_df,
        _analyze_single_subtracted_image_global_thresh, 
        _analyze_unsubtracted_locally, 
        _compare_and_fuse_particle_dfs,
        _process_particle_coordinates_for_tracking,
        _track_particles_3d
    )
    from src.stages.stage_02_quality_filter import (
        _process_single_particle_for_quality,
        _filter_and_score_analysis_results,
    )
    from src.core.particle_operations import ParticleCropper, detect_cropped_particle_on_boundary
    from src.core.sharpness_assessment import AVAILABLE_SHARPNESS_METRICS
    from src.core.image_io_operations import load_image as core_load_image, convert_to_grayscale
    from src.core.image_enhancement import subtract_images, apply_blur
    from src.core.masking_operations import mask_well_area, apply_mask 
    from src.core.region_props_calculator import find_contours as core_find_contours, calculate_contour_measurements
    from src.core.segmentation import segment_main_body_from_image_float 
    from src.core.well_analysis import create_well_mask_black_plate, calculate_well_brightness_peak, match_well_brightness


except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
UPDATE_DEBOUNCE_MS = 300 
BG_OFFSET_MIN, BG_OFFSET_MAX = 1, 500 
DEFAULT_INT_MIN, DEFAULT_INT_MAX = 0, 10000
DEFAULT_FLOAT_MIN, DEFAULT_FLOAT_MAX = 0.0, 100.0
DEFAULT_ROI_SIZE_MIN, DEFAULT_ROI_SIZE_MAX = 30, 500

class FeatureViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Pipeline Viewer"); master.geometry("1600x950") 
        self.image_path = None; self.image_array_gray_float = None 
        self.default_params = {}; self.current_params = {} 
        self.param_vars = {} 
        self.param_str_vars = {} 
        self.param_specs = {} 
        self.param_widgets = {} 
        self.update_job_id = None; self.active_visualization_mode = tk.StringVar(value="S1")
        self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame()
        self.s1_selected_fg_filepath_var = tk.StringVar()
        self.s1_convolution_kernel_list = None; self.s1_raw_config = {} 
        self.s1_system_config_from_file = {} 
        self.s1_bg_offset_val_str = tk.StringVar()
        self.s1_particle_df_for_tracking = pd.DataFrame() # Holds loaded S1 CSV data
        self.s2_status_text_var = tk.StringVar(value="S2 Status: Load Dir & select FG frame.")
        
        self.s2_final_sharp_particles_df_for_viewer = pd.DataFrame() 
        self.s2_selected_sharp_particle_var = tk.StringVar()
        self.s2_sharp_particle_attributes_text = None 
        self.fig_s2_single_crop, self.ax_s2_single_crop = None, None
        self.canvas_s2_single_crop = None

        try:
            load_config(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml")); cfg = get_config()
            
            self.s1_system_config_from_file = cfg.get('system', {}) 
            global_source_illum_default = self.s1_system_config_from_file.get('source_image_illumination_type', 'darkfield')
            global_thresh_cfg_default = self.s1_system_config_from_file.get('thresholding', {})
            if not global_thresh_cfg_default:
                messagebox.showwarning("Config Warning", "Global thresholding config 'system.thresholding' missing.")
                global_thresh_cfg_default = {'mode': 'otsu', 'otsu_threshold_multiplier': 0.45, 'fixed_threshold_value': 70}

            s3_ml_cfg = cfg.get('stage_03_ml_classification', {}); 
            s3_feat_params_structured = s3_ml_cfg.get('feature_extraction_params', {})
            s3_main_body_seg_cfg = s3_feat_params_structured.get('main_body_segmentation', {})
            s3_egg_detect_cfg = s3_feat_params_structured.get('egg_detection', {})
            
            self.s1_raw_config = cfg.get('stage_01_initial_detection', {})
            s1_particle_cfg = self.s1_raw_config.get('particle_analysis', {}) 
            s1_unsub_particle_cfg = self.s1_raw_config.get('unsubtracted_particle_analysis', {})
            s1_comp_cfg = self.s1_raw_config.get('particle_comparison', {}) 
            s1_well_mask_cfg = self.s1_raw_config.get('well_masking', {}) 
            s1_black_plate_well_analysis_cfg = self.s1_raw_config.get('black_plate_well_analysis', {})
            s1_tracking_cfg = self.s1_raw_config.get('tracking', {})
            
            s2_cfg = cfg.get('stage_02_quality_filter', {})
            s2_intermediate_crop_cfg = s2_cfg.get('intermediate_crop', {})
            s2_black_mask_cfg = s2_cfg.get('black_mask_check', {})
            s2_final_sharp_filter_cfg = s2_cfg.get('final_sharp_object_filter', {})
            s2_filters_enabled_cfg = s2_cfg.get('filters_enabled', {})
            s2_final_crop_settings_cfg = s2_cfg.get('final_sharp_crop_settings', {})

            self.s1_convolution_kernel_list = s1_particle_cfg.get('convolution_kernel')
            
            self.default_params = {
                'viewer_source_illumination_type': global_source_illum_default,
                'viewer_thresholding_mode': global_thresh_cfg_default.get('mode', 'otsu'),
                'viewer_otsu_threshold_multiplier': global_thresh_cfg_default.get('otsu_threshold_multiplier', 0.45),
                'viewer_fixed_threshold_value': global_thresh_cfg_default.get('fixed_threshold_value', 70),
            }
            self.default_params.update({f"main_body_seg_{k}": v for k, v in s3_main_body_seg_cfg.items()})
            self.default_params.update({f"egg_detect_{k}": v for k, v in s3_egg_detect_cfg.items()})
            self.default_params.update({
                's1_bg_offset_val': self.s1_raw_config.get('bg_offset_val', 110),
                's1_sub_apply_convolution': s1_particle_cfg.get('apply_convolution', True),
                's1_sub_min_microorganism_area': s1_particle_cfg.get('min_microorganism_area', 200),
                's1_sub_microorganism_blur_kernel': s1_particle_cfg.get('microorganism_blur_kernel', 5),
                's1_well_mask_create_mask': s1_well_mask_cfg.get('create_mask', True), 
                's1_well_mask_auto_center_mask': s1_well_mask_cfg.get('auto_center_mask', True),
                's1_well_mask_radius_px': s1_well_mask_cfg.get('radius_px', 2000),
                's1_well_mask_radius_fraction': s1_well_mask_cfg.get('radius_fraction', 0.67),
                's1_unsub_min_particle_area': s1_unsub_particle_cfg.get('min_particle_area', 300),
                's1_unsub_local_otsu_roi_size': s1_unsub_particle_cfg.get('local_otsu_roi_size', 150),
                's1_comp_search_radius_factor': s1_comp_cfg.get('search_radius_factor', 2.0),
                's1_comp_default_particle_area': s1_comp_cfg.get('default_particle_area', 500),
            })
            self.default_params['s1_bpwa_enable_brightness_adjustment'] = s1_black_plate_well_analysis_cfg.get('enable_brightness_adjustment', True)
            s1_bpwa_well_id_cfg = s1_black_plate_well_analysis_cfg.get('well_identification', {})
            self.default_params.update({f"s1_bpwa_well_id_{k}": v for k,v in s1_bpwa_well_id_cfg.items()})
            s1_bpwa_peak_cfg = s1_black_plate_well_analysis_cfg.get('brightness_peak_calculation', {})
            self.default_params.update({f"s1_bpwa_peak_{k}": v for k,v in s1_bpwa_peak_cfg.items()})
            s1_bpwa_match_cfg = s1_black_plate_well_analysis_cfg.get('brightness_matching', {})
            self.default_params.update({f"s1_bpwa_match_{k}": v for k,v in s1_bpwa_match_cfg.items()})
            
            self.default_params.update({
                's2_intermediate_crop_mode': s2_intermediate_crop_cfg.get('mode', 'rotational_rectangle'),
                's2_intermediate_crop_square_size': s2_intermediate_crop_cfg.get('square_size', 128),
                's2_intermediate_crop_rotational_padding': s2_intermediate_crop_cfg.get('rotational_padding', 10),
                's2_black_mask_min_area_ratio': s2_black_mask_cfg.get('min_area_ratio', 0.05),
                's2_final_sharpness_metric_to_filter': 'combined_sharpness', 
                's2_final_sharpness_threshold_min': s2_final_sharp_filter_cfg.get('combined_sharpness_min', 0.3),
                's2_final_crop_settings_centroid_square_size': s2_final_crop_settings_cfg.get('centroid_square_size', 150),
                's2_final_crop_settings_rotational_padding': s2_final_crop_settings_cfg.get('rotational_padding', 10),
                's2_enable_black_mask_filter': s2_filters_enabled_cfg.get('black_mask_artifact', True),
                's2_enable_on_boundary_filter': s2_filters_enabled_cfg.get('on_crop_boundary', True),
                's2_enable_final_sharpness_filter': s2_filters_enabled_cfg.get('final_sharpness', True),
            })
            
            self.default_params.update({f"tracking_{k}": v for k,v in s1_tracking_cfg.items()})


            self.current_params = self.default_params.copy()
            self._initialize_param_vars_and_specs()

        except Exception as e: logger.exception("Config Error"); messagebox.showerror("Config Error", f"{e}"); master.destroy(); return
        self.temp_dir = tempfile.TemporaryDirectory(); master.protocol("WM_DELETE_WINDOW", self.on_closing); self.setup_gui(); self.on_mode_change()

    def _initialize_param_vars_and_specs(self):
        all_specs = {
            **self._get_global_param_specs(), 
            **self._get_s1_param_specs(), 
            **self._get_s1_black_plate_well_param_specs(),
            **self._get_tracking_param_specs(),
            **self._get_s2_param_specs(), 
            **self._get_main_body_seg_param_specs(), 
            **self._get_egg_detect_param_specs()     
        }
        for name, default_val in self.default_params.items():
            spec = all_specs.get(name)
            if spec: 
                self.param_specs[name] = spec; param_type = spec['type']
                if default_val is None: 
                    if param_type == 'int': default_val = 0
                    elif param_type == 'float': default_val = 0.0
                    elif param_type == 'bool': default_val = False
                    elif param_type == 'choice' and spec.get('options'): default_val = str(spec['options'][0])
                if param_type == 'int':
                    self.param_vars[name] = tk.IntVar(value=int(default_val))
                    self.param_str_vars[name] = tk.StringVar(value=str(int(default_val)))
                    self.param_vars[name].trace_add("write", lambda n,i,m,var_name=name: self._sync_num_to_str(var_name))
                elif param_type == 'float':
                    self.param_vars[name] = tk.DoubleVar(value=float(default_val))
                    res = spec.get('res', 0.1); fmt_str = str(res); num_decimals = len(fmt_str.split('.')[-1]) if '.' in fmt_str else 0; fmt = f".{max(1,num_decimals)}f"
                    self.param_str_vars[name] = tk.StringVar(value=f"{float(default_val):{fmt}}")
                    self.param_vars[name].trace_add("write", lambda n,i,m,var_name=name, f=fmt: self._sync_num_to_str(var_name, f))
                elif param_type == 'choice': self.param_vars[name] = tk.StringVar(value=str(default_val))
                elif param_type == 'bool': self.param_vars[name] = tk.BooleanVar(value=bool(default_val))

    def _sync_num_to_str(self, param_name, fmt=None):
        try:
            num_var, str_var = self.param_vars[param_name], self.param_str_vars[param_name]
            if isinstance(num_var, tk.IntVar): str_var.set(str(num_var.get()))
            elif isinstance(num_var, tk.DoubleVar): str_var.set(f"{num_var.get():{fmt if fmt else '.2f'}}")
        except (tk.TclError, KeyError): pass 

    def on_closing(self): 
        if self.update_job_id: self.master.after_cancel(self.update_job_id)
        for canvas_attr in ['canvas_raw', 'canvas_dynamic_vis', 'canvas_s2_single_crop', 'canvas_tracking_2d', 'canvas_tracking_3d']: 
            if hasattr(self, canvas_attr) and getattr(self, canvas_attr):
                try: getattr(self, canvas_attr).get_tk_widget().destroy()
                except tk.TclError: pass
        plt.close('all'); self.temp_dir.cleanup(); self.master.destroy()

    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        top_controls_frame = ttk.Frame(main_frame); top_controls_frame.pack(fill=tk.X, pady=5)
        self.load_button = ttk.Button(top_controls_frame, text="Load", command=self.dispatch_load_action); self.load_button.pack(side=tk.LEFT, padx=5)
        self.s1_fg_image_selector_label = ttk.Label(top_controls_frame, text="FG (S1/S2):")
        self.s1_fg_image_selector = ttk.Combobox(top_controls_frame, textvariable=self.s1_selected_fg_filepath_var, state="disabled", width=40) 
        self.s1_fg_image_selector.bind("<<ComboboxSelected>>", self.on_s1_fg_image_selected)
        self.s2_status_label = ttk.Label(top_controls_frame, textvariable=self.s2_status_text_var)
        self.status_label = ttk.Label(top_controls_frame, text="Load."); self.status_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X) 
        toggle_frame = ttk.Frame(top_controls_frame); toggle_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(toggle_frame, text="Show:").pack(side=tk.LEFT)
        for mode_disp, mode_val in [("S1", "S1"), ("Tracking", "Tracking"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4")]: 
            ttk.Radiobutton(toggle_frame, text=mode_disp, variable=self.active_visualization_mode, value=mode_val, command=self.on_mode_change).pack(side=tk.LEFT)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL); paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        controls_frame_outer = ttk.Labelframe(paned_window, text="Params", padding="10"); paned_window.add(controls_frame_outer, weight=1) 
        controls_frame_outer.grid_rowconfigure(0, weight=1); controls_frame_outer.grid_columnconfigure(0, weight=1) 
        canvas_params = tk.Canvas(controls_frame_outer); scrollbar_params = ttk.Scrollbar(controls_frame_outer, orient="vertical", command=canvas_params.yview)
        self.controls_frame_inner = ttk.Frame(canvas_params)
        self.controls_frame_inner.bind("<Configure>", lambda e: canvas_params.configure(scrollregion=canvas_params.bbox("all")))
        canvas_params.create_window((0,0), window=self.controls_frame_inner, anchor="nw"); canvas_params.configure(yscrollcommand=scrollbar_params.set)
        canvas_params.grid(row=0, column=0, sticky="nsew"); scrollbar_params.grid(row=0, column=1, sticky="ns")
        self.controls_frame_inner.bind('<Enter>', lambda e: self._bind_mousewheel(e, canvas_params)); self.controls_frame_inner.bind('<Leave>', lambda e: self._unbind_mousewheel(e, canvas_params))
        
        self.global_params_frame = ttk.Labelframe(self.controls_frame_inner, text="Global Viewer Settings", padding=5)
        self.create_global_param_controls(self.global_params_frame)
        
        self.s1_params_frame_main = ttk.Frame(self.controls_frame_inner) 
        self.s1_bpwa_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Black Plate Well & Brightness Adj.", padding=5)
        self.create_s1_black_plate_well_param_controls(self.s1_bpwa_params_frame) 
        
        self.s1_subtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Subtracted Analysis Params", padding=5)
        self.create_s1_subtracted_param_controls(self.s1_subtracted_params_frame)
        self.s1_wellmask_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 General Well Masking Params", padding=5)
        self.create_s1_wellmask_param_controls(self.s1_wellmask_params_frame)
        self.s1_unsubtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Unsubtracted FG Analysis Params (Local Thresh)", padding=5)
        self.create_s1_unsubtracted_param_controls(self.s1_unsubtracted_params_frame)
        self.s1_comparison_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Particle Comparison Params", padding=5)
        self.create_s1_comparison_param_controls(self.s1_comparison_params_frame)

        self.tracking_params_frame_main = ttk.Frame(self.controls_frame_inner)
        self.create_tracking_param_controls(self.tracking_params_frame_main)
        
        self.s2_params_frame_main = ttk.Frame(self.controls_frame_inner)
        self.create_s2_param_controls(self.s2_params_frame_main)
        
        self.s3_s4_params_main_frame = ttk.Frame(self.controls_frame_inner) 
        self.s3_s4_main_body_seg_params_frame = ttk.Labelframe(self.s3_s4_params_main_frame, text="Main Body Segmentation (S3/S4)", padding=5)
        self.create_main_body_seg_param_controls(self.s3_s4_main_body_seg_params_frame) 
        self.s3_egg_params_frame = ttk.Labelframe(self.s3_s4_params_main_frame, text="Egg Detection (S3 only)", padding=5)
        self.create_egg_detect_param_controls(self.s3_egg_params_frame) 

        param_buttons = ttk.Frame(self.controls_frame_inner); param_buttons.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=5)
        ttk.Button(param_buttons, text="Reset", command=self.reset_parameters).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(param_buttons, text="Update", command=self.process_and_display_safe).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        
        self.vis_notebook = ttk.Notebook(paned_window); paned_window.add(self.vis_notebook, weight=3)
        self.main_vis_tab = ttk.Frame(self.vis_notebook); self.vis_notebook.add(self.main_vis_tab, text=" Frame Overview ")
        self.fig_raw, self.ax_raw = plt.subplots(); self.ax_raw.set_title("Raw Input"); self.ax_raw.set_axis_off()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.main_vis_tab); self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_raw.draw()
        self.fig_dynamic_vis = plt.figure(); self.ax_dynamic_vis = None 
        self.canvas_dynamic_vis = FigureCanvasTkAgg(self.fig_dynamic_vis, master=self.main_vis_tab); self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_dynamic_vis.draw()
        
        self.tracking_vis_tab = ttk.Frame(self.vis_notebook)
        self.setup_tracking_tab(self.tracking_vis_tab)
        
        self.s2_single_crop_tab = ttk.Frame(self.vis_notebook) 
        self.setup_s2_single_crop_tab(self.s2_single_crop_tab)

    def setup_tracking_tab(self, parent_tab_frame):
        tracking_paned = ttk.PanedWindow(parent_tab_frame, orient=tk.VERTICAL); tracking_paned.pack(fill=tk.BOTH, expand=True)
        
        self.tracking_summary_text = scrolledtext.ScrolledText(tracking_paned, wrap=tk.WORD, height=6, state=tk.DISABLED)
        tracking_paned.add(self.tracking_summary_text, weight=1)
        
        tracking_plots_frame = ttk.Frame(tracking_paned); tracking_paned.add(tracking_plots_frame, weight=5)
        
        self.fig_tracking_2d, self.ax_tracking_2d = plt.subplots(); self.ax_tracking_2d.set_title("2D Tracks"); self.ax_tracking_2d.set_axis_off()
        self.canvas_tracking_2d = FigureCanvasTkAgg(self.fig_tracking_2d, master=tracking_plots_frame)
        self.canvas_tracking_2d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.fig_tracking_3d = plt.figure(); self.ax_tracking_3d = self.fig_tracking_3d.add_subplot(111, projection='3d')
        self.ax_tracking_3d.set_title("3D Tracks"); self.ax_tracking_3d.set_axis_off()
        self.canvas_tracking_3d = FigureCanvasTkAgg(self.fig_tracking_3d, master=tracking_plots_frame)
        self.canvas_tracking_3d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

    def setup_s2_single_crop_tab(self, parent_tab_frame):
        s2_crop_selector_frame = ttk.Frame(parent_tab_frame, padding=5); s2_crop_selector_frame.pack(fill=tk.X)
        ttk.Label(s2_crop_selector_frame, text="Select Sharp Particle:").pack(side=tk.LEFT, padx=(0,5))
        self.s2_sharp_particle_selector = ttk.Combobox(s2_crop_selector_frame, textvariable=self.s2_selected_sharp_particle_var, state="disabled", width=60)
        self.s2_sharp_particle_selector.pack(side=tk.LEFT, fill=tk.X, expand=True); self.s2_sharp_particle_selector.bind("<<ComboboxSelected>>", self.on_s2_sharp_particle_selected)
        s2_crop_display_paned = ttk.PanedWindow(parent_tab_frame, orient=tk.HORIZONTAL); s2_crop_display_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        s2_crop_image_frame = ttk.Frame(s2_crop_display_paned, padding=5); s2_crop_display_paned.add(s2_crop_image_frame, weight=2)
        self.fig_s2_single_crop, self.ax_s2_single_crop = plt.subplots(figsize=(5,5)); self.ax_s2_single_crop.set_title("S2 Final Crop"); self.ax_s2_single_crop.set_axis_off()
        self.canvas_s2_single_crop = FigureCanvasTkAgg(self.fig_s2_single_crop, master=s2_crop_image_frame); self.canvas_s2_single_crop.get_tk_widget().pack(fill=tk.BOTH, expand=True); self.canvas_s2_single_crop.draw()
        s2_crop_attrs_frame = ttk.Labelframe(s2_crop_display_paned, text="Particle Attributes", padding=10); s2_crop_display_paned.add(s2_crop_attrs_frame, weight=1)
        self.s2_sharp_particle_attributes_text = scrolledtext.ScrolledText(s2_crop_attrs_frame, wrap=tk.WORD, height=20, width=40, state=tk.DISABLED); self.s2_sharp_particle_attributes_text.pack(fill=tk.BOTH, expand=True)

    def on_mode_change(self):
        mode = self.active_visualization_mode.get()
        
        # Hide all parameter frames first
        self.global_params_frame.pack_forget()
        self.s1_params_frame_main.pack_forget()
        self.tracking_params_frame_main.pack_forget()
        self.s2_params_frame_main.pack_forget()
        self.s3_s4_params_main_frame.pack_forget()
        
        # Hide context-specific widgets
        self.s1_fg_image_selector_label.pack_forget()
        self.s1_fg_image_selector.pack_forget()
        self.s2_status_label.pack_forget()
        self.canvas_raw.get_tk_widget().pack_forget()
        self.canvas_dynamic_vis.get_tk_widget().pack_forget()
        
        for tab in [self.s2_single_crop_tab, self.tracking_vis_tab, self.main_vis_tab]:
            try: self.vis_notebook.index(tab); self.vis_notebook.forget(tab) 
            except tk.TclError: pass 
        
        if mode == "S1":
            self.vis_notebook.add(self.main_vis_tab, text=" Frame Overview ")
            self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.global_params_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
            self.s1_params_frame_main.pack(fill=tk.X, expand=True)
            self.s1_bpwa_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_subtracted_params_frame.pack(fill=tk.X, expand=True, pady=3); self.s1_wellmask_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_unsubtracted_params_frame.pack(fill=tk.X, expand=True, pady=3); self.s1_comparison_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.load_button.config(text="Load S1 Dir"); self.s1_fg_image_selector_label.pack(side=tk.LEFT, padx=(10,2), before=self.status_label)
            self.s1_fg_image_selector.pack(side=tk.LEFT, padx=2, before=self.status_label); self.s1_fg_image_selector.config(state="readonly" if self.s1_image_directory_path and not self.s1_image_file_list_df.empty else "disabled")
        elif mode == "Tracking":
            self.vis_notebook.add(self.tracking_vis_tab, text=" Tracking Visualization ")
            self.tracking_params_frame_main.pack(fill=tk.X, expand=True)
            self.load_button.config(text="Load S1 Particle CSV")
        elif mode == "S2":
            self.vis_notebook.add(self.main_vis_tab, text=" Frame Overview ")
            self.vis_notebook.add(self.s2_single_crop_tab, text=" S2 Crop Inspector ") 
            self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.global_params_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
            self.s1_params_frame_main.pack(fill=tk.X, expand=True) 
            self.s1_bpwa_params_frame.pack(fill=tk.X, expand=True, pady=3) 
            self.s1_subtracted_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_wellmask_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_unsubtracted_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_comparison_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s2_params_frame_main.pack(fill=tk.X, expand=True) 
            self.load_button.config(text="Load S2 Dir (uses S1 FG)"); self.s1_fg_image_selector_label.pack(side=tk.LEFT, padx=(10,2), before=self.status_label); self.s1_fg_image_selector.pack(side=tk.LEFT, padx=2, before=self.status_label)
            self.s1_fg_image_selector.config(state="readonly" if self.s1_image_directory_path and not self.s1_image_file_list_df.empty else "disabled")
            self.s2_status_label.pack(side=tk.LEFT, padx=(10,0), after=self.s1_fg_image_selector)
            self.s2_selected_sharp_particle_var.set(""); self.update_s2_single_crop_display() 
        elif mode in ["S3", "S4"]:
            self.vis_notebook.add(self.main_vis_tab, text=" Frame Overview ")
            self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.global_params_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
            self.s3_s4_params_main_frame.pack(fill=tk.X, expand=True)
            self.s3_s4_main_body_seg_params_frame.pack(fill=tk.X, expand=True, pady=3)
            if mode == "S3": self.s3_egg_params_frame.pack(fill=tk.X, expand=True, pady=3)
            else: self.s3_egg_params_frame.pack_forget()
            self.load_button.config(text="Load Img (S3/S4)")
        
        self.update_global_threshold_param_visibility(); 
        if mode == "S2": self._update_s2_crop_param_visibility()
        if mode != "S1" and mode !="S2": self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame(); self.s1_fg_image_selector['values'] = []; self.s1_selected_fg_filepath_var.set("")
        if mode != "Tracking": self.s1_particle_df_for_tracking = pd.DataFrame()
        self.update_raw_image_display(); self.schedule_update()
    
    def on_s1_fg_image_selected(self, event=None): self.update_raw_image_display(); self.schedule_update()
    def _bind_mousewheel(self, event, canvas): canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))
    def _unbind_mousewheel(self, event, canvas): canvas.unbind_all("<MouseWheel>")
    def _on_mousewheel(self, event, canvas): canvas.yview_scroll(1 if (event.num == 5 or event.delta < 0) else -1, "units")
    
    def schedule_update(self): 
        if self.update_job_id: self.master.after_cancel(self.update_job_id)
        self.update_job_id = self.master.after(UPDATE_DEBOUNCE_MS, self.process_and_display_safe)
        if self.active_visualization_mode.get() == "S2":
             self.s2_selected_sharp_particle_var.set("") 
             self.update_s2_single_crop_display()

    def update_global_threshold_param_visibility(self, *args):
        mode_var = self.param_vars.get('viewer_thresholding_mode')
        if not mode_var: return
        mode = mode_var.get()
        otsu_widget_name = 'viewer_otsu_threshold_multiplier'; fixed_widget_name = 'viewer_fixed_threshold_value'
        
        if otsu_widget_name in self.param_widgets and self.param_widgets[otsu_widget_name]:
            if mode == 'otsu': self.param_widgets[otsu_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[otsu_widget_name].pack_forget()
        
        if fixed_widget_name in self.param_widgets and self.param_widgets[fixed_widget_name]:
            if mode == 'fixed': self.param_widgets[fixed_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[fixed_widget_name].pack_forget()

    def _update_s2_crop_param_visibility(self, *args):
        mode_var = self.param_vars.get('s2_intermediate_crop_mode')
        if not mode_var: return
        mode = mode_var.get()
        
        widgets_to_check = {
            'rotational_rectangle': ['s2_intermediate_crop_rotational_padding', 's2_final_crop_settings_rotational_padding'],
            'centroid_square': ['s2_intermediate_crop_square_size', 's2_final_crop_settings_centroid_square_size']
        }

        for widget_name in widgets_to_check['rotational_rectangle']:
            if widget_name in self.param_widgets and self.param_widgets[widget_name]:
                if mode == 'rotational_rectangle': self.param_widgets[widget_name].pack(fill=tk.X, pady=2, padx=5)
                else: self.param_widgets[widget_name].pack_forget()
        
        for widget_name in widgets_to_check['centroid_square']:
            if widget_name in self.param_widgets and self.param_widgets[widget_name]:
                if mode == 'centroid_square': self.param_widgets[widget_name].pack(fill=tk.X, pady=2, padx=5)
                else: self.param_widgets[widget_name].pack_forget()
        
    def _adjust_param_value(self, param_name, delta):
        num_var, spec = self.param_vars.get(param_name), self.param_specs.get(param_name)
        if not num_var or not spec: return
        current_val, param_type = num_var.get(), spec['type']
        step = spec.get('step', 1 if param_type == 'int' else spec.get('res', 0.1))
        new_val = current_val + (delta * step)
        min_val, max_val = spec.get('min', DEFAULT_INT_MIN if param_type == 'int' else DEFAULT_FLOAT_MIN), spec.get('max', DEFAULT_INT_MAX if param_type == 'int' else DEFAULT_FLOAT_MAX)
        new_val = max(min_val, min(new_val, max_val))
        if param_type == 'int' and spec.get('step') == 2 and int(new_val) % 2 == 0: 
             new_val_candidate_down = int(new_val) - 1; new_val_candidate_up = int(new_val) + 1
             if new_val_candidate_down >= min_val : new_val = new_val_candidate_down
             elif new_val_candidate_up <= max_val: new_val = new_val_candidate_up
             new_val = max(min_val, min(new_val, max_val))
        if param_type == 'float': num_var.set(round(new_val, len(str(step).split('.')[-1]) if '.' in str(step) else 0))
        else: num_var.set(int(new_val))
        self.schedule_update()

    def _on_param_entry_commit(self, param_name, event=None):
        num_var, str_var, spec = self.param_vars.get(param_name), self.param_str_vars.get(param_name), self.param_specs.get(param_name)
        if not num_var or not str_var or not spec: return
        try:
            val_from_str = float(str_var.get()) if spec['type'] == 'float' else int(str_var.get())
            min_val, max_val = spec.get('min', DEFAULT_INT_MIN if spec['type'] == 'int' else DEFAULT_FLOAT_MIN), spec.get('max', DEFAULT_INT_MAX if spec['type'] == 'int' else DEFAULT_FLOAT_MAX)
            clamped_val = max(min_val, min(val_from_str, max_val))
            if spec['type'] == 'int' and spec.get('step') == 2 and int(clamped_val) % 2 == 0: 
                clamped_val_candidate_down = int(clamped_val) - 1; clamped_val_candidate_up = int(clamped_val) + 1
                if clamped_val_candidate_down >= min_val: clamped_val = clamped_val_candidate_down
                elif clamped_val_candidate_up <= max_val: clamped_val = clamped_val_candidate_up
                clamped_val = max(min_val, min(clamped_val, max_val))
            num_var.set(clamped_val); self.schedule_update()
        except ValueError: str_var.set(str(num_var.get())) 
        self.master.focus()

    def _create_generic_param_control(self, parent_frame, name, spec, current_val_for_var):
        param_type = spec['type']; container = ttk.Frame(parent_frame)
        label_text_raw = name.replace('_', ' ').title()
        label_text = re.sub(r"Main Body Seg\s*", "", label_text_raw, flags=re.IGNORECASE) 
        label_text = re.sub(r"Egg Detect\s*", "", label_text, flags=re.IGNORECASE) 
        label_text = re.sub(r"S1 Bpwa Well Id\s*", "", label_text, flags=re.IGNORECASE) 
        label_text = re.sub(r"S1 Bpwa Peak\s*", "", label_text, flags=re.IGNORECASE)   
        label_text = re.sub(r"S1 Bpwa Match\s*", "", label_text, flags=re.IGNORECASE) 
        label_text = re.sub(r"S1 Bpwa\s*", "", label_text, flags=re.IGNORECASE) 
        label_text = re.sub(r"S2\s*(Intermediate|Final|Enable|Black Mask|)\s*", "", label_text, flags=re.IGNORECASE)
        label_text = re.sub(r"S\d\s*(Sub|Unsub|Comp|Well Mask|Enable|Viewer|)\s*", "", label_text, flags=re.IGNORECASE)
        
        
        ttk.Label(container, text=label_text.strip() + ":", width=28, anchor="w").pack(side=tk.LEFT, padx=(0,5))
        tk_var = self.param_vars.get(name) 
        if param_type == 'choice':
            cmd = self.schedule_update 
            if name == 'viewer_thresholding_mode': cmd = self.update_global_threshold_param_visibility
            if name == 's2_intermediate_crop_mode': cmd = self._update_s2_crop_param_visibility
            ttk.OptionMenu(container, tk_var, current_val_for_var if current_val_for_var is not None else spec['options'][0], *spec['options'], command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X)
        elif param_type in ['int', 'float']:
            str_tk_var = self.param_str_vars.get(name)
            ttk.Button(container, text="-", command=lambda p_name=name: self._adjust_param_value(p_name, -1), width=2).pack(side=tk.LEFT)
            entry = ttk.Entry(container, textvariable=str_tk_var, width=7, justify='center')
            entry.pack(side=tk.LEFT, padx=2); entry.bind("<Return>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e)); entry.bind("<FocusOut>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e))
            ttk.Button(container, text="+", command=lambda p_name=name: self._adjust_param_value(p_name, 1), width=2).pack(side=tk.LEFT)
        elif param_type == 'bool': ttk.Checkbutton(container, variable=tk_var, command=self.schedule_update).pack(side=tk.LEFT)
        
        conditional_s2_widgets = [
            's2_intermediate_crop_square_size', 's2_intermediate_crop_rotational_padding',
            's2_final_crop_settings_centroid_square_size', 's2_final_crop_settings_rotational_padding'
        ]
        if not (name in ['viewer_otsu_threshold_multiplier', 'viewer_fixed_threshold_value'] or name in conditional_s2_widgets):
            container.pack(fill=tk.X, pady=2, padx=5)
        self.param_widgets[name] = container 
    
    def _get_global_param_specs(self):
        return {
            'viewer_source_illumination_type': {'options': ['darkfield', 'collimated'], 'type': 'choice'},
            'viewer_thresholding_mode': {'options': ['otsu', 'fixed'], 'type': 'choice'},
            'viewer_otsu_threshold_multiplier': {'min': 0.1, 'max': 2.0, 'type': 'float', 'res': 0.05},
            'viewer_fixed_threshold_value': {'min': 0, 'max': 255, 'type': 'int', 'step': 1},
        }

    def _get_s1_param_specs(self): 
        s1_sub_specs = {
            's1_bg_offset_val': {'min': BG_OFFSET_MIN, 'max': BG_OFFSET_MAX, 'type': 'int', 'step': 1},
            's1_sub_apply_convolution': {'type': 'bool'}, 
            's1_sub_min_microorganism_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10}, 
            's1_sub_microorganism_blur_kernel': {'min': 0, 'max': 15, 'type': 'int', 'step': 2}, 
        }
        s1_wm_specs = { 
            's1_well_mask_create_mask': {'type': 'bool'}, 
            's1_well_mask_auto_center_mask': {'type': 'bool'},
            's1_well_mask_radius_px': {'min': 100, 'max': 4000, 'type': 'int', 'step': 50}, 
            's1_well_mask_radius_fraction': {'min': 0.1, 'max': 1.0, 'type': 'float', 'res': 0.01},
        }
        s1_unsub_specs = { 
            's1_unsub_min_particle_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10},
            's1_unsub_local_otsu_roi_size': {'min': DEFAULT_ROI_SIZE_MIN, 'max': DEFAULT_ROI_SIZE_MAX, 'type': 'int', 'step': 10}, 
        }
        s1_comp_specs = {
            's1_comp_search_radius_factor': {'min': 0.1, 'max': 10.0, 'type': 'float', 'res': 0.1},
            's1_comp_default_particle_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10},
        }
        return {**s1_sub_specs, **s1_wm_specs, **s1_unsub_specs, **s1_comp_specs}

    def _get_s1_black_plate_well_param_specs(self): 
        return {
            's1_bpwa_enable_brightness_adjustment': {'type': 'bool'},
            's1_bpwa_well_id_low_threshold_value': {'min': 1, 'max': 50, 'type': 'int', 'step': 1},
            's1_bpwa_well_id_min_expected_diameter_px': {'min': 100, 'max': 5000, 'type': 'int', 'step': 100},
            's1_bpwa_well_id_max_expected_diameter_px': {'min': 200, 'max': 6000, 'type': 'int', 'step': 100},
            's1_bpwa_well_id_min_circularity': {'min': 0.5, 'max': 1.0, 'type': 'float', 'res': 0.01},
            's1_bpwa_peak_min_intensity_for_peak_search': {'min': 0, 'max': 200, 'type': 'int', 'step': 5},
            's1_bpwa_peak_percentile_for_peak_proxy': {'min': 50, 'max': 99, 'type': 'int', 'step': 1},
            's1_bpwa_match_max_peak_difference_allowed': {'min': 0, 'max': 50, 'type': 'int', 'step': 1},
            's1_bpwa_match_adjustment_method': {'options': ['global_shift', 'histogram_matching'], 'type': 'choice'},
            's1_bpwa_match_target_peak_offset_after_shift': {'min': -20, 'max': 20, 'type': 'int', 'step': 1},
        }
        
    def _get_tracking_param_specs(self):
        return {
            'tracking_enabled': {'type': 'bool'},
            'tracking_pixel_size_um': {'min': 0.1, 'max': 10.0, 'type': 'float', 'res': 0.1},
            'tracking_magnification': {'min': 0.1, 'max': 10.0, 'type': 'float', 'res': 0.001},
            'tracking_search_range_mm': {'min': 0.01, 'max': 5.0, 'type': 'float', 'res': 0.01},
            'tracking_memory_frames': {'min': 0, 'max': 20, 'type': 'int', 'step': 1},
            'tracking_min_track_length': {'min': 2, 'max': 100, 'type': 'int', 'step': 1},
        }

    def _get_s2_param_specs(self):
        return {
            's2_intermediate_crop_mode': {'options': ['rotational_rectangle', 'centroid_square', 'bounding_box'], 'type': 'choice'},
            's2_intermediate_crop_square_size': {'min': 32, 'max': 512, 'type': 'int', 'step': 4},
            's2_intermediate_crop_rotational_padding': {'min': 0, 'max': 50, 'type': 'int', 'step': 1},
            's2_black_mask_min_area_ratio': {'min': 0.01, 'max': 0.5, 'type': 'float', 'res': 0.01},
            's2_final_sharpness_metric_to_filter': {'options': AVAILABLE_SHARPNESS_METRICS + ['combined_sharpness'], 'type': 'choice'},
            's2_final_sharpness_threshold_min': {'min': 0.0, 'max': 5000.0, 'type': 'float', 'res': 0.01}, 
            's2_final_crop_settings_centroid_square_size': {'min': 32, 'max': 512, 'type': 'int', 'step': 4},
            's2_final_crop_settings_rotational_padding': {'min': 0, 'max': 50, 'type': 'int', 'step': 1},
            's2_enable_black_mask_filter': {'type': 'bool'},
            's2_enable_on_boundary_filter': {'type': 'bool'},
            's2_enable_final_sharpness_filter': {'type': 'bool'},
        }

    def _get_main_body_seg_param_specs(self): 
        return {
            'main_body_seg_enable_unsharp_masking': {'type': 'bool'},
            'main_body_seg_unsharp_mask_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float', 'res': 0.1},
            'main_body_seg_unsharp_mask_amount': {'min': 0.1, 'max': 5.0, 'type': 'float', 'res': 0.1},
            'main_body_seg_min_noise_size': {'min': 1, 'max': 1000, 'type': 'int', 'step': 10},
            'main_body_seg_closing_selem_size': {'min': 0, 'max': 10, 'type': 'int', 'step': 1},
            'main_body_seg_min_significant_area': {'min': 10, 'max': 2000, 'type': 'int', 'step': 10},
        }

    def _get_egg_detect_param_specs(self): 
        return {
            'egg_detect_gaussian_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float', 'res': 0.1}, 
            'egg_detect_egg_posterior_pixel_fraction': {'min': 0.05, 'max': 0.95, 'type': 'float', 'res': 0.01}, 
            'egg_detect_egg_threshold_quantile': {'min': 0.01, 'max': 0.99, 'type': 'float', 'res': 0.01},
            'egg_detect_egg_min_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10}, 
            'egg_detect_egg_max_area': {'min': 50, 'max': 20000, 'type': 'int', 'step': 100},
            'egg_detect_egg_min_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 
            'egg_detect_egg_max_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_detect_egg_min_eccentricity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 
            'egg_detect_orientation_diff_tolerance_deg': {'min': 0, 'max': 90, 'type': 'int', 'step': 1}
        }

    def create_global_param_controls(self, parent_frame):
        global_specs = self._get_global_param_specs()
        ordered_global_params = ['viewer_source_illumination_type', 'viewer_thresholding_mode', 
                                 'viewer_otsu_threshold_multiplier', 'viewer_fixed_threshold_value']
        for name in ordered_global_params:
            if name in self.default_params:
                self._create_generic_param_control(parent_frame, name, global_specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s1_black_plate_well_param_controls(self, parent_frame): 
        specs = self._get_s1_black_plate_well_param_specs()
        ordered_params = [
            's1_bpwa_enable_brightness_adjustment',
            's1_bpwa_well_id_low_threshold_value',
            's1_bpwa_well_id_min_expected_diameter_px',
            's1_bpwa_well_id_max_expected_diameter_px',
            's1_bpwa_well_id_min_circularity',
            's1_bpwa_peak_min_intensity_for_peak_search',
            's1_bpwa_peak_percentile_for_peak_proxy',
            's1_bpwa_match_max_peak_difference_allowed',
            's1_bpwa_match_adjustment_method',
            's1_bpwa_match_target_peak_offset_after_shift',
        ]
        for name in ordered_params:
            if name in self.default_params and name in specs: 
                self._create_generic_param_control(parent_frame, name, specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s1_subtracted_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = ['s1_bg_offset_val', 's1_sub_apply_convolution', 
                          's1_sub_min_microorganism_area', 's1_sub_microorganism_blur_kernel']
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_s1_wellmask_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = ['s1_well_mask_create_mask', 's1_well_mask_auto_center_mask','s1_well_mask_radius_px', 's1_well_mask_radius_fraction']
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s1_unsubtracted_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = ['s1_unsub_min_particle_area', 's1_unsub_local_otsu_roi_size'] 
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))
                
    def create_s1_comparison_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = ['s1_comp_search_radius_factor', 's1_comp_default_particle_area']
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_tracking_param_controls(self, parent_frame):
        ttk.Label(parent_frame, text="Tracking Parameters", font='Helvetica 10 bold').pack(fill=tk.X, pady=(5,2))
        tracking_specs = self._get_tracking_param_specs()
        ordered_tracking_params = [
            'tracking_enabled', 'tracking_pixel_size_um', 'tracking_magnification',
            'tracking_search_range_mm', 'tracking_memory_frames', 'tracking_min_track_length'
        ]
        for name in ordered_tracking_params:
            if name in self.default_params:
                self._create_generic_param_control(parent_frame, name, tracking_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_s2_param_controls(self, parent_frame):
        ttk.Label(parent_frame, text="S2 Quality Filter Params", font='Helvetica 10 bold').pack(fill=tk.X, pady=(5,2))
        s2_specs = self._get_s2_param_specs()
        
        filter_toggles_frame = ttk.Labelframe(parent_frame, text="Filter Toggles", padding=5)
        filter_toggles_frame.pack(fill=tk.X, expand=True, pady=3)
        ordered_s2_toggles = ['s2_enable_black_mask_filter', 's2_enable_on_boundary_filter', 's2_enable_final_sharpness_filter']
        for name in ordered_s2_toggles:
            if name in self.default_params:
                self._create_generic_param_control(filter_toggles_frame, name, s2_specs[name], self.current_params.get(name, self.default_params[name]))
        
        other_params_frame = ttk.Labelframe(parent_frame, text="Other S2 Parameters", padding=5)
        other_params_frame.pack(fill=tk.X, expand=True, pady=3)
        ordered_s2_params = [
            's2_intermediate_crop_mode', 
            's2_intermediate_crop_square_size', 's2_intermediate_crop_rotational_padding',
            's2_black_mask_min_area_ratio', 
            's2_final_sharpness_metric_to_filter', 's2_final_sharpness_threshold_min', 
            's2_final_crop_settings_centroid_square_size', 's2_final_crop_settings_rotational_padding'
        ]
        for name in ordered_s2_params:
            if name in self.default_params:
                self._create_generic_param_control(other_params_frame, name, s2_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_main_body_seg_param_controls(self, parent_frame): 
        specs = self._get_main_body_seg_param_specs()
        ordered_params = [
            'main_body_seg_enable_unsharp_masking', 'main_body_seg_unsharp_mask_sigma', 
            'main_body_seg_unsharp_mask_amount', 'main_body_seg_min_noise_size', 
            'main_body_seg_closing_selem_size', 'main_body_seg_min_significant_area',
        ]
        for name in ordered_params:
            if name in self.default_params and name in specs: 
                self._create_generic_param_control(parent_frame, name, specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_egg_detect_param_controls(self, parent_frame): 
        specs = self._get_egg_detect_param_specs()
        ordered_params = [ 
            'egg_detect_gaussian_sigma', 
            'egg_detect_egg_posterior_pixel_fraction', 'egg_detect_egg_threshold_quantile',
            'egg_detect_egg_min_area', 'egg_detect_egg_max_area', 'egg_detect_egg_min_solidity', 
            'egg_detect_egg_max_solidity', 'egg_detect_egg_min_eccentricity', 
            'egg_detect_orientation_diff_tolerance_deg'
        ]
        for name in ordered_params:
            if name in self.default_params and name in specs:
                self._create_generic_param_control(parent_frame, name, specs[name], self.current_params.get(name, self.default_params[name]))
    
    def reset_parameters(self): 
        if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
        for name, default_val in self.default_params.items():
            if name in self.param_vars: self.param_vars[name].set(default_val)
        self.on_mode_change() 
        self.update_global_threshold_param_visibility() 

    def dispatch_load_action(self):
        mode = self.active_visualization_mode.get()
        if mode == "Tracking": self._load_tracking_csv()
        elif mode in ["S1", "S2"]: self._load_s1_directory()
        else: self._load_s3_s4_image()

    def _load_tracking_csv(self):
        path = filedialog.askopenfilename(
            title="Select Stage 1 Particle CSV for Tracking",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not path:
            self.status_label.config(text="Tracking CSV selection cancelled.")
            return
        
        self.status_label.config(text=f"Loading CSV: {os.path.basename(path)}...")
        self.master.update_idletasks()
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("Loaded CSV file is empty.")
            required_cols = ['X', 'Y', 'source_image_filename']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {', '.join(required_cols)}")
            
            self.s1_particle_df_for_tracking = df
            self.status_label.config(text=f"Loaded {len(df)} particles from {os.path.basename(path)}")
            self.schedule_update()

        except Exception as e:
            messagebox.showerror("CSV Load Error", f"Failed to load or validate CSV:\n{e}")
            self.status_label.config(text="Failed to load CSV.")
            self.s1_particle_df_for_tracking = pd.DataFrame()

    def _load_s1_directory(self): 
        current_mode = self.active_visualization_mode.get()
        dir_path = filedialog.askdirectory(title=f"Select {current_mode} Image Directory")
        if not dir_path: self.status_label.config(text=f"{current_mode} dir selection cancelled."); self.master.after(10, self.on_mode_change); return
        self.s1_image_directory_path = dir_path
        self.status_label.config(text=f"Loading {current_mode} dir: {os.path.basename(dir_path)}..."); self.master.update_idletasks()
        try:
            parsed_df = _parse_image_filenames(dir_path);
            if parsed_df is None or parsed_df.empty: raise ValueError("No valid images found by parser.")
            self.s1_image_file_list_df = _filter_image_df(parsed_df); self.s1_image_file_list_df.reset_index(drop=True, inplace=True)
            if self.s1_image_file_list_df.empty: raise ValueError("No images after S1 filtering rules.")
            filenames = self.s1_image_file_list_df['filename'].tolist()
            self.s1_fg_image_selector['values'] = filenames
            if filenames: self.s1_selected_fg_filepath_var.set(filenames[0])
            self.status_label.config(text=f"{current_mode} Dir: {os.path.basename(dir_path)}. Select FG frame.")
        except Exception as e:
            messagebox.showerror(f"{current_mode} Load Error", f"{e}")
            self.status_label.config(text=f"Fail {current_mode} load."); self.s1_image_directory_path=None; self.s1_image_file_list_df=pd.DataFrame(); self.s1_fg_image_selector['values']=[]; self.s1_selected_fg_filepath_var.set("")
        finally: 
            self.master.after(10, self.on_mode_change) 
            if self.s1_image_directory_path and not self.s1_image_file_list_df.empty: self.schedule_update()

    def _load_s3_s4_image(self): 
        path = filedialog.askopenfilename(title="Select Image (S3/S4)", filetypes=(("Images", "*.png *.tif *.jpg"), ("All", "*.*")))
        if not path: self.status_label.config(text="S3/S4 selection cancelled."); self.master.after(10, self.on_mode_change); return
        self.image_path = path; self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame() 
        self.status_label.config(text=f"Loading {os.path.basename(path)}..."); self.master.update_idletasks()
        try:
            img = skimage.io.imread(path, as_gray=True)
            if img.ndim==3 and img.shape[2] in [3,4]: img = skimage.color.rgb2gray(img[:,:,:3])
            self.image_array_gray_float = skimage.util.img_as_float(img) 
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}.")
        except Exception as e:
            messagebox.showerror("Img Load Error", f"{e}")
            self.status_label.config(text="Fail img load."); self.image_path=None; self.image_array_gray_float=None
        finally: 
            self.master.after(10, self.on_mode_change)
            if self.image_path: self.schedule_update()

    def update_raw_image_display(self): 
        self.ax_raw.clear(); mode = self.active_visualization_mode.get(); title = "Raw Input"; img_show = None
        if mode in ["S1", "S2"] and self.s1_image_directory_path and not self.s1_image_file_list_df.empty:
            fn = self.s1_selected_fg_filepath_var.get()
            if fn:
                path = os.path.join(self.s1_image_directory_path, fn)
                if os.path.exists(path): img_show_raw_uint8 = core_load_image(path, 'gray'); img_show = skimage.util.img_as_float(img_show_raw_uint8); title = f"Sample Frame: {fn}"
                else: title = f"{mode}: FG Not found"
            else: title = f"{mode}: Select Dir"
        elif self.image_array_gray_float is not None and mode in ["S3","S4"]: img_show = self.image_array_gray_float; title = f"Input: {os.path.basename(self.image_path or 'No Img')}"
        if img_show is not None: self.ax_raw.imshow(img_show, cmap='gray', vmin=0.0, vmax=1.0)
        else: self.ax_raw.imshow(np.zeros((100,100)), cmap='gray') 
        self.ax_raw.set_title(title, fontsize=8); self.ax_raw.set_axis_off(); self.canvas_raw.draw()

    def collect_current_params(self):
        gui_params = {name: var.get() for name, var in self.param_vars.items()}
        
        viewer_global_illum = gui_params.get('viewer_source_illumination_type')
        viewer_global_thresh_cfg = {
            'mode': gui_params.get('viewer_thresholding_mode'),
            'otsu_threshold_multiplier': gui_params.get('viewer_otsu_threshold_multiplier'),
            'fixed_threshold_value': gui_params.get('viewer_fixed_threshold_value'),
        }

        self.current_params['s1_subtracted_effective'] = {k.replace('s1_sub_', ''):v for k,v in gui_params.items() if k.startswith('s1_sub_')}
        self.current_params['s1_subtracted_effective']['convolution_kernel'] = self.s1_convolution_kernel_list
        self.current_params['s1_bg_offset_effective'] = gui_params.get('s1_bg_offset_val')
        self.current_params['s1_well_mask_effective'] = {k.replace('s1_well_mask_', ''):v for k,v in gui_params.items() if k.startswith('s1_well_mask_')}
        self.current_params['s1_unsubtracted_effective'] = {'min_particle_area': gui_params.get('s1_unsub_min_particle_area'), 'local_otsu_roi_size': gui_params.get('s1_unsub_local_otsu_roi_size')}
        self.current_params['s1_comparison_effective'] = {k.replace('s1_comp_', ''):v for k,v in gui_params.items() if k.startswith('s1_comp_')}
        
        s1_bpwa_collected_cfg = {'enable_brightness_adjustment': gui_params.get('s1_bpwa_enable_brightness_adjustment')}
        s1_bpwa_well_id_collected = {}; s1_bpwa_peak_collected = {}; s1_bpwa_match_collected = {}
        for k_gui in gui_params:
            if k_gui.startswith('s1_bpwa_well_id_'): s1_bpwa_well_id_collected[k_gui.replace('s1_bpwa_well_id_', '')] = gui_params[k_gui]
            elif k_gui.startswith('s1_bpwa_peak_'): s1_bpwa_peak_collected[k_gui.replace('s1_bpwa_peak_', '')] = gui_params[k_gui]
            elif k_gui.startswith('s1_bpwa_match_'): s1_bpwa_match_collected[k_gui.replace('s1_bpwa_match_', '')] = gui_params[k_gui]
        s1_bpwa_collected_cfg['well_identification'] = s1_bpwa_well_id_collected
        s1_bpwa_collected_cfg['brightness_peak_calculation'] = s1_bpwa_peak_collected
        s1_bpwa_collected_cfg['brightness_matching'] = s1_bpwa_match_collected
        self.current_params['s1_black_plate_well_analysis_effective'] = s1_bpwa_collected_cfg
        
        self.current_params['tracking_effective'] = {k.replace('tracking_', ''):v for k,v in gui_params.items() if k.startswith('tracking_')}

        self.current_params['s2_effective'] = {
            'intermediate_crop': {
                'mode': gui_params.get('s2_intermediate_crop_mode'), 
                'square_size': gui_params.get('s2_intermediate_crop_square_size'), 
                'rotational_padding': gui_params.get('s2_intermediate_crop_rotational_padding'),
                'bbox_column_mapping': get_config().get('stage_02_quality_filter', {}).get('intermediate_crop', {}).get('bbox_column_mapping', {})
            },
            'black_mask_check': {'min_area_ratio': gui_params.get('s2_black_mask_min_area_ratio')},
            'filters_enabled': {
                'black_mask_artifact': gui_params.get('s2_enable_black_mask_filter'), 
                'on_crop_boundary': gui_params.get('s2_enable_on_boundary_filter'), 
                'final_sharpness': gui_params.get('s2_enable_final_sharpness_filter')
            },
            'sharpness_metrics': AVAILABLE_SHARPNESS_METRICS, 
            'final_sharp_object_filter': { f"{gui_params.get('s2_final_sharpness_metric_to_filter')}_min": gui_params.get('s2_final_sharpness_threshold_min')},
            'final_sharp_crop_settings': {
                'centroid_square_size': gui_params.get('s2_final_crop_settings_centroid_square_size'),
                'rotational_padding': gui_params.get('s2_final_crop_settings_rotational_padding')
            }
        }
        
        s3_s4_params = {
            'source_image_illumination_type': viewer_global_illum,
            **viewer_global_thresh_cfg,
        }
        for k_gui, spec_main in self._get_main_body_seg_param_specs().items():
            s3_s4_params[k_gui.replace('main_body_seg_', '')] = gui_params.get(k_gui)
        for k_gui, spec_egg in self._get_egg_detect_param_specs().items():
            s3_s4_params[k_gui.replace('egg_detect_', '')] = gui_params.get(k_gui)
            
        self.current_params['s3_s4_segmentation_and_feature_params'] = s3_s4_params
            
        return self.current_params

    def process_and_display_safe(self): 
        active_mode, valid_input = self.active_visualization_mode.get(), False
        if active_mode in ["S1", "S2"]:
            if self.s1_image_directory_path and not self.s1_image_file_list_df.empty and self.s1_selected_fg_filepath_var.get(): valid_input = True
            else: self.status_label.config(text=f"{active_mode}: Load dir & select FG frame.")
        elif active_mode == "Tracking":
            if not self.s1_particle_df_for_tracking.empty: valid_input = True
            else: self.status_label.config(text="Tracking: Load S1 particle CSV.")
        elif self.image_array_gray_float is not None: valid_input = True 
        else: self.status_label.config(text="S3/S4: Load image.")

        if not valid_input:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
            if self.fig_dynamic_vis: 
                self.fig_dynamic_vis.clear()
                self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111)
                self.ax_dynamic_vis.set_axis_off()
                self.canvas_dynamic_vis.draw()
            if active_mode == "S2": 
                self.s2_status_text_var.set("S2 Status: Load Dir & select FG frame.")
                self.s2_final_sharp_particles_df_for_viewer = pd.DataFrame() 
                self.s2_sharp_particle_selector['values'] = []
                self.s2_selected_sharp_particle_var.set("")
                self.update_s2_single_crop_display()
            return
        self.status_label.config(text="Processing..."); self.master.update_idletasks()
        if active_mode == "S2": self.s2_status_text_var.set("S2 Status: Processing...")
        if active_mode == "Tracking": self.tracking_summary_text.config(state=tk.NORMAL); self.tracking_summary_text.delete(1.0, tk.END); self.tracking_summary_text.insert(tk.END, "Processing CSV for tracking..."); self.tracking_summary_text.config(state=tk.DISABLED); self.master.update_idletasks()
        try: self.process_and_display(); self.status_label.config(text=f"Ready. Mode: {active_mode}.")
        except DataValidationError as dve: logger.error(f"Data Validation Error: {dve}"); messagebox.showerror("Data Error", f"{dve}"); self.status_label.config(text=f"Data Error: {dve}")
        except Exception as e: logger.exception("Proc Error"); messagebox.showerror("Error", f"{e}"); self.status_label.config(text=f"Error: {e}")
        finally:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None

    def process_and_display(self): 
        params = self.collect_current_params() 
        mode = self.active_visualization_mode.get()
        
        if isinstance(self.ax_dynamic_vis, np.ndarray): 
            for ax_item in self.ax_dynamic_vis.flat: ax_item.clear(); ax_item.set_axis_off()
        elif self.ax_dynamic_vis is not None: self.ax_dynamic_vis.clear(); self.ax_dynamic_vis.set_axis_off()
        self.fig_dynamic_vis.clear() 
        
        viewer_global_source_illum = self.param_vars['viewer_source_illumination_type'].get()
        viewer_global_thresh_cfg = {
            'mode': self.param_vars['viewer_thresholding_mode'].get(),
            'otsu_threshold_multiplier': self.param_vars['viewer_otsu_threshold_multiplier'].get(),
            'fixed_threshold_value': self.param_vars['viewer_fixed_threshold_value'].get(),
        }

        if mode == "S1" or mode == "S2": 
            s1_sub_eff = params['s1_subtracted_effective']; s1_wm_eff = params['s1_well_mask_effective'] 
            s1_unsub_eff = params['s1_unsubtracted_effective']; s1_comp_eff = params['s1_comparison_effective']
            s1_bg_off = params['s1_bg_offset_effective']
            s1_bpwa_eff_cfg = params['s1_black_plate_well_analysis_effective'] 
            
            fg_fn = self.s1_selected_fg_filepath_var.get()
            bg_filename_s1 = "N/A" 
            if not self.s1_image_file_list_df.empty and fg_fn:
                fg_row_series_list = self.s1_image_file_list_df[self.s1_image_file_list_df['filename'] == fg_fn]
                if fg_row_series_list.empty: raise ValueError(f"S1/S2 Viewer: Selected FG filename '{fg_fn}' not found.")
                fg_row = fg_row_series_list.iloc[0]
                fg_row_positional_index = self.s1_image_file_list_df.index.get_loc(fg_row.name)
                
                bg_df_idx = (fg_row_positional_index + s1_bg_off) % len(self.s1_image_file_list_df)
                bg_row = self.s1_image_file_list_df.iloc[bg_df_idx]
                bg_filename_s1 = bg_row['filename']
            else: raise ValueError("S1/S2 Viewer: Image list or FG selection is empty.")

            title_prefix = "S1 Output" if mode == "S1" else "S2 Quality Filter (Sim. S1)"
            self.fig_dynamic_vis.suptitle(f"{title_prefix} - FG: {fg_fn}\nBG: {bg_filename_s1}", fontsize=9, y=0.99)

            img_fg_orig_uint8 = core_load_image(fg_row['full_path'],'gray')
            img_bg_orig_uint8 = core_load_image(bg_row['full_path'],'gray')
            if img_fg_orig_uint8 is None or img_bg_orig_uint8 is None: raise ValueError("S1/S2 Viewer: FG/BG load fail.")
            
            img_a_for_sub_prep = img_fg_orig_uint8.copy()
            img_b_for_sub_prep = img_bg_orig_uint8.copy()
            if s1_bpwa_eff_cfg.get('enable_brightness_adjustment', False):
                logger.debug(f"Viewer: Applying brightness matching for FG: {fg_fn}, BG: {bg_filename_s1}")
                img_a_for_sub_prep, img_b_for_sub_prep = match_well_brightness(
                    img_fg_orig_uint8, img_bg_orig_uint8, s1_bpwa_eff_cfg
                )
            
            general_well_mask_for_roi_uint8 = None
            if s1_wm_eff.get('create_mask'): 
                _, general_well_mask_for_roi_uint8 = mask_well_area(img_fg_orig_uint8.copy(), s1_wm_eff, self.s1_system_config_from_file)

            img_a_inv_ready = img_a_for_sub_prep.copy() 
            img_b_inv_ready = img_b_for_sub_prep.copy() 
            if viewer_global_source_illum == 'collimated': 
                 img_a_inv_ready = cv2.bitwise_not(img_a_inv_ready)
                 img_b_inv_ready = cv2.bitwise_not(img_b_inv_ready)
            subtracted_img_uint8 = subtract_images(img_a_inv_ready, img_b_inv_ready)
            if subtracted_img_uint8 is None: raise ValueError("S1/S2 Viewer: Subtraction failed.")
            
            img_for_sub_analysis_uint8 = apply_mask(subtracted_img_uint8, general_well_mask_for_roi_uint8) \
                                         if general_well_mask_for_roi_uint8 is not None else subtracted_img_uint8
            
            if mode == "S1":
                axes_s1 = self.fig_dynamic_vis.subplots(2, 2) 
                self.ax_dynamic_vis = axes_s1

                df_sub_list_dict, binary_mask_sub_disp = _analyze_single_subtracted_image_global_thresh(img_for_sub_analysis_uint8, s1_sub_eff, viewer_global_thresh_cfg)
                df_subtracted_particles_viewer = pd.DataFrame(df_sub_list_dict) 
                if not df_subtracted_particles_viewer.empty: df_subtracted_particles_viewer['SAMPLE_ID'] = "viewer_s1"; df_subtracted_particles_viewer['source_image_filename'] = fg_fn
                
                img_for_unsub_prep = img_fg_orig_uint8.copy() 
                if general_well_mask_for_roi_uint8 is not None: 
                    if viewer_global_source_illum == 'collimated':
                        pixels_in_well = img_fg_orig_uint8[general_well_mask_for_roi_uint8 == 255]; 
                        if pixels_in_well.size > 0: hist = cv2.calcHist([img_fg_orig_uint8], [0], general_well_mask_for_roi_uint8, [256], [0,256]); dom_bg = np.argmax(hist); filled_out = np.full_like(img_fg_orig_uint8, dom_bg); filled_out[general_well_mask_for_roi_uint8 == 255] = img_fg_orig_uint8[general_well_mask_for_roi_uint8 == 255]; img_for_unsub_prep = filled_out
                        else: img_for_unsub_prep = apply_mask(img_fg_orig_uint8.copy(), general_well_mask_for_roi_uint8) if general_well_mask_for_roi_uint8 is not None else img_fg_orig_uint8.copy()
                    else: img_for_unsub_prep = apply_mask(img_fg_orig_uint8.copy(), general_well_mask_for_roi_uint8) if general_well_mask_for_roi_uint8 is not None else img_fg_orig_uint8.copy()
                
                df_unsub_list_dict, composite_binary_mask_unsub_disp, (example_roi_img_disp, example_roi_hist_ravel_disp, example_roi_otsu_disp) = _analyze_unsubtracted_locally(img_for_unsub_prep, df_sub_list_dict, s1_unsub_eff['local_otsu_roi_size'], s1_unsub_eff, viewer_global_thresh_cfg, viewer_global_source_illum)
                
                cfg_well_id_vis = s1_bpwa_eff_cfg.get('well_identification', {})
                low_thresh_vis = cfg_well_id_vis.get('low_threshold_value', 5)
                
                largest_contour_fg = None
                _, thresh_img_fg = cv2.threshold(img_fg_orig_uint8, low_thresh_vis, 255, cv2.THRESH_BINARY)
                contours_fg = core_find_contours(thresh_img_fg, mode="external", method="simple")
                if contours_fg: largest_contour_fg = max(contours_fg, key=cv2.contourArea)

                largest_contour_bg = None
                _, thresh_img_bg = cv2.threshold(img_bg_orig_uint8, low_thresh_vis, 255, cv2.THRESH_BINARY)
                contours_bg = core_find_contours(thresh_img_bg, mode="external", method="simple")
                if contours_bg: largest_contour_bg = max(contours_bg, key=cv2.contourArea)

                ax_sub = axes_s1[0,0]
                ax_sub.imshow(binary_mask_sub_disp if binary_mask_sub_disp is not None else np.zeros_like(img_fg_orig_uint8), cmap='gray')
                ax_sub.set_title(f"Subtracted Binary ({len(df_sub_list_dict)} p.)", fontsize=8)
                ax_sub.set_axis_off()

                if largest_contour_fg is not None:
                    h_im, w_im = img_fg_orig_uint8.shape; xc, yc = np.meshgrid(np.arange(w_im), np.arange(h_im))
                    fg_well_mask = np.zeros_like(img_fg_orig_uint8, dtype=np.uint8)
                    cv2.drawContours(fg_well_mask, [largest_contour_fg], -1, 1, thickness=cv2.FILLED)
                    ax_sub.contourf(xc, yc, fg_well_mask, levels=[0.5, 1.5], colors='blue', hatches=['//'], alpha=0.4)
                    legend_elements_sub = [Patch(facecolor='blue', edgecolor='blue', hatch='//', alpha=0.4, label='FG Well Area')]
                    ax_sub.legend(handles=legend_elements_sub, loc='upper right', fontsize='xx-small')

                ax_unsub = axes_s1[0,1]
                ax_unsub.imshow(composite_binary_mask_unsub_disp if composite_binary_mask_unsub_disp is not None else np.zeros_like(img_fg_orig_uint8), cmap='gray')
                ax_unsub.set_title(f"Unsub Local Binary Comp. ({len(df_unsub_list_dict)} p.)", fontsize=8)
                ax_unsub.set_axis_off()

                if largest_contour_bg is not None:
                    h_im, w_im = img_bg_orig_uint8.shape; xc, yc = np.meshgrid(np.arange(w_im), np.arange(h_im))
                    bg_well_mask = np.zeros_like(img_bg_orig_uint8, dtype=np.uint8)
                    cv2.drawContours(bg_well_mask, [largest_contour_bg], -1, 1, thickness=cv2.FILLED)
                    ax_unsub.contourf(xc, yc, bg_well_mask, levels=[0.5, 1.5], colors='yellow', hatches=['\\\\'], alpha=0.5)
                    legend_elements_unsub = [Patch(facecolor='yellow', edgecolor='yellow', hatch='\\\\', alpha=0.5, label='BG Well Area')]
                    ax_unsub.legend(handles=legend_elements_unsub, loc='upper right', fontsize='xx-small')
                
                if example_roi_img_disp is not None:
                    axes_s1[1,0].imshow(example_roi_img_disp, cmap='gray'); axes_s1[1,0].set_title("Example Local ROI (for hist)", fontsize=8); axes_s1[1,0].set_axis_off()
                    if example_roi_hist_ravel_disp is not None and example_roi_hist_ravel_disp.size > 0:
                        axes_s1[1,1].hist(example_roi_hist_ravel_disp, bins=256, range=(0.0,1.0), fc='b', ec='b', alpha=0.7); axes_s1[1,1].set_title("Hist of Example ROI", fontsize=8)
                        if example_roi_otsu_disp is not None:
                            axes_s1[1,1].axvline(example_roi_otsu_disp, color='r', ls='--', lw=1)
                            otsu_mult_viewer = viewer_global_thresh_cfg.get('otsu_threshold_multiplier', 0.45); final_local_calc_thresh = example_roi_otsu_disp * otsu_mult_viewer
                            axes_s1[1,1].axvline(final_local_calc_thresh, color='m', linestyle=':', linewidth=1)
                        axes_s1[1,1].tick_params(axis='both', which='major', labelsize=7); axes_s1[1,1].set_xlim(0,1)
                    else: axes_s1[1,1].text(0.5,0.5,"No Hist Data", ha='center', va='center'); axes_s1[1,1].set_axis_off()
                else: axes_s1[1,0].text(0.5,0.5,"No ROI", ha='center', va='center'); axes_s1[1,0].set_axis_off(); axes_s1[1,1].text(0.5,0.5,"No Hist", ha='center', va='center'); axes_s1[1,1].set_axis_off()
            
            elif mode == "S2":
                self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111)
                df_sub_list_dict_s2, _ = _analyze_single_subtracted_image_global_thresh(img_for_sub_analysis_uint8, s1_sub_eff, viewer_global_thresh_cfg) 
                df_sub_s2 = pd.DataFrame(df_sub_list_dict_s2)
                if not df_sub_s2.empty: df_sub_s2['SAMPLE_ID'] = "viewer_s2"; df_sub_s2['source_image_filename'] = fg_fn
                
                img_for_unsub_prep_s2 = img_fg_orig_uint8.copy() 
                if general_well_mask_for_roi_uint8 is not None:
                    if viewer_global_source_illum == 'collimated':
                        pixels_in_well = img_fg_orig_uint8[general_well_mask_for_roi_uint8 == 255]; 
                        if pixels_in_well.size > 0: hist = cv2.calcHist([img_fg_orig_uint8], [0], general_well_mask_for_roi_uint8, [256], [0,256]); dom_bg = np.argmax(hist); filled_out = np.full_like(img_fg_orig_uint8, dom_bg); filled_out[general_well_mask_for_roi_uint8 == 255] = img_fg_orig_uint8[general_well_mask_for_roi_uint8 == 255]; img_for_unsub_prep_s2 = filled_out
                        else: img_for_unsub_prep_s2 = apply_mask(img_fg_orig_uint8.copy(), general_well_mask_for_roi_uint8) if general_well_mask_for_roi_uint8 is not None else img_fg_orig_uint8.copy()
                    else: img_for_unsub_prep_s2 = apply_mask(img_fg_orig_uint8.copy(), general_well_mask_for_roi_uint8) if general_well_mask_for_roi_uint8 is not None else img_fg_orig_uint8.copy()

                df_unsub_s2_list_dict, _, _ = _analyze_unsubtracted_locally(
                    img_for_unsub_prep_s2, 
                    df_sub_list_dict_s2,
                    s1_unsub_eff['local_otsu_roi_size'], 
                    s1_unsub_eff, 
                    viewer_global_thresh_cfg, 
                    viewer_global_source_illum
                )
                df_unsub_s2 = pd.DataFrame(df_unsub_s2_list_dict) 
                if not df_unsub_s2.empty: df_unsub_s2['SAMPLE_ID'] = "viewer_s2"; df_unsub_s2['source_image_filename'] = fg_fn
                
                s1_particles_for_frame_df = _compare_and_fuse_particle_dfs(df_sub_s2, df_unsub_s2, s1_comp_eff)
                if not s1_particles_for_frame_df.empty: s1_particles_for_frame_df = s1_particles_for_frame_df.reset_index().rename(columns={'index': 's1_viewer_fused_idx'})
                self.s2_status_text_var.set(f"S2: S1 found {len(s1_particles_for_frame_df)} p. Analyzing quality...")
                if s1_particles_for_frame_df.empty:
                    self.ax_dynamic_vis.imshow(cv2.cvtColor(img_fg_orig_uint8, cv2.COLOR_GRAY2BGR)); self.ax_dynamic_vis.set_axis_off(); self.s2_final_sharp_particles_df_for_viewer = pd.DataFrame() 
                else:
                    s2_eff_config = params['s2_effective']
                    s2_intermediate_cropper = ParticleCropper(s2_eff_config['intermediate_crop']); s2_temp_dir = os.path.join(self.temp_dir.name, f"s2_viewer_crops_{fg_fn.replace('.png','').replace('.','_')}"); os.makedirs(s2_temp_dir, exist_ok=True)
                    s2_single_particle_results_list = []
                    for _, s1_row in s1_particles_for_frame_df.iterrows(): 
                        s2_res_obj = _process_single_particle_for_quality(s1_row, core_load_image(fg_row['full_path'], 'color'), img_fg_orig_uint8, s2_intermediate_cropper, s2_temp_dir, s2_eff_config, viewer_global_source_illum, viewer_global_thresh_cfg )
                        s2_res_obj.input_particle_data['s1_viewer_fused_idx'] = s1_row['s1_viewer_fused_idx']; s2_single_particle_results_list.append(s2_res_obj.to_dict())
                    s2_detailed_df = pd.DataFrame(s2_single_particle_results_list)
                    s2_merged_for_score = pd.merge(s1_particles_for_frame_df, s2_detailed_df, left_on='s1_viewer_fused_idx', right_on='input_particle_index', how='left', suffixes=('', '_s2analysis'))
                    s2_suitable_scored, _ = _filter_and_score_analysis_results(s2_merged_for_score, s2_eff_config)
                    status_parts_s2 = [f"S1_Fused: {len(s1_particles_for_frame_df)}"]; artifact_status = []
                    if s2_eff_config['filters_enabled']['black_mask_artifact']: artifact_status.append("BMaskOn") 
                    else: artifact_status.append("BMaskOff")
                    if s2_eff_config['filters_enabled']['on_crop_boundary']: artifact_status.append("BoundOn") 
                    else: artifact_status.append("BoundOff")
                    status_parts_s2.append(f"ArtF({','.join(artifact_status)}): {len(s2_suitable_scored)}")
                    self.s2_final_sharp_particles_df_for_viewer = s2_suitable_scored.copy() 
                    if not s2_suitable_scored.empty and s2_eff_config['filters_enabled']['final_sharpness']:
                        sharp_filt_cfg = s2_eff_config['final_sharp_object_filter']
                        if sharp_filt_cfg:
                            metric_key, thresh_val = list(sharp_filt_cfg.items())[0]; metric_actual = metric_key.replace("_min","")
                            if metric_actual in self.s2_final_sharp_particles_df_for_viewer.columns:
                                pass_mask = pd.to_numeric(self.s2_final_sharp_particles_df_for_viewer[metric_actual], errors='coerce').fillna(float('-inf')) > thresh_val; self.s2_final_sharp_particles_df_for_viewer = self.s2_final_sharp_particles_df_for_viewer[pass_mask]
                                status_parts_s2.append(f"SharpFOn({metric_actual}>{thresh_val:.2f}): {len(self.s2_final_sharp_particles_df_for_viewer)}")
                    elif s2_eff_config['filters_enabled']['final_sharpness']: status_parts_s2.append(f"SharpFOn: 0")
                    else: status_parts_s2.append(f"SharpFOff: {len(self.s2_final_sharp_particles_df_for_viewer)}")
                    self.s2_status_text_var.set(" | ".join(status_parts_s2)); self.update_s2_sharp_particle_selector() 
                    display_s2_final = cv2.cvtColor(img_fg_orig_uint8, cv2.COLOR_GRAY2BGR) 
                    if not self.s2_final_sharp_particles_df_for_viewer.empty:
                        for _, p_row in self.s2_final_sharp_particles_df_for_viewer.iterrows():
                            try: x,y,w,h = map(int, [p_row['bounding_x'],p_row['bounding_y'],p_row['bounding_w'],p_row['bounding_h']]); cv2.rectangle(display_s2_final, (x,y), (x+w,y+h), (0,255,255), 2) 
                            except: pass 
                    self.ax_dynamic_vis.imshow(display_s2_final)
                    if general_well_mask_for_roi_uint8 is not None: 
                        h_im, w_im = general_well_mask_for_roi_uint8.shape; xc_s2, yc_s2 = np.meshgrid(np.arange(w_im), np.arange(h_im)); masked_out_areas_s2 = (general_well_mask_for_roi_uint8 == 0) 
                        self.ax_dynamic_vis.contourf(xc_s2, yc_s2, masked_out_areas_s2, levels=[0.5,1.5], colors='red', hatches=['//'], alpha=0.3)
                    self.ax_dynamic_vis.set_title(f"S2 Filtered Output ({len(self.s2_final_sharp_particles_df_for_viewer)} particles)", fontsize=8, loc='center'); self.ax_dynamic_vis.set_axis_off()
                    try:
                        if os.path.exists(s2_temp_dir): shutil.rmtree(s2_temp_dir)
                    except Exception: pass
            self.fig_dynamic_vis.tight_layout(rect=[0, 0, 1, 0.96])
        
        elif mode == "Tracking":
            tracking_eff = params['tracking_effective']
            if not tracking_eff.get('enabled'):
                self.tracking_summary_text.config(state=tk.NORMAL); self.tracking_summary_text.delete(1.0, tk.END); self.tracking_summary_text.insert(tk.END, "Tracking is disabled in parameters. Enable it to see results."); self.tracking_summary_text.config(state=tk.DISABLED)
                self.ax_tracking_2d.clear(); self.ax_tracking_2d.set_title("2D Tracks (Disabled)"); self.ax_tracking_2d.set_axis_off(); self.canvas_tracking_2d.draw()
                self.ax_tracking_3d.clear(); self.ax_tracking_3d.set_title("3D Tracks (Disabled)"); self.ax_tracking_3d.set_axis_off(); self.canvas_tracking_3d.draw()
                return
            
            temp_s1_csv_path = os.path.join(self.temp_dir.name, "s1_for_tracking_viewer.csv")
            self.s1_particle_df_for_tracking.to_csv(temp_s1_csv_path, index=False)

            processed_data = _process_particle_coordinates_for_tracking(
                temp_s1_csv_path, 
                tracking_eff['pixel_size_um'], 
                tracking_eff['magnification']
            )
            
            tracked_data = _track_particles_3d(
                processed_data, 
                tracking_eff['search_range_mm'], 
                tracking_eff['memory_frames'], 
                tracking_eff['min_track_length']
            )

            num_total_particles = len(processed_data)
            num_tracks = 0
            num_tracked_particles = 0
            if not tracked_data.empty:
                num_tracks = tracked_data['tracked_particle'].nunique()
                num_tracked_particles = len(tracked_data)
            
            percent_tracked = (num_tracked_particles / num_total_particles * 100) if num_total_particles > 0 else 0
            
            summary_str = (
                f"Total Particles from CSV: {num_total_particles}\n"
                f"Number of Tracks Found: {num_tracks}\n"
                f"Particles in Tracks: {num_tracked_particles} ({percent_tracked:.1f}% of total)"
            )
            self.tracking_summary_text.config(state=tk.NORMAL); self.tracking_summary_text.delete(1.0, tk.END); self.tracking_summary_text.insert(tk.END, summary_str); self.tracking_summary_text.config(state=tk.DISABLED)
            
            self.ax_tracking_2d.clear(); self.ax_tracking_3d.clear()
            self.ax_tracking_2d.set_title("2D Particle Tracks (X vs Y)", fontsize=9); self.ax_tracking_2d.set_xlabel("X (mm)", fontsize=8); self.ax_tracking_2d.set_ylabel("Y (mm)", fontsize=8)
            self.ax_tracking_3d.set_title("3D Particle Tracks", fontsize=9); self.ax_tracking_3d.set_xlabel("X (mm)", fontsize=8); self.ax_tracking_3d.set_ylabel("Y (mm)", fontsize=8); self.ax_tracking_3d.set_zlabel("Z (mm)", fontsize=8)

            if not tracked_data.empty:
                base_cmap = plt.cm.get_cmap('tab20')
                unique_particles = sorted(tracked_data['tracked_particle'].unique())
                colors = {pid: base_cmap(i % 20) for i, pid in enumerate(unique_particles)}
                
                for particle_no, track in tracked_data.groupby('tracked_particle'):
                    self.ax_tracking_2d.plot(track['x'], track['y'], color=colors[particle_no], linewidth=1)
                    self.ax_tracking_3d.plot(track['x'], track['y'], track['z'], color=colors[particle_no], linewidth=1)
            
            self.canvas_tracking_2d.draw(); self.canvas_tracking_3d.draw()

        elif mode in ["S3", "S4"]: 
            self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111) 
            s3_s4_processing_params = params['s3_s4_segmentation_and_feature_params']
            self.ax_dynamic_vis.set_axis_off(); img_f_original = self.image_array_gray_float; title_b = os.path.basename(self.image_path or '')
            viewer_illum_s3s4_eff = s3_s4_processing_params.get('source_image_illumination_type')
            viewer_thresh_s3s4_eff = s3_s4_processing_params.get('thresholding_mode')
            unsharp_enabled_eff = s3_s4_processing_params.get('enable_unsharp_masking')
            title_suffix = f"(Illum: {viewer_illum_s3s4_eff}, Thresh: {viewer_thresh_s3s4_eff}, Unsharp: {'On' if unsharp_enabled_eff else 'Off'})"

            if mode == "S3":
                self.ax_dynamic_vis.set_title(f"S3: {title_b} {title_suffix}", fontsize=8)
                s3f,s3mp,s3eap = extract_ml_features(self.image_path, s3_s4_processing_params, img_f_original) 
                if not (s3f or s3mp or s3eap): self.ax_dynamic_vis.imshow(img_f_original, cmap='gray'); self.ax_dynamic_vis.set_title("S3: Failed", fontsize=8)
                else:
                    tmp_s3_vis = os.path.join(self.temp_dir.name, "s3_viewer_vis.png")
                    visualize_ml_features(img_f_original, s3mp, s3eap, tmp_s3_vis, ml_feature_params=s3_s4_processing_params) 
                    if os.path.exists(tmp_s3_vis): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp_s3_vis))
                    else: self.ax_dynamic_vis.imshow(img_f_original, cmap='gray')
            elif mode == "S4":
                self.ax_dynamic_vis.set_title(f"S4: {title_b} {title_suffix}", fontsize=8)
                _, _, s4_main_props_from_seg = segment_main_body_from_image_float(img_f_original.copy(), s3_s4_processing_params)
                s4_main_props_for_vis = None
                if s4_main_props_from_seg:
                    egg_post_frac = s3_s4_processing_params.get('egg_posterior_pixel_fraction', 0.45) 
                    s4_main_props_for_vis, _, _ = _calculate_and_attach_posterior_attributes(s4_main_props_from_seg, img_f_original.shape, egg_post_frac)
                s4rm, s4fm, _ = segment_main_body_from_image_float(img_f_original.copy(), s3_s4_processing_params) 
                if not (s4rm is not None or s4fm is not None) : self.ax_dynamic_vis.imshow(img_f_original, cmap='gray'); self.ax_dynamic_vis.set_title("S4: Segmentation Failed", fontsize=8)
                else:
                    tmp_s4_vis = os.path.join(self.temp_dir.name, "s4_viewer_vis.png")
                    visualize_detailed_morphology(img_f_original, s4_main_props_for_vis, s4rm, s4fm, tmp_s4_vis)
                    if os.path.exists(tmp_s4_vis): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp_s4_vis))
                    else: self.ax_dynamic_vis.imshow(img_f_original, cmap='gray')
            self.fig_dynamic_vis.tight_layout()
        self.canvas_dynamic_vis.draw()
    
    def update_s2_sharp_particle_selector(self):
        if not self.s2_final_sharp_particles_df_for_viewer.empty:
            df = self.s2_final_sharp_particles_df_for_viewer
            idx_col = 's1_viewer_fused_idx' if 's1_viewer_fused_idx' in df.columns else 'input_particle_index'
            if idx_col not in df.columns: df[idx_col] = df.index 
            df['X_display'] = df.get('X', pd.Series(np.nan, index=df.index)); df['Y_display'] = df.get('Y', pd.Series(np.nan, index=df.index))
            particle_display_names = [f"P_S1fusedIdx{int(row.get(idx_col, row.name))}_X{int(row.get('X_display', -1))}_Y{int(row.get('Y_display', -1))}" for _, row in df.iterrows()]
            self.s2_sharp_particle_selector['values'] = particle_display_names
            if particle_display_names: self.s2_selected_sharp_particle_var.set(particle_display_names[0]); self.s2_sharp_particle_selector.config(state="readonly"); self.on_s2_sharp_particle_selected() 
            else: self.s2_sharp_particle_selector.config(state="disabled"); self.s2_selected_sharp_particle_var.set("")
        else: self.s2_sharp_particle_selector['values'] = []; self.s2_sharp_particle_selector.config(state="disabled"); self.s2_selected_sharp_particle_var.set("")
        self.update_s2_single_crop_display()

    def on_s2_sharp_particle_selected(self, event=None): self.update_s2_single_crop_display()

    def update_s2_single_crop_display(self):
        self.ax_s2_single_crop.clear(); self.s2_sharp_particle_attributes_text.config(state=tk.NORMAL); self.s2_sharp_particle_attributes_text.delete(1.0, tk.END)
        selected_particle_str = self.s2_selected_sharp_particle_var.get()
        if not selected_particle_str or self.s2_final_sharp_particles_df_for_viewer.empty:
            self.ax_s2_single_crop.imshow(np.zeros((100,100)), cmap='gray'); self.ax_s2_single_crop.set_title("S2 Final Crop"); self.s2_sharp_particle_attributes_text.insert(tk.END, "No particle selected.")
        else:
            try:
                match = re.match(r"P_S1fusedIdx(\d+)_X(-?\d+)_Y(-?\d+)", selected_particle_str)
                if not match: raise ValueError(f"Could not parse S2 particle string: {selected_particle_str}")
                s1_fused_idx = int(match.group(1)); idx_col_to_match = 's1_viewer_fused_idx' if 's1_viewer_fused_idx' in self.s2_final_sharp_particles_df_for_viewer.columns else 'input_particle_index'
                selected_rows = self.s2_final_sharp_particles_df_for_viewer[self.s2_final_sharp_particles_df_for_viewer[idx_col_to_match].astype(int) == s1_fused_idx]
                if selected_rows.empty: raise ValueError(f"No S2 particle data for S1 Fused Idx {s1_fused_idx}")
                
                particle_attrs = selected_rows.iloc[0]
                fg_fn = self.s1_selected_fg_filepath_var.get()
                original_frame_path = os.path.join(self.s1_image_directory_path, fg_fn)
                orig_color_img = core_load_image(original_frame_path, 'color')
                orig_gray_img = convert_to_grayscale(orig_color_img)
                if orig_color_img is None or orig_gray_img is None: raise ValueError("Could not load original frame for S2 crop.")
                
                final_crop_mode = self.param_vars['s2_intermediate_crop_mode'].get()
                final_cropper_cfg = {'mode': final_crop_mode}
                if final_crop_mode == 'centroid_square':
                    final_cropper_cfg['square_size'] = self.param_vars['s2_final_crop_settings_centroid_square_size'].get()
                elif final_crop_mode == 'rotational_rectangle':
                    final_cropper_cfg['rotational_padding'] = self.param_vars['s2_final_crop_settings_rotational_padding'].get()
                
                final_cropper = ParticleCropper(final_cropper_cfg)
                
                final_crop_gray, _ = final_cropper.crop_single_particle_from_image(orig_color_img, orig_gray_img, particle_attrs, output_dir_for_crop=None)

                if final_crop_gray is not None:
                    final_crop_bgr = cv2.cvtColor(final_crop_gray, cv2.COLOR_GRAY2BGR)
                    viewer_illum_s2 = self.param_vars['viewer_source_illumination_type'].get(); 
                    viewer_thresh_params_s2 = {
                        'thresholding_mode': self.param_vars['viewer_thresholding_mode'].get(),
                        'otsu_threshold_multiplier': self.param_vars['viewer_otsu_threshold_multiplier'].get(),
                        'fixed_threshold_value': self.param_vars['viewer_fixed_threshold_value'].get()
                    }
                    is_on_this_crop_boundary = detect_cropped_particle_on_boundary(
                        final_crop_gray, 
                        f"viewer_fc_S1fusedIdx{s1_fused_idx}.png", 
                        viewer_illum_s2, 
                        viewer_thresh_params_s2 
                    )
                    self.ax_s2_single_crop.imshow(final_crop_bgr); self.ax_s2_single_crop.set_title(f"S2 Final Crop ({final_crop_gray.shape[1]}x{final_crop_gray.shape[0]})")
                    attr_text = f"S1 Frame: {fg_fn}\nS1 Fused Idx: {s1_fused_idx}\nGlobal Centroid: X={int(particle_attrs.get('X', -1))}, Y={int(particle_attrs.get('Y', -1))}\n\n"
                    attr_text += f"S2 Intermed. Crop: {particle_attrs.get('cropped_filename', 'N/A')}\n  Black Mask: {particle_attrs.get('is_black_mask', 'N/A')}\n  On Intermed. Boundary: {particle_attrs.get('is_on_crop_boundary', 'N/A')}\n\n" 
                    attr_text += f"S2 Final Crop (this view):\n  Mode: {final_crop_mode}\n  Boundary Flag (this view): {is_on_this_crop_boundary}\n\nSharpness:\n"
                    for metric in AVAILABLE_SHARPNESS_METRICS: attr_text += f"  {metric}: {particle_attrs.get(metric, np.nan):.2f}\n  {metric}_norm: {particle_attrs.get(metric+'_norm', np.nan):.3f}\n"
                    attr_text += f"  combined_sharpness: {particle_attrs.get('combined_sharpness', np.nan):.3f}\n"; self.s2_sharp_particle_attributes_text.insert(tk.END, attr_text)
                else: self.ax_s2_single_crop.imshow(np.zeros((100,100)), cmap='gray'); self.s2_sharp_particle_attributes_text.insert(tk.END, "Error final cropping.")
            except Exception as e: self.ax_s2_single_crop.imshow(np.zeros((100,100)), cmap='gray'); self.s2_sharp_particle_attributes_text.insert(tk.END, f"Error: {e}")
        self.ax_s2_single_crop.set_axis_off(); self.canvas_s2_single_crop.draw(); self.s2_sharp_particle_attributes_text.config(state=tk.DISABLED)

def main(): 
    try: load_config(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml"))
    except Exception as e: messagebox.showerror("Startup Error", f"Failed to load config: {e}"); return
    root = tk.Tk(); FeatureViewerApp(root); root.mainloop()

if __name__ == '__main__':
    main()