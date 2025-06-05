# interactive_viewer.py (to be placed in well_plate_analyzer/scripts/)

import sys
import os
import pandas as pd

# Add the project root (well_plate_analyzer) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg') # Must be before importing pyplot
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

try:
    from src.utils.config_loader import get_config, load_config 
    from src.core.ml_feature_extractor import extract_ml_features
    from src.visualization.feature_visualization import visualize_ml_features
    from src.core.detailed_morphology_calculator import reconstruct_main_body_masks 
    from src.visualization.stage_04_visualization import visualize_detailed_morphology 
    from src.utils.common_exceptions import ConfigError, DataValidationError 
    from src.stages.stage_01_initial_detection import (
        _parse_image_filenames, 
        _analyze_single_subtracted_image_global_thresh, 
        _filter_image_df,
        _analyze_unsubtracted_locally, 
        _compare_and_fuse_particle_dfs 
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
        master.title("Interactive Viewer (S1 Local Unsub Thresh)"); master.geometry("1600x950") 
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
            s3_feat_cfg_base = s3_ml_cfg.get('feature_extraction_params', {})
            
            self.s1_raw_config = cfg.get('stage_01_initial_detection', {})
            s1_particle_cfg = self.s1_raw_config.get('particle_analysis', {}) 
            s1_unsub_particle_cfg = self.s1_raw_config.get('unsubtracted_particle_analysis', {})
            s1_comp_cfg = self.s1_raw_config.get('particle_comparison', {}) 
            s1_well_mask_cfg = self.s1_raw_config.get('well_masking', {})
            
            s2_cfg = cfg.get('stage_02_quality_filter', {})
            s2_intermediate_crop_cfg = s2_cfg.get('intermediate_crop', {})
            s2_black_mask_cfg = s2_cfg.get('black_mask_check', {})
            s2_final_sharp_filter_cfg = s2_cfg.get('final_sharp_object_filter', {})
            s2_filters_enabled_cfg = s2_cfg.get('filters_enabled', {})
            self.s2_final_crop_size_from_config = s2_cfg.get('final_sharp_crop_size', 150) 

            self.s1_convolution_kernel_list = s1_particle_cfg.get('convolution_kernel')
            
            self.default_params = {
                'viewer_source_illumination_type': global_source_illum_default,
                'viewer_thresholding_mode': global_thresh_cfg_default.get('mode', 'otsu'),
                'viewer_otsu_threshold_multiplier': global_thresh_cfg_default.get('otsu_threshold_multiplier', 0.45),
                'viewer_fixed_threshold_value': global_thresh_cfg_default.get('fixed_threshold_value', 70),
            }
            self.default_params.update({f"s3_{k}": v for k, v in s3_feat_cfg_base.items()})
            self.default_params.update({
                's1_bg_offset_val': self.s1_raw_config.get('bg_offset_val', 110),
                's1_sub_apply_convolution': s1_particle_cfg.get('apply_convolution', True),
                's1_sub_min_microorganism_area': s1_particle_cfg.get('min_microorganism_area', 200),
                's1_sub_microorganism_blur_kernel': s1_particle_cfg.get('microorganism_blur_kernel', 5),
            })
            self.default_params.update({
                's1_well_mask_create_mask': s1_well_mask_cfg.get('create_mask', True),
                's1_well_mask_auto_center_mask': s1_well_mask_cfg.get('auto_center_mask', True),
                's1_well_mask_radius_px': s1_well_mask_cfg.get('radius_px', 2000),
                's1_well_mask_radius_fraction': s1_well_mask_cfg.get('radius_fraction', 0.67)
            })
            self.default_params.update({ 
                's1_unsub_min_particle_area': s1_unsub_particle_cfg.get('min_particle_area', 300),
                's1_unsub_local_otsu_roi_size': s1_unsub_particle_cfg.get('local_otsu_roi_size', 150) 
            })
            self.default_params.update({
                's1_comp_search_radius_factor': s1_comp_cfg.get('search_radius_factor', 2.0),
                's1_comp_default_particle_area': s1_comp_cfg.get('default_particle_area', 500),
            })
            self.default_params.update({
                's2_intermediate_crop_mode': s2_intermediate_crop_cfg.get('mode', 'centroid_square'),
                's2_intermediate_crop_square_size': s2_intermediate_crop_cfg.get('square_size', 128),
                's2_black_mask_min_area_ratio': s2_black_mask_cfg.get('min_area_ratio', 0.05),
                's2_final_sharpness_metric_to_filter': 'combined_sharpness', 
                's2_final_sharpness_threshold_min': s2_final_sharp_filter_cfg.get('combined_sharpness_min', 0.3),
                's2_final_crop_size': self.s2_final_crop_size_from_config,
                's2_enable_black_mask_filter': s2_filters_enabled_cfg.get('black_mask_artifact', True),
                's2_enable_on_boundary_filter': s2_filters_enabled_cfg.get('on_crop_boundary', True),
                's2_enable_final_sharpness_filter': s2_filters_enabled_cfg.get('final_sharpness', True),
            })

            self.current_params = self.default_params.copy()
            self._initialize_param_vars_and_specs()

        except Exception as e: logger.exception("Config Error"); messagebox.showerror("Config Error", f"{e}"); master.destroy(); return
        self.temp_dir = tempfile.TemporaryDirectory(); master.protocol("WM_DELETE_WINDOW", self.on_closing); self.setup_gui(); self.on_mode_change()

    def _initialize_param_vars_and_specs(self):
        all_specs = {
            **self._get_global_param_specs(), **self._get_s1_param_specs(), 
            **self._get_s2_param_specs(), **self._get_s3_param_specs()
        }
        for name, default_val in self.default_params.items():
            spec = all_specs.get(name)
            if spec: 
                self.param_specs[name] = spec; param_type = spec['type']
                if param_type == 'int':
                    self.param_vars[name] = tk.IntVar(value=int(default_val))
                    self.param_str_vars[name] = tk.StringVar(value=str(int(default_val)))
                    self.param_vars[name].trace_add("write", lambda n,i,m,var_name=name: self._sync_num_to_str(var_name))
                elif param_type == 'float':
                    self.param_vars[name] = tk.DoubleVar(value=float(default_val))
                    res = spec.get('res', 0.1); fmt_str = str(res)
                    num_decimals = len(fmt_str.split('.')[-1]) if '.' in fmt_str else 0
                    fmt = f".{max(1,num_decimals)}f"
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
        for canvas_attr in ['canvas_raw', 'canvas_dynamic_vis', 'canvas_s2_single_crop']: 
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
        for mode_disp, mode_val in [("S1", "S1"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4")]: 
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
        self.s1_subtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Subtracted Analysis Params", padding=5)
        self.create_s1_subtracted_param_controls(self.s1_subtracted_params_frame)
        self.s1_wellmask_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Well Masking Params", padding=5)
        self.create_s1_wellmask_param_controls(self.s1_wellmask_params_frame)
        self.s1_unsubtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Unsubtracted FG Analysis Params (Local Thresh)", padding=5)
        self.create_s1_unsubtracted_param_controls(self.s1_unsubtracted_params_frame)
        self.s1_comparison_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Particle Comparison Params", padding=5)
        self.create_s1_comparison_param_controls(self.s1_comparison_params_frame)
        self.s2_params_frame_main = ttk.Frame(self.controls_frame_inner)
        self.create_s2_param_controls(self.s2_params_frame_main)
        self.s3_params_frame = ttk.Frame(self.controls_frame_inner); 
        self.s3_s4_params_container_frame = ttk.Labelframe(self.s3_params_frame, text="S3/S4 Core Feature Params", padding=5)
        self.create_s3_param_controls(self.s3_s4_params_container_frame)

        param_buttons = ttk.Frame(self.controls_frame_inner); param_buttons.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=5)
        ttk.Button(param_buttons, text="Reset", command=self.reset_parameters).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(param_buttons, text="Update", command=self.process_and_display_safe).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        
        self.vis_notebook = ttk.Notebook(paned_window); paned_window.add(self.vis_notebook, weight=3)
        main_vis_tab = ttk.Frame(self.vis_notebook); self.vis_notebook.add(main_vis_tab, text=" Frame Overview ")
        self.fig_raw, self.ax_raw = plt.subplots(); self.ax_raw.set_title("Raw Input"); self.ax_raw.set_axis_off()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=main_vis_tab); self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_raw.draw()
        self.fig_dynamic_vis = plt.figure(); self.ax_dynamic_vis = None 
        self.canvas_dynamic_vis = FigureCanvasTkAgg(self.fig_dynamic_vis, master=main_vis_tab); self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_dynamic_vis.draw()
        self.s2_single_crop_tab = ttk.Frame(self.vis_notebook) 
        self.setup_s2_single_crop_tab(self.s2_single_crop_tab)

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
        self.global_params_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
        self.s1_params_frame_main.pack_forget(); self.s2_params_frame_main.pack_forget(); self.s3_params_frame.pack_forget()
        self.s1_fg_image_selector_label.pack_forget(); self.s1_fg_image_selector.pack_forget(); self.s2_status_label.pack_forget()
        try: self.vis_notebook.index(self.s2_single_crop_tab); self.vis_notebook.forget(self.s2_single_crop_tab) 
        except tk.TclError: pass 
        
        if mode == "S1":
            self.s1_params_frame_main.pack(fill=tk.X, expand=True)
            self.s1_subtracted_params_frame.pack(fill=tk.X, expand=True, pady=3); self.s1_wellmask_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_unsubtracted_params_frame.pack(fill=tk.X, expand=True, pady=3); self.s1_comparison_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.load_button.config(text="Load S1 Dir"); self.s1_fg_image_selector_label.pack(side=tk.LEFT, padx=(10,2), before=self.status_label)
            self.s1_fg_image_selector.pack(side=tk.LEFT, padx=2, before=self.status_label); self.s1_fg_image_selector.config(state="readonly" if self.s1_image_directory_path and not self.s1_image_file_list_df.empty else "disabled")
            self.update_global_threshold_param_visibility() 
        elif mode == "S2":
            self.s2_params_frame_main.pack(fill=tk.X, expand=True); self.load_button.config(text="Load S2 Dir") 
            self.s1_fg_image_selector_label.pack(side=tk.LEFT, padx=(10,2), before=self.status_label); self.s1_fg_image_selector.pack(side=tk.LEFT, padx=2, before=self.status_label)
            self.s1_fg_image_selector.config(state="readonly" if self.s1_image_directory_path and not self.s1_image_file_list_df.empty else "disabled")
            self.s2_status_label.pack(side=tk.LEFT, padx=(10,0), after=self.s1_fg_image_selector); self.vis_notebook.add(self.s2_single_crop_tab, text=" S2 Crop Inspector ") 
            self.s2_selected_sharp_particle_var.set(""); self.update_s2_single_crop_display() 
        elif mode in ["S3", "S4"]:
            self.s3_params_frame.pack(fill=tk.X, expand=True)
            self.s3_s4_params_container_frame.pack(fill=tk.X, expand=True, pady=3)
            self.load_button.config(text="Load Img (S3/S4)")
            self.update_global_threshold_param_visibility() 

        if mode != "S1" and mode !="S2": self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame(); self.s1_fg_image_selector['values'] = []; self.s1_selected_fg_filepath_var.set("")
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
        if otsu_widget_name in self.param_widgets:
            if mode == 'otsu': self.param_widgets[otsu_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[otsu_widget_name].pack_forget()
        if fixed_widget_name in self.param_widgets:
            if mode == 'fixed': self.param_widgets[fixed_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[fixed_widget_name].pack_forget()
        self.schedule_update()
    
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
        label_text = label_text_raw.replace("S1 Sub ", "").replace("S1 Unsub ", "").replace("S1 Comp ", "").replace("S1 Well Mask ", "")
        label_text = label_text.replace("S2 Enable ", "").replace("S2 Viewer ", "").replace("S2 ", "")
        label_text = label_text.replace("S3 ", "").replace("Viewer ", "") 
        ttk.Label(container, text=label_text + ":", width=25, anchor="w").pack(side=tk.LEFT, padx=(0,5))
        tk_var = self.param_vars.get(name) 
        if param_type == 'choice':
            cmd = self.schedule_update 
            if name == 'viewer_thresholding_mode': cmd = self.update_global_threshold_param_visibility
            ttk.OptionMenu(container, tk_var, current_val_for_var, *spec['options'], command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X)
        elif param_type in ['int', 'float']:
            str_tk_var = self.param_str_vars.get(name)
            ttk.Button(container, text="-", command=lambda p_name=name: self._adjust_param_value(p_name, -1), width=2).pack(side=tk.LEFT)
            entry = ttk.Entry(container, textvariable=str_tk_var, width=7, justify='center')
            entry.pack(side=tk.LEFT, padx=2); entry.bind("<Return>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e)); entry.bind("<FocusOut>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e))
            ttk.Button(container, text="+", command=lambda p_name=name: self._adjust_param_value(p_name, 1), width=2).pack(side=tk.LEFT)
        elif param_type == 'bool': ttk.Checkbutton(container, variable=tk_var, command=self.schedule_update).pack(side=tk.LEFT)
        if not (name in ['viewer_otsu_threshold_multiplier', 'viewer_fixed_threshold_value']):
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

    def _get_s2_param_specs(self):
        return {
            's2_intermediate_crop_mode': {'options': ['bounding_box', 'centroid_square'], 'type': 'choice'},
            's2_intermediate_crop_square_size': {'min': 32, 'max': 512, 'type': 'int', 'step': 4},
            's2_black_mask_min_area_ratio': {'min': 0.01, 'max': 0.5, 'type': 'float', 'res': 0.01},
            's2_final_sharpness_metric_to_filter': {'options': AVAILABLE_SHARPNESS_METRICS + ['combined_sharpness'], 'type': 'choice'},
            's2_final_sharpness_threshold_min': {'min': 0.0, 'max': 5000.0, 'type': 'float', 'res': 0.01}, 
            's2_final_crop_size': {'min': 32, 'max': 512, 'type': 'int', 'step': 4},
            's2_enable_black_mask_filter': {'type': 'bool'},
            's2_enable_on_boundary_filter': {'type': 'bool'},
            's2_enable_final_sharpness_filter': {'type': 'bool'},
        }

    def _get_s3_param_specs(self): 
        return {'s3_gaussian_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float', 'res': 0.1},
                's3_closing_selem_size': {'min': 0, 'max': 10, 'type': 'int', 'step': 1},
                's3_min_noise_size': {'min': 1, 'max': 1000, 'type': 'int', 'step': 10}, 
                's3_min_significant_area': {'min': 10, 'max': 2000, 'type': 'int', 'step': 10},
                's3_egg_posterior_pixel_fraction': {'min': 0.05, 'max': 0.95, 'type': 'float', 'res': 0.01}, 
                's3_egg_threshold_quantile': {'min': 0.01, 'max': 0.99, 'type': 'float', 'res': 0.01},
                's3_egg_min_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10}, 
                's3_egg_max_area': {'min': 50, 'max': 20000, 'type': 'int', 'step': 100},
                's3_egg_min_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 
                's3_egg_max_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
                's3_egg_min_eccentricity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 
                's3_orientation_diff_tolerance_deg': {'min': 0, 'max': 90, 'type': 'int', 'step': 1} }

    def create_global_param_controls(self, parent_frame):
        global_specs = self._get_global_param_specs()
        ordered_global_params = ['viewer_source_illumination_type', 'viewer_thresholding_mode', 
                                 'viewer_otsu_threshold_multiplier', 'viewer_fixed_threshold_value']
        for name in ordered_global_params:
            if name in self.default_params:
                self._create_generic_param_control(parent_frame, name, global_specs[name], self.current_params.get(name, self.default_params[name]))

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
        ordered_s2_params = ['s2_intermediate_crop_mode', 's2_intermediate_crop_square_size', 's2_black_mask_min_area_ratio', 
                             's2_final_sharpness_metric_to_filter', 's2_final_sharpness_threshold_min', 's2_final_crop_size']
        for name in ordered_s2_params:
            if name in self.default_params:
                self._create_generic_param_control(other_params_frame, name, s2_specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s3_param_controls(self, parent_frame): 
        s3_specs = self._get_s3_param_specs()
        ordered_s3 = [p for p in s3_specs] 
        for name in ordered_s3:
            if name in self.default_params and name in s3_specs: 
                self._create_generic_param_control(parent_frame, name, s3_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def reset_parameters(self): 
        if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
        for name, default_val in self.default_params.items():
            if name in self.param_vars: self.param_vars[name].set(default_val)
        self.on_mode_change(); self.update_global_threshold_param_visibility()

    def dispatch_load_action(self):
        if self.active_visualization_mode.get() in ["S1", "S2"]: self._load_s1_directory()
        else: self._load_s3_s4_image()

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
                if os.path.exists(path): img_show_raw_uint8 = core_load_image(path, 'gray'); img_show = skimage.util.img_as_float(img_show_raw_uint8); title = f"{mode} FG: {fn}"
                else: title = f"{mode} FG: Not found"
            else: title = f"{mode}: Select FG frame"
        elif self.image_array_gray_float is not None and mode in ["S3","S4"]: img_show = self.image_array_gray_float; title = f"Input: {os.path.basename(self.image_path or 'No Img')}"
        if img_show is not None: self.ax_raw.imshow(img_show, cmap='gray', vmin=0.0, vmax=1.0)
        else: self.ax_raw.imshow(np.zeros((100,100)), cmap='gray') 
        self.ax_raw.set_title(title, fontsize=8); self.ax_raw.set_axis_off(); self.canvas_raw.draw()

    def collect_current_params(self):
        gui_params = {name: var.get() for name, var in self.param_vars.items()}
        global_illumination_viewer = gui_params.get('viewer_source_illumination_type')
        global_threshold_config_viewer = {
            'mode': gui_params.get('viewer_thresholding_mode'),
            'otsu_threshold_multiplier': gui_params.get('viewer_otsu_threshold_multiplier'),
            'fixed_threshold_value': gui_params.get('viewer_fixed_threshold_value'),
        }
        self.current_params['s1_subtracted_effective'] = {k.replace('s1_sub_', ''):v for k,v in gui_params.items() if k.startswith('s1_sub_')}
        self.current_params['s1_subtracted_effective']['convolution_kernel'] = self.s1_convolution_kernel_list
        self.current_params['s1_bg_offset_effective'] = gui_params.get('s1_bg_offset_val', self.default_params['s1_bg_offset_val'])
        self.current_params['s1_well_mask_effective'] = {k.replace('s1_well_mask_', ''):v for k,v in gui_params.items() if k.startswith('s1_well_mask_')}
        self.current_params['s1_unsubtracted_effective'] = {
            'min_particle_area': gui_params.get('s1_unsub_min_particle_area'),
            'local_otsu_roi_size': gui_params.get('s1_unsub_local_otsu_roi_size') 
        }
        self.current_params['s1_unsubtracted_global_thresh_effective'] = global_threshold_config_viewer.copy()
        self.current_params['s1_unsubtracted_illum_effective'] = global_illumination_viewer
        self.current_params['s1_comparison_effective'] = {k.replace('s1_comp_', ''):v for k,v in gui_params.items() if k.startswith('s1_comp_')}
        self.current_params['s2_effective'] = {
            'intermediate_crop': {
                'mode': gui_params.get('s2_intermediate_crop_mode'),
                'square_size': gui_params.get('s2_intermediate_crop_square_size'),
                'bbox_column_mapping': get_config().get('stage_02_quality_filter', {}).get('intermediate_crop', {}).get('bbox_column_mapping', {'x': 'bounding_x', 'y': 'bounding_y', 'w': 'bounding_w', 'h': 'bounding_h'})
            },
            'black_mask_check': {'min_area_ratio': gui_params.get('s2_black_mask_min_area_ratio')},
            'filters_enabled': { 
                'black_mask_artifact': gui_params.get('s2_enable_black_mask_filter', True),
                'on_crop_boundary': gui_params.get('s2_enable_on_boundary_filter', True),
                'final_sharpness': gui_params.get('s2_enable_final_sharpness_filter', True),
            },
            'sharpness_metrics': AVAILABLE_SHARPNESS_METRICS, 
            'final_sharp_object_filter': { f"{gui_params.get('s2_final_sharpness_metric_to_filter')}_min": gui_params.get('s2_final_sharpness_threshold_min')},
            'final_sharp_crop_size': gui_params.get('s2_final_crop_size'),
            'viewer_source_illumination_type': global_illumination_viewer 
        }
        self.current_params['s3_core_effective'] = {k.replace('s3_', ''):v for k,v in gui_params.items() if k.startswith('s3_')}
        self.current_params['s3_core_effective']['source_image_illumination_type'] = global_illumination_viewer
        self.current_params['s3_core_effective'].update(global_threshold_config_viewer)
        return self.current_params

    def process_and_display_safe(self): 
        active_mode, valid_input = self.active_visualization_mode.get(), False
        if active_mode in ["S1", "S2"]:
            if self.s1_image_directory_path and not self.s1_image_file_list_df.empty and self.s1_selected_fg_filepath_var.get(): valid_input = True
            else: self.status_label.config(text=f"{active_mode}: Load dir & select FG frame.")
        elif self.image_array_gray_float is not None: valid_input = True 
        else: self.status_label.config(text="S3/S4: Load image.")
        if not valid_input:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
            if self.fig_dynamic_vis: # Check if figure exists
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
        try: self.process_and_display(); self.status_label.config(text=f"Ready. Mode: {active_mode}.")
        except DataValidationError as dve: logger.error(f"Data Validation Error: {dve}"); messagebox.showerror("Data Error", f"{dve}"); self.status_label.config(text=f"Data Error: {dve}")
        except Exception as e: logger.exception("Proc Error"); messagebox.showerror("Error", f"{e}"); self.status_label.config(text=f"Error: {e}")
        finally:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None

    def process_and_display(self): 
        params = self.collect_current_params() 
        mode = self.active_visualization_mode.get()
        
        # Correctly handle clearing/recreating axes
        if isinstance(self.ax_dynamic_vis, np.ndarray): # It's an array of Axes (e.g., from subplots)
            for ax_item in self.ax_dynamic_vis.flat: # Iterate and clear each one
                ax_item.clear()
                ax_item.set_axis_off()
        elif self.ax_dynamic_vis is not None: # It's a single Axes object
            self.ax_dynamic_vis.clear()
            self.ax_dynamic_vis.set_axis_off()
        
        # If switching from a multi-plot mode (like S1 old) to single, or vice-versa,
        # it's often safer to clear the whole figure and re-add subplots as needed.
        self.fig_dynamic_vis.clear() 
        
        viewer_global_source_illum = params['s1_unsubtracted_illum_effective']
        viewer_global_thresh_cfg = params['s1_unsubtracted_global_thresh_effective']

        if mode == "S1":
            # Setup for 2x2 S1 display
            axes_s1 = self.fig_dynamic_vis.subplots(2, 2) 
            self.ax_dynamic_vis = axes_s1 # Store the array of axes
            
            s1_sub_eff = params['s1_subtracted_effective']
            s1_wm_eff = params['s1_well_mask_effective']
            s1_unsub_eff = params['s1_unsubtracted_effective']
            s1_comp_eff = params['s1_comparison_effective']
            s1_bg_off = params['s1_bg_offset_effective']
            
            fg_fn = self.s1_selected_fg_filepath_var.get()
            self.fig_dynamic_vis.suptitle(f"S1 Output (Local Unsub) - FG: {fg_fn}", fontsize=10)

            fg_row = self.s1_image_file_list_df[self.s1_image_file_list_df['filename'] == fg_fn].iloc[0]
            bg_row = self.s1_image_file_list_df.iloc[(fg_row.name + s1_bg_off) % len(self.s1_image_file_list_df)]
            img_fg_orig_uint8 = core_load_image(fg_row['full_path'],'gray')
            img_bg_orig_uint8 = core_load_image(bg_row['full_path'],'gray')
            if img_fg_orig_uint8 is None or img_bg_orig_uint8 is None: raise ValueError("S1 Viewer: FG/BG load fail.")
            
            well_mask_uint8 = None
            if s1_wm_eff.get('create_mask'):
                _, well_mask_uint8 = mask_well_area(img_fg_orig_uint8.copy(), s1_wm_eff, self.s1_system_config_from_file)

            # Subtracted Path
            img_a_sub, img_b_sub = img_fg_orig_uint8.copy(), img_bg_orig_uint8.copy()
            if viewer_global_source_illum == 'collimated':
                img_a_sub = cv2.bitwise_not(img_a_sub); img_b_sub = cv2.bitwise_not(img_b_sub)
            subtracted_img = subtract_images(img_a_sub, img_b_sub)
            if subtracted_img is None: raise ValueError("S1 Viewer: Subtraction failed.")
            img_for_sub_analysis = apply_mask(subtracted_img, well_mask_uint8) if well_mask_uint8 is not None else subtracted_img
            
            df_sub_list_dict, binary_mask_sub_disp = _analyze_single_subtracted_image_global_thresh(
                img_for_sub_analysis, s1_sub_eff, viewer_global_thresh_cfg
            )
            df_subtracted_particles_viewer = pd.DataFrame(df_sub_list_dict)
            if not df_subtracted_particles_viewer.empty:
                df_subtracted_particles_viewer['SAMPLE_ID'] = "viewer_s1"
                df_subtracted_particles_viewer['source_image_filename'] = fg_fn

            # Unsubtracted Path (Local)
            img_for_unsub_prep = img_fg_orig_uint8.copy()
            if well_mask_uint8 is not None:
                if viewer_global_source_illum == 'collimated':
                    pixels_in_well = img_fg_orig_uint8[well_mask_uint8 == 255]
                    if pixels_in_well.size > 0:
                        hist = cv2.calcHist([img_fg_orig_uint8], [0], well_mask_uint8, [256], [0,256])
                        dom_bg = np.argmax(hist); filled_out = np.full_like(img_fg_orig_uint8, dom_bg); filled_out[well_mask_uint8 == 255] = img_fg_orig_uint8[well_mask_uint8 == 255]
                        img_for_unsub_prep = filled_out
                    else: img_for_unsub_prep = apply_mask(img_fg_orig_uint8.copy(), well_mask_uint8) if well_mask_uint8 is not None else img_fg_orig_uint8.copy()
                else: img_for_unsub_prep = apply_mask(img_fg_orig_uint8.copy(), well_mask_uint8) if well_mask_uint8 is not None else img_fg_orig_uint8.copy()
            
            df_unsub_list_dict, composite_binary_mask_unsub_disp, \
                (example_roi_img_disp, example_roi_hist_ravel_disp, example_roi_otsu_disp) = _analyze_unsubtracted_locally(
                    img_for_unsub_prep, df_sub_list_dict, 
                    s1_unsub_eff['local_otsu_roi_size'], 
                    s1_unsub_eff, viewer_global_thresh_cfg, viewer_global_source_illum
            )
            df_unsubtracted_particles_viewer = pd.DataFrame(df_unsub_list_dict)
            if not df_unsubtracted_particles_viewer.empty:
                df_unsubtracted_particles_viewer['SAMPLE_ID'] = "viewer_s1"
                df_unsubtracted_particles_viewer['source_image_filename'] = fg_fn

            df_fused_s1_viewer = _compare_and_fuse_particle_dfs(df_subtracted_particles_viewer, df_unsubtracted_particles_viewer, s1_comp_eff)
            
            axes_s1[0,0].imshow(binary_mask_sub_disp if binary_mask_sub_disp is not None else np.zeros_like(img_fg_orig_uint8), cmap='gray')
            axes_s1[0,0].set_title(f"Subtracted Binary ({len(df_sub_list_dict)} p.)", fontsize=8); axes_s1[0,0].set_axis_off()
            
            axes_s1[0,1].imshow(composite_binary_mask_unsub_disp if composite_binary_mask_unsub_disp is not None else np.zeros_like(img_fg_orig_uint8), cmap='gray')
            axes_s1[0,1].set_title(f"Unsub Local Binary Comp. ({len(df_unsub_list_dict)} p.)", fontsize=8); axes_s1[0,1].set_axis_off()

            if example_roi_img_disp is not None:
                axes_s1[1,0].imshow(example_roi_img_disp, cmap='gray')
                axes_s1[1,0].set_title("Example Local ROI (for hist)", fontsize=8); axes_s1[1,0].set_axis_off()
                if example_roi_hist_ravel_disp is not None and example_roi_hist_ravel_disp.size > 0:
                    axes_s1[1,1].hist(example_roi_hist_ravel_disp, bins=256, range=(0.0,1.0), fc='b', ec='b', alpha=0.7)
                    axes_s1[1,1].set_title("Hist of Example ROI", fontsize=8)
                    if example_roi_otsu_disp is not None:
                        axes_s1[1,1].axvline(example_roi_otsu_disp, color='r', ls='--', lw=1)
                        otsu_mult_viewer = viewer_global_thresh_cfg.get('otsu_threshold_multiplier', 0.45)
                        final_local_calc_thresh = example_roi_otsu_disp * otsu_mult_viewer
                        axes_s1[1,1].axvline(final_local_calc_thresh, color='m', linestyle=':', linewidth=1)
                    axes_s1[1,1].tick_params(axis='both', which='major', labelsize=7); axes_s1[1,1].set_xlim(0,1)
                else: axes_s1[1,1].text(0.5,0.5,"No Hist Data", ha='center', va='center'); axes_s1[1,1].set_axis_off()
            else: 
                axes_s1[1,0].text(0.5,0.5,"No ROI", ha='center', va='center'); axes_s1[1,0].set_axis_off()
                axes_s1[1,1].text(0.5,0.5,"No Hist", ha='center', va='center'); axes_s1[1,1].set_axis_off()

        elif mode == "S2": 
            self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111) 
            self.s2_status_text_var.set("S2 Status: Simulating S1 for frame...")
            
            s1_sub_eff = params['s1_subtracted_effective']
            s1_wm_eff = params['s1_well_mask_effective']
            s1_unsub_eff = params['s1_unsubtracted_effective']
            s1_comp_eff = params['s1_comparison_effective']
            s1_bg_off = params['s1_bg_offset_effective']
            s2_eff_config = params['s2_effective']

            fg_fn = self.s1_selected_fg_filepath_var.get()
            self.fig_dynamic_vis.suptitle(f"S2 Quality Filter Output - FG: {fg_fn}", fontsize=10)
            
            fg_row = self.s1_image_file_list_df[self.s1_image_file_list_df['filename'] == fg_fn].iloc[0]
            img_fg_orig_uint8_s2 = core_load_image(fg_row['full_path'], 'gray')
            img_fg_orig_color_s2 = core_load_image(fg_row['full_path'], 'color') 
            bg_row_s2 = self.s1_image_file_list_df.iloc[(fg_row.name + s1_bg_off) % len(self.s1_image_file_list_df)]
            img_bg_orig_uint8_s2 = core_load_image(bg_row_s2['full_path'], 'gray')
            if img_fg_orig_uint8_s2 is None or img_fg_orig_color_s2 is None or img_bg_orig_uint8_s2 is None: raise ValueError("S2 Viewer: FG/BG load fail.")
            
            well_mask_uint8_s2 = None
            if s1_wm_eff.get('create_mask'): _, well_mask_uint8_s2 = mask_well_area(img_fg_orig_uint8_s2.copy(), s1_wm_eff, self.s1_system_config_from_file)
            
            img_a_sub_s2, img_b_sub_s2 = img_fg_orig_uint8_s2.copy(), img_bg_orig_uint8_s2.copy()
            if viewer_global_source_illum == 'collimated': img_a_sub_s2, img_b_sub_s2 = cv2.bitwise_not(img_a_sub_s2), cv2.bitwise_not(img_b_sub_s2)
            sub_img_s2 = subtract_images(img_a_sub_s2, img_b_sub_s2)
            if sub_img_s2 is None: sub_img_s2 = np.zeros_like(img_fg_orig_uint8_s2) 
            img_for_sub_s2 = apply_mask(sub_img_s2, well_mask_uint8_s2) if well_mask_uint8_s2 is not None else sub_img_s2
            df_sub_s2_list_dict, _ = _analyze_single_subtracted_image_global_thresh(img_for_sub_s2, s1_sub_eff, viewer_global_thresh_cfg)
            df_sub_s2 = pd.DataFrame(df_sub_s2_list_dict)
            if not df_sub_s2.empty: df_sub_s2['SAMPLE_ID'] = "viewer_s2"; df_sub_s2['source_image_filename'] = fg_fn

            img_for_unsub_prep_s2 = img_fg_orig_uint8_s2.copy()
            if well_mask_uint8_s2 is not None:
                if viewer_global_source_illum == 'collimated':
                    pixels_in_well = img_fg_orig_uint8_s2[well_mask_uint8_s2 == 255]
                    if pixels_in_well.size > 0:
                        hist = cv2.calcHist([img_fg_orig_uint8_s2], [0], well_mask_uint8_s2, [256], [0,256])
                        dom_bg = np.argmax(hist); filled_out = np.full_like(img_fg_orig_uint8_s2, dom_bg); filled_out[well_mask_uint8_s2 == 255] = img_fg_orig_uint8_s2[well_mask_uint8_s2 == 255]
                        img_for_unsub_prep_s2 = filled_out
                    else: img_for_unsub_prep_s2 = apply_mask(img_fg_orig_uint8_s2.copy(), well_mask_uint8_s2) if well_mask_uint8_s2 is not None else img_fg_orig_uint8_s2.copy()
                else: img_for_unsub_prep_s2 = apply_mask(img_fg_orig_uint8_s2.copy(), well_mask_uint8_s2) if well_mask_uint8_s2 is not None else img_fg_orig_uint8_s2.copy()
            
            df_unsub_s2_list_dict, _, _ = _analyze_unsubtracted_locally(img_for_unsub_prep_s2, df_sub_s2_list_dict, s1_unsub_eff['local_otsu_roi_size'], s1_unsub_eff, viewer_global_thresh_cfg, viewer_global_source_illum)
            df_unsub_s2 = pd.DataFrame(df_unsub_s2_list_dict)
            if not df_unsub_s2.empty: df_unsub_s2['SAMPLE_ID'] = "viewer_s2"; df_unsub_s2['source_image_filename'] = fg_fn

            s1_particles_for_frame_df = _compare_and_fuse_particle_dfs(df_sub_s2, df_unsub_s2, s1_comp_eff)
            # Add a unique index for merging with S2 analysis results
            if not s1_particles_for_frame_df.empty:
                 s1_particles_for_frame_df = s1_particles_for_frame_df.reset_index().rename(columns={'index': 's1_viewer_fused_idx'})

            
            self.s2_status_text_var.set(f"S2: S1 found {len(s1_particles_for_frame_df)} p. Analyzing quality...")
            if s1_particles_for_frame_df.empty:
                self.ax_dynamic_vis.imshow(cv2.cvtColor(img_fg_orig_uint8_s2, cv2.COLOR_GRAY2BGR)); self.ax_dynamic_vis.set_axis_off()
                self.s2_final_sharp_particles_df_for_viewer = pd.DataFrame() 
            else:
                s2_intermediate_cropper = ParticleCropper(s2_eff_config['intermediate_crop'])
                s2_temp_dir = os.path.join(self.temp_dir.name, f"s2_viewer_crops_{fg_fn.replace('.png','').replace('.','_')}")
                os.makedirs(s2_temp_dir, exist_ok=True)
                s2_single_particle_results_list = []
                for _, s1_row in s1_particles_for_frame_df.iterrows(): # s1_row is now a row from the fused S1 df
                    s2_res_obj = _process_single_particle_for_quality(s1_row, img_fg_orig_color_s2, img_fg_orig_uint8_s2, s2_intermediate_cropper, s2_temp_dir, s2_eff_config, viewer_global_source_illum, viewer_global_thresh_cfg['mode'], viewer_global_thresh_cfg['otsu_threshold_multiplier'], viewer_global_thresh_cfg['fixed_threshold_value'])
                    # Add the s1_viewer_fused_idx to the result object's input data for merging
                    s2_res_obj.input_particle_data['s1_viewer_fused_idx'] = s1_row['s1_viewer_fused_idx']
                    s2_single_particle_results_list.append(s2_res_obj.to_dict())
                
                s2_detailed_df = pd.DataFrame(s2_single_particle_results_list)

                # Merge based on the 's1_viewer_fused_idx' (which corresponds to 'input_particle_index' in s2_detailed_df)
                s2_merged_for_score = pd.merge(s1_particles_for_frame_df, s2_detailed_df, 
                                                left_on='s1_viewer_fused_idx', right_on='input_particle_index', 
                                                how='left', suffixes=('', '_s2analysis'))
                
                s2_suitable_scored, _ = _filter_and_score_analysis_results(s2_merged_for_score, s2_eff_config)
                
                status_parts_s2 = [f"S1_Fused: {len(s1_particles_for_frame_df)}"]
                artifact_status = []
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
                            pass_mask = pd.to_numeric(self.s2_final_sharp_particles_df_for_viewer[metric_actual], errors='coerce').fillna(float('-inf')) > thresh_val
                            self.s2_final_sharp_particles_df_for_viewer = self.s2_final_sharp_particles_df_for_viewer[pass_mask]
                            status_parts_s2.append(f"SharpFOn({metric_actual}>{thresh_val:.2f}): {len(self.s2_final_sharp_particles_df_for_viewer)}")
                elif s2_eff_config['filters_enabled']['final_sharpness']: status_parts_s2.append(f"SharpFOn: 0")
                else: status_parts_s2.append(f"SharpFOff: {len(self.s2_final_sharp_particles_df_for_viewer)}")
                self.s2_status_text_var.set(" | ".join(status_parts_s2)); self.update_s2_sharp_particle_selector() 
                
                display_s2_final = cv2.cvtColor(img_fg_orig_uint8_s2, cv2.COLOR_GRAY2BGR)
                if not self.s2_final_sharp_particles_df_for_viewer.empty:
                    for _, p_row in self.s2_final_sharp_particles_df_for_viewer.iterrows():
                        try: x,y,w,h = map(int, [p_row['bounding_x'],p_row['bounding_y'],p_row['bounding_w'],p_row['bounding_h']]); cv2.rectangle(display_s2_final, (x,y), (x+w,y+h), (0,255,255), 2) 
                        except: pass 
                self.ax_dynamic_vis.imshow(display_s2_final)
                if well_mask_uint8_s2 is not None: 
                    h_im, w_im = well_mask_uint8_s2.shape; xc_s2, yc_s2 = np.meshgrid(np.arange(w_im), np.arange(h_im)); masked_out_areas_s2 = (well_mask_uint8_s2 == 0) 
                    self.ax_dynamic_vis.contourf(xc_s2, yc_s2, masked_out_areas_s2, levels=[0.5,1.5], colors='red', hatches=['//'], alpha=0.3)
                self.ax_dynamic_vis.set_title(f"S2 Filtered Output ({len(self.s2_final_sharp_particles_df_for_viewer)} particles)", fontsize=8, loc='center'); self.ax_dynamic_vis.set_axis_off()
                try:
                    if os.path.exists(s2_temp_dir): shutil.rmtree(s2_temp_dir)
                except Exception: pass

        elif mode in ["S3", "S4"]: 
            self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111) 
            s3_core_eff_params = params['s3_core_effective'] 
            self.ax_dynamic_vis.set_axis_off(); img_f = self.image_array_gray_float; title_b = os.path.basename(self.image_path or '')
            viewer_illum_s3s4 = s3_core_eff_params.get('source_image_illumination_type')
            viewer_thresh_s3s4 = s3_core_eff_params.get('thresholding_mode')
            if mode == "S3":
                self.ax_dynamic_vis.set_title(f"S3: {title_b} (Illum: {viewer_illum_s3s4}, Thresh: {viewer_thresh_s3s4})", fontsize=8)
                s3f,s3mp,s3eap = extract_ml_features(self.image_path, s3_core_eff_params, img_f) 
                if not (s3f or s3mp or s3eap): self.ax_dynamic_vis.imshow(img_f, cmap='gray'); self.ax_dynamic_vis.set_title("S3: Failed", fontsize=8)
                else:
                    tmp = os.path.join(self.temp_dir.name, "s3.png"); visualize_ml_features(img_f, s3mp, s3eap, tmp, s3_core_eff_params) 
                    if os.path.exists(tmp): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp))
                    else: self.ax_dynamic_vis.imshow(img_f, cmap='gray')
            elif mode == "S4":
                self.ax_dynamic_vis.set_title(f"S4: {title_b} (Illum: {viewer_illum_s3s4}, Thresh: {viewer_thresh_s3s4})", fontsize=8)
                _, s3mp_s4, _ = extract_ml_features(self.image_path, s3_core_eff_params, img_f) 
                s4rm, s4fm, _ = reconstruct_main_body_masks(img_f, s3_core_eff_params) 
                if not (s4rm is not None or s4fm is not None) : self.ax_dynamic_vis.imshow(img_f, cmap='gray'); self.ax_dynamic_vis.set_title("S4: Failed", fontsize=8)
                else:
                    tmp = os.path.join(self.temp_dir.name, "s4.png"); visualize_detailed_morphology(img_f, s3mp_s4, s4rm, s4fm, tmp)
                    if os.path.exists(tmp): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp))
                    else: self.ax_dynamic_vis.imshow(img_f, cmap='gray')
        self.canvas_dynamic_vis.draw()

    def update_s2_sharp_particle_selector(self):
        if not self.s2_final_sharp_particles_df_for_viewer.empty:
            df = self.s2_final_sharp_particles_df_for_viewer
            
            # Use 's1_viewer_fused_idx' if it exists from S1 sim, else 'input_particle_index'
            idx_col = 's1_viewer_fused_idx' if 's1_viewer_fused_idx' in df.columns else 'input_particle_index'
            if idx_col not in df.columns: df[idx_col] = df.index # Fallback if both missing

            df['X_display'] = df.get('X', pd.Series(np.nan, index=df.index))
            df['Y_display'] = df.get('Y', pd.Series(np.nan, index=df.index))

            particle_display_names = [
                f"P_S1fusedIdx{int(row.get(idx_col, row.name))}_X{int(row.get('X_display', -1))}_Y{int(row.get('Y_display', -1))}" 
                for _, row in df.iterrows()
            ]

            self.s2_sharp_particle_selector['values'] = particle_display_names
            if particle_display_names:
                self.s2_selected_sharp_particle_var.set(particle_display_names[0])
                self.s2_sharp_particle_selector.config(state="readonly"); self.on_s2_sharp_particle_selected() 
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
                # Parsing "P_S1fusedIdx<val>_X<val>_Y<val>"
                match = re.match(r"P_S1fusedIdx(\d+)_X(-?\d+)_Y(-?\d+)", selected_particle_str)
                if not match: raise ValueError(f"Could not parse S2 particle string: {selected_particle_str}")
                
                s1_fused_idx = int(match.group(1))
                idx_col_to_match = 's1_viewer_fused_idx' if 's1_viewer_fused_idx' in self.s2_final_sharp_particles_df_for_viewer.columns else 'input_particle_index'

                selected_rows = self.s2_final_sharp_particles_df_for_viewer[
                    self.s2_final_sharp_particles_df_for_viewer[idx_col_to_match].astype(int) == s1_fused_idx
                ]
                if selected_rows.empty: raise ValueError(f"No S2 particle data for S1 Fused Idx {s1_fused_idx}")
                particle_attrs = selected_rows.iloc[0]

                fg_fn = self.s1_selected_fg_filepath_var.get()
                original_frame_path = os.path.join(self.s1_image_directory_path, fg_fn)
                orig_color_img = core_load_image(original_frame_path, 'color'); orig_gray_img = convert_to_grayscale(orig_color_img)
                if orig_color_img is None or orig_gray_img is None: raise ValueError("Could not load original frame for S2 crop.")
                
                final_crop_size = self.param_vars['s2_final_crop_size'].get()
                final_cropper = ParticleCropper({'mode': 'centroid_square', 'square_size': final_crop_size})
                
                crop_centroid_x = particle_attrs.get('X') 
                crop_centroid_y = particle_attrs.get('Y')
                if pd.isna(crop_centroid_x) or pd.isna(crop_centroid_y): raise ValueError("Missing X/Y for S2 final crop in selected particle attrs.")

                particle_series_for_crop = pd.Series({'X': int(crop_centroid_x), 'Y': int(crop_centroid_y)})
                final_crop_gray, _ = final_cropper.crop_single_particle_from_image(orig_color_img, orig_gray_img, particle_series_for_crop, output_dir_for_crop=None)
                
                if final_crop_gray is not None:
                    final_crop_bgr = cv2.cvtColor(final_crop_gray, cv2.COLOR_GRAY2BGR)
                    viewer_illum_s2 = self.param_vars['viewer_source_illumination_type'].get() 
                    viewer_thresh_mode_s2 = self.param_vars['viewer_thresholding_mode'].get()
                    viewer_otsu_mult_s2 = self.param_vars['viewer_otsu_threshold_multiplier'].get()
                    viewer_fixed_val_s2 = self.param_vars['viewer_fixed_threshold_value'].get()
                    is_on_this_crop_boundary = detect_cropped_particle_on_boundary(final_crop_gray, f"viewer_fc_S1fusedIdx{s1_fused_idx}.png", viewer_illum_s2, viewer_thresh_mode_s2, viewer_otsu_mult_s2, viewer_fixed_val_s2)
                    self.ax_s2_single_crop.imshow(final_crop_bgr); self.ax_s2_single_crop.set_title(f"S2 Final Crop ({final_crop_gray.shape[1]}x{final_crop_gray.shape[0]})")
                    attr_text = f"S1 Frame: {fg_fn}\nS1 Fused Idx: {s1_fused_idx}\nGlobal Centroid: X={int(crop_centroid_x)}, Y={int(crop_centroid_y)}\n\n"
                    attr_text += f"S2 Intermed. Crop: {particle_attrs.get('cropped_filename', 'N/A')}\n  Black Mask: {particle_attrs.get('is_black_mask', 'N/A')}\n  On Intermed. Boundary: {particle_attrs.get('is_on_crop_boundary', 'N/A')}\n\n" 
                    attr_text += f"S2 Final Crop (this view):\n  Boundary Flag (this view): {is_on_this_crop_boundary}\n\nSharpness:\n"
                    for metric in AVAILABLE_SHARPNESS_METRICS: attr_text += f"  {metric}: {particle_attrs.get(metric, np.nan):.2f}\n  {metric}_norm: {particle_attrs.get(metric+'_norm', np.nan):.3f}\n"
                    attr_text += f"  combined_sharpness: {particle_attrs.get('combined_sharpness', np.nan):.3f}\n"
                    self.s2_sharp_particle_attributes_text.insert(tk.END, attr_text)
                else: self.ax_s2_single_crop.imshow(np.zeros((100,100)), cmap='gray'); self.s2_sharp_particle_attributes_text.insert(tk.END, "Error final cropping.")
            except Exception as e:
                self.ax_s2_single_crop.imshow(np.zeros((100,100)), cmap='gray'); self.s2_sharp_particle_attributes_text.insert(tk.END, f"Error: {e}")
        self.ax_s2_single_crop.set_axis_off(); self.canvas_s2_single_crop.draw(); self.s2_sharp_particle_attributes_text.config(state=tk.DISABLED)

def main():
    try: load_config(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml"))
    except Exception as e: messagebox.showerror("Startup Error", f"Failed to load config: {e}"); return
    root = tk.Tk(); FeatureViewerApp(root); root.mainloop()

if __name__ == '__main__':
    main()