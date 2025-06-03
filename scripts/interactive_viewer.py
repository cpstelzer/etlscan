# interactive_viewer.py (to be placed in well_plate_analyzer/scripts/)

import sys
import os
import pandas as pd

# Add the project root (well_plate_analyzer) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg') # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skimage.io
import skimage.util
import skimage.color
import numpy as np
import tempfile
import logging
import cv2 
from sklearn.neighbors import BallTree # For particle comparison

try:
    from src.utils.config_loader import get_config, load_config 
    from src.core.ml_feature_extractor import extract_ml_features
    from src.visualization.feature_visualization import visualize_ml_features
    from src.core.detailed_morphology_calculator import reconstruct_main_body_masks 
    from src.visualization.stage_04_visualization import visualize_detailed_morphology 
    from src.utils.common_exceptions import ConfigError, DataValidationError 
    # Import S1 specific analysis functions
    from src.stages.stage_01_initial_detection import (
        _parse_image_filenames, 
        _analyze_single_subtracted_image, 
        _filter_image_df,
        _analyze_unsubtracted_fg_image, 
        _compare_and_fuse_particle_dfs 
    )
    from src.core.image_io_operations import load_image as core_load_image 
    from src.core.image_enhancement import subtract_images, apply_blur, apply_threshold
    from src.core.masking_operations import mask_well_area, apply_mask

except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
UPDATE_DEBOUNCE_MS = 300 
BG_OFFSET_MIN, BG_OFFSET_MAX = 1, 500 
DEFAULT_INT_MIN, DEFAULT_INT_MAX = 0, 10000
DEFAULT_FLOAT_MIN, DEFAULT_FLOAT_MAX = 0.0, 100.0

class FeatureViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Viewer (S1/S3/S4 Toggle)"); master.geometry("1450x900")
        self.image_path = None; self.image_array_gray_float = None 
        self.default_params = {}; self.current_params = {} 
        self.param_vars = {} 
        self.param_str_vars = {} 
        self.param_specs = {} 
        self.param_widgets = {} 
        self.update_job_id = None; self.active_visualization_mode = tk.StringVar(value="S1") # Default to S1
        self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame()
        self.s1_selected_fg_filepath_var = tk.StringVar()
        self.s1_convolution_kernel_list = None; self.s1_raw_config = {} 
        self.s1_system_config_from_file = {}
        self.s1_bg_offset_val_str = tk.StringVar() 

        try:
            load_config(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml")); cfg = get_config()
            s3_ml_cfg = cfg.get('stage_03_ml_classification', {}); s3_feat_cfg = s3_ml_cfg.get('feature_extraction_params', {})
            if not s3_feat_cfg: messagebox.showwarning("Config Warning", "S3 params missing.")
            
            self.s1_raw_config = cfg.get('stage_01_initial_detection', {}); self.s1_system_config_from_file = cfg.get('system', {})
            s1_particle_cfg = self.s1_raw_config.get('particle_analysis', {}) 
            s1_unsub_particle_cfg = self.s1_raw_config.get('unsubtracted_particle_analysis', {}) 
            s1_comp_cfg = self.s1_raw_config.get('particle_comparison', {}) 
            s1_well_mask_cfg = self.s1_raw_config.get('well_masking', {})
            
            if not s1_particle_cfg: messagebox.showwarning("Config Warning", "S1 particle_analysis params missing.")
            if not s1_unsub_particle_cfg: messagebox.showwarning("Config Warning", "S1 unsubtracted_particle_analysis params missing.")
            if not s1_comp_cfg: messagebox.showwarning("Config Warning", "S1 particle_comparison params missing.")
            if not s1_well_mask_cfg: messagebox.showwarning("Config Warning", "S1 well mask params missing.")
            
            self.s1_convolution_kernel_list = s1_particle_cfg.get('convolution_kernel')
            self.default_params = {f"s3_{k}": v for k, v in s3_feat_cfg.items()}
            
            self.default_params.update({
                's1_bg_offset_val': self.s1_raw_config.get('bg_offset_val', 110),
                's1_sub_illumination': s1_particle_cfg.get('illumination', 'darkfield'), 
                's1_sub_apply_convolution': s1_particle_cfg.get('apply_convolution', True),
                's1_sub_microorganism_threshold': s1_particle_cfg.get('microorganism_threshold', 25),
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
                's1_unsub_illumination': s1_unsub_particle_cfg.get('illumination', 'collimated'),
                's1_unsub_blur_kernel_size': s1_unsub_particle_cfg.get('blur_kernel_size', 3),
                's1_unsub_thresholding_mode': s1_unsub_particle_cfg.get('thresholding_mode', 'otsu'),
                's1_unsub_fixed_threshold_value': s1_unsub_particle_cfg.get('fixed_threshold_value', 70),
                's1_unsub_min_particle_area': s1_unsub_particle_cfg.get('min_particle_area', 300),
            })
            self.default_params.update({
                's1_comp_search_radius_factor': s1_comp_cfg.get('search_radius_factor', 2.0),
                's1_comp_default_particle_area': s1_comp_cfg.get('default_particle_area', 500),
            })

            self.current_params = self.default_params.copy()
            self._initialize_param_vars_and_specs()

        except Exception as e: logger.exception("Config Error"); messagebox.showerror("Config Error", f"{e}"); master.destroy(); return
        self.temp_dir = tempfile.TemporaryDirectory(); master.protocol("WM_DELETE_WINDOW", self.on_closing); self.setup_gui(); self.on_mode_change()

    def _initialize_param_vars_and_specs(self):
        s1_specs_map = self._get_s1_param_specs(); s3_specs_map = self._get_s3_param_specs()
        all_specs = {**s1_specs_map, **s3_specs_map} 
        for name, default_val in self.default_params.items():
            spec = all_specs.get(name)
            if spec:
                self.param_specs[name] = spec 
                param_type = spec['type']
                if param_type == 'int':
                    self.param_vars[name] = tk.IntVar(value=int(default_val))
                    self.param_str_vars[name] = tk.StringVar(value=str(int(default_val)))
                    self.param_vars[name].trace_add("write", lambda n,i,m,var_name=name: self._sync_num_to_str(var_name))
                elif param_type == 'float':
                    self.param_vars[name] = tk.DoubleVar(value=float(default_val))
                    res = spec.get('res', 0.1); fmt = f".{max(1, str(res).split('.')[-1].__len__())}f"
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
        logger.info("Closing app...");
        if self.update_job_id: self.master.after_cancel(self.update_job_id)
        for canvas_attr in ['canvas_raw', 'canvas_dynamic_vis']:
            if hasattr(self, canvas_attr) and getattr(self, canvas_attr):
                try: getattr(self, canvas_attr).get_tk_widget().destroy()
                except tk.TclError: pass
        plt.close('all') 
        try: self.temp_dir.cleanup(); logger.info("Temp dir cleaned.")
        except Exception as e: logger.error(f"Temp dir cleanup error: {e}")
        self.master.destroy(); logger.info("App closed.")

    def setup_gui(self): 
        main_frame = ttk.Frame(self.master, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        top_controls_frame = ttk.Frame(main_frame); top_controls_frame.pack(fill=tk.X, pady=5)
        self.load_button = ttk.Button(top_controls_frame, text="Load", command=self.dispatch_load_action); self.load_button.pack(side=tk.LEFT, padx=5)
        self.s1_fg_image_selector_label = ttk.Label(top_controls_frame, text="FG (S1):")
        self.s1_fg_image_selector = ttk.Combobox(top_controls_frame, textvariable=self.s1_selected_fg_filepath_var, state="disabled", width=50)
        self.s1_fg_image_selector.bind("<<ComboboxSelected>>", self.on_s1_fg_image_selected)
        self.status_label = ttk.Label(top_controls_frame, text="Load."); self.status_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        toggle_frame = ttk.Frame(top_controls_frame); toggle_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(toggle_frame, text="Show:").pack(side=tk.LEFT)
        for mode_disp, mode_val in [("S1", "S1"), ("S3", "S3"), ("S4", "S4")]: 
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
        
        self.s1_params_frame_main = ttk.Frame(self.controls_frame_inner) 
        
        self.s1_subtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Subtracted Analysis Params", padding=5)
        self.create_s1_subtracted_param_controls(self.s1_subtracted_params_frame)
        
        self.s1_wellmask_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Well Masking Params", padding=5)
        self.create_s1_wellmask_param_controls(self.s1_wellmask_params_frame)

        self.s1_unsubtracted_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Unsubtracted FG Analysis Params", padding=5)
        self.create_s1_unsubtracted_param_controls(self.s1_unsubtracted_params_frame)

        self.s1_comparison_params_frame = ttk.Labelframe(self.s1_params_frame_main, text="S1 Particle Comparison Params", padding=5)
        self.create_s1_comparison_param_controls(self.s1_comparison_params_frame)

        self.s3_params_frame = ttk.Frame(self.controls_frame_inner); ttk.Label(self.s3_params_frame, text="S3 ML Params", font='Helvetica 10 bold').pack(fill=tk.X, pady=(5,2)); self.create_s3_param_controls(self.s3_params_frame)
        
        param_buttons = ttk.Frame(self.controls_frame_inner); param_buttons.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=5)
        ttk.Button(param_buttons, text="Reset", command=self.reset_parameters).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(param_buttons, text="Update", command=self.process_and_display_safe).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        plots_container = ttk.Frame(paned_window); paned_window.add(plots_container, weight=3) 
        self.fig_raw, self.ax_raw = plt.subplots(); self.ax_raw.set_title("Raw Input"); self.ax_raw.set_axis_off()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=plots_container); self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_raw.draw()
        self.fig_dynamic_vis = plt.figure(); self.ax_dynamic_vis = None 
        self.canvas_dynamic_vis = FigureCanvasTkAgg(self.fig_dynamic_vis, master=plots_container); self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2); self.canvas_dynamic_vis.draw()
    
    def on_mode_change(self):
        mode = self.active_visualization_mode.get()
        self.s1_params_frame_main.pack_forget()
        self.s3_params_frame.pack_forget()
        
        self.s1_fg_image_selector_label.pack_forget(); self.s1_fg_image_selector.pack_forget()

        if mode == "S1":
            self.s1_params_frame_main.pack(fill=tk.X, expand=True)
            self.s1_subtracted_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_wellmask_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_unsubtracted_params_frame.pack(fill=tk.X, expand=True, pady=3)
            self.s1_comparison_params_frame.pack(fill=tk.X, expand=True, pady=3)

            self.load_button.config(text="Load S1 Dir")
            self.s1_fg_image_selector_label.pack(side=tk.LEFT, padx=(10,2), before=self.status_label)
            self.s1_fg_image_selector.pack(side=tk.LEFT, padx=2, before=self.status_label)
            self.s1_fg_image_selector.config(state="readonly" if self.s1_image_directory_path and not self.s1_image_file_list_df.empty else "disabled")
            self.update_s1_unsub_threshold_param_visibility() 
        elif mode in ["S3", "S4"]:
            self.s3_params_frame.pack(fill=tk.X, expand=True); self.load_button.config(text="Load Img (S3/S4)")
            if mode == "S4": self.update_s3_threshold_param_visibility() 
        
        if mode != "S1": 
            self.s1_image_directory_path = None; self.s1_image_file_list_df = pd.DataFrame(); self.s1_fg_image_selector['values'] = []; self.s1_selected_fg_filepath_var.set("")
        
        self.update_raw_image_display(); self.schedule_update()


    def on_s1_fg_image_selected(self, event=None): self.update_raw_image_display(); self.schedule_update()
    def _bind_mousewheel(self, event, canvas): canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))
    def _unbind_mousewheel(self, event, canvas): canvas.unbind_all("<MouseWheel>")
    def _on_mousewheel(self, event, canvas): canvas.yview_scroll(1 if (event.num == 5 or event.delta < 0) else -1, "units")
    def schedule_update(self):
        if self.update_job_id: self.master.after_cancel(self.update_job_id)
        self.update_job_id = self.master.after(UPDATE_DEBOUNCE_MS, self.process_and_display_safe)

    def update_s3_threshold_param_visibility(self, *args): 
        if self.active_visualization_mode.get() not in ["S3", "S4"]: return 
        mode_var = self.param_vars.get('s3_thresholding_mode')
        if not mode_var: return
        mode = mode_var.get()
        
        otsu_widget_name = 's3_otsu_threshold_multiplier'
        fixed_widget_name = 's3_fixed_threshold_value'

        if otsu_widget_name in self.param_widgets:
            if mode == 'otsu': self.param_widgets[otsu_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[otsu_widget_name].pack_forget()
        
        if fixed_widget_name in self.param_widgets:
            if mode == 'fixed': self.param_widgets[fixed_widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[fixed_widget_name].pack_forget()

        if self.active_visualization_mode.get() in ["S3", "S4"]: self.schedule_update()


    def update_s1_unsub_threshold_param_visibility(self, *args):
        if self.active_visualization_mode.get() != "S1": return
        mode_var = self.param_vars.get('s1_unsub_thresholding_mode')
        if not mode_var: return 
        
        mode = mode_var.get()
        widget_name = 's1_unsub_fixed_threshold_value'
        
        if widget_name in self.param_widgets:
            if mode == 'fixed': self.param_widgets[widget_name].pack(fill=tk.X, pady=2, padx=5)
            else: self.param_widgets[widget_name].pack_forget()
        if self.active_visualization_mode.get() == "S1": self.schedule_update()
    
    def _adjust_param_value(self, param_name, delta):
        num_var, spec = self.param_vars.get(param_name), self.param_specs.get(param_name)
        if not num_var or not spec: return
        current_val, param_type = num_var.get(), spec['type']
        step = spec.get('step', 1 if param_type == 'int' else spec.get('res', 0.1))
        new_val = current_val + (delta * step)
        min_val, max_val = spec.get('min', DEFAULT_INT_MIN if param_type == 'int' else DEFAULT_FLOAT_MIN), spec.get('max', DEFAULT_INT_MAX if param_type == 'int' else DEFAULT_FLOAT_MAX)
        new_val = max(min_val, min(new_val, max_val))
        if param_type == 'int' and spec.get('step') == 2 and int(new_val) % 2 == 0: 
             new_val_candidate_down = int(new_val) - 1
             new_val_candidate_up = int(new_val) + 1
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
                clamped_val_candidate_down = int(clamped_val) - 1
                clamped_val_candidate_up = int(clamped_val) + 1
                if clamped_val_candidate_down >= min_val: clamped_val = clamped_val_candidate_down
                elif clamped_val_candidate_up <= max_val: clamped_val = clamped_val_candidate_up
                clamped_val = max(min_val, min(clamped_val, max_val))
            num_var.set(clamped_val); self.schedule_update()
        except ValueError: str_var.set(str(num_var.get())) 
        self.master.focus()

    def _create_generic_param_control(self, parent_frame, name, spec, current_val_for_var):
        param_type = spec['type']
        container = ttk.Frame(parent_frame)
        
        label_text_raw = name.replace('_', ' ').title()
        if name.startswith("s1_sub_"): label_text = label_text_raw[7:]
        elif name.startswith("s1_unsub_"): label_text = label_text_raw[9:]
        elif name.startswith("s1_comp_"): label_text = label_text_raw[8:]
        elif name.startswith("s1_well_mask_"): label_text = label_text_raw[13:]
        elif name.startswith("s3_"): label_text = label_text_raw[3:]
        else: label_text = label_text_raw 

        ttk.Label(container, text=label_text + ":", width=25, anchor="w").pack(side=tk.LEFT, padx=(0,5))
        
        tk_var = self.param_vars.get(name) 
        
        if param_type == 'choice':
            cmd = None
            if name == 's3_thresholding_mode': cmd = self.update_s3_threshold_param_visibility
            elif name == 's1_unsub_thresholding_mode': cmd = self.update_s1_unsub_threshold_param_visibility
            else: cmd = self.schedule_update
            ttk.OptionMenu(container, tk_var, current_val_for_var, *spec['options'], command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X)
        elif param_type in ['int', 'float']:
            str_tk_var = self.param_str_vars.get(name)
            ttk.Button(container, text="-", command=lambda p_name=name: self._adjust_param_value(p_name, -1), width=2).pack(side=tk.LEFT)
            entry = ttk.Entry(container, textvariable=str_tk_var, width=7, justify='center')
            entry.pack(side=tk.LEFT, padx=2); entry.bind("<Return>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e)); entry.bind("<FocusOut>", lambda e, p_name=name: self._on_param_entry_commit(p_name, e))
            ttk.Button(container, text="+", command=lambda p_name=name: self._adjust_param_value(p_name, 1), width=2).pack(side=tk.LEFT)
        elif param_type == 'bool': ttk.Checkbutton(container, variable=tk_var, command=self.schedule_update).pack(side=tk.LEFT)

        if not (name in ['s3_otsu_threshold_multiplier', 's3_fixed_threshold_value', 's1_unsub_fixed_threshold_value']):
            container.pack(fill=tk.X, pady=2, padx=5)
        
        self.param_widgets[name] = container 
    
    def _get_s1_param_specs(self): 
        s1_sub_specs = {
            's1_bg_offset_val': {'min': BG_OFFSET_MIN, 'max': BG_OFFSET_MAX, 'type': 'int', 'step': 1},
            's1_sub_illumination': {'options': ['darkfield', 'collimated'], 'type': 'choice'},
            's1_sub_apply_convolution': {'type': 'bool'}, 
            's1_sub_microorganism_threshold': {'min': 0, 'max': 255, 'type': 'int', 'step': 1},
            's1_sub_min_microorganism_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10}, 
            's1_sub_microorganism_blur_kernel': {'min': 1, 'max': 15, 'type': 'int', 'step': 2}, 
        }
        s1_wm_specs = {
            's1_well_mask_create_mask': {'type': 'bool'}, 
            's1_well_mask_auto_center_mask': {'type': 'bool'},
            's1_well_mask_radius_px': {'min': 100, 'max': 4000, 'type': 'int', 'step': 50}, 
            's1_well_mask_radius_fraction': {'min': 0.1, 'max': 1.0, 'type': 'float', 'res': 0.01},
        }
        s1_unsub_specs = {
            's1_unsub_illumination': {'options': ['darkfield', 'collimated'], 'type': 'choice'},
            's1_unsub_blur_kernel_size': {'min': 0, 'max': 15, 'type': 'int', 'step': 2}, 
            's1_unsub_thresholding_mode': {'options': ['otsu', 'fixed'], 'type': 'choice'},
            's1_unsub_fixed_threshold_value': {'min': 0, 'max': 255, 'type': 'int', 'step': 1},
            's1_unsub_min_particle_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10},
        }
        s1_comp_specs = {
            's1_comp_search_radius_factor': {'min': 0.1, 'max': 10.0, 'type': 'float', 'res': 0.1},
            's1_comp_default_particle_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10},
        }
        return {**s1_sub_specs, **s1_wm_specs, **s1_unsub_specs, **s1_comp_specs}


    def _get_s3_param_specs(self): 
        return {'s3_illumination': {'options': ['darkfield', 'collimated'], 'type': 'choice'}, 's3_gaussian_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float', 'res': 0.1},
                's3_thresholding_mode': {'options': ['otsu', 'fixed'], 'type': 'choice'}, 's3_otsu_threshold_multiplier': {'min': 0.1, 'max': 2.0, 'type': 'float', 'res': 0.05}, 
                's3_fixed_threshold_value': {'min': 0, 'max': 255, 'type': 'int', 'step': 1}, 's3_closing_selem_size': {'min': 0, 'max': 10, 'type': 'int', 'step': 1},
                's3_min_noise_size': {'min': 1, 'max': 1000, 'type': 'int', 'step': 10}, 's3_min_significant_area': {'min': 10, 'max': 2000, 'type': 'int', 'step': 10},
                's3_egg_posterior_pixel_fraction': {'min': 0.05, 'max': 0.95, 'type': 'float', 'res': 0.01}, 's3_egg_threshold_quantile': {'min': 0.01, 'max': 0.99, 'type': 'float', 'res': 0.01},
                's3_egg_min_area': {'min': 10, 'max': 5000, 'type': 'int', 'step': 10}, 's3_egg_max_area': {'min': 50, 'max': 20000, 'type': 'int', 'step': 100},
                's3_egg_min_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 's3_egg_max_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
                's3_egg_min_eccentricity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01}, 's3_orientation_diff_tolerance_deg': {'min': 0, 'max': 90, 'type': 'int', 'step': 1} }

    def create_s1_subtracted_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = [
            's1_bg_offset_val', 's1_sub_illumination', 's1_sub_apply_convolution', 
            's1_sub_microorganism_threshold', 's1_sub_min_microorganism_area', 
            's1_sub_microorganism_blur_kernel'
        ]
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))
    
    def create_s1_wellmask_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = [
            's1_well_mask_create_mask', 's1_well_mask_auto_center_mask',
            's1_well_mask_radius_px', 's1_well_mask_radius_fraction'
        ]
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s1_unsubtracted_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = [
            's1_unsub_illumination', 's1_unsub_blur_kernel_size', 
            's1_unsub_thresholding_mode', 's1_unsub_fixed_threshold_value', 
            's1_unsub_min_particle_area'
        ]
        for name in ordered_params:
            if name in self.default_params: 
                self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))
                
    def create_s1_comparison_param_controls(self, parent_frame):
        s1_specs = self._get_s1_param_specs()
        ordered_params = [
            's1_comp_search_radius_factor', 's1_comp_default_particle_area'
        ]
        for name in ordered_params:
            if name in self.default_params: self._create_generic_param_control(parent_frame, name, s1_specs[name], self.current_params.get(name, self.default_params[name]))

    def create_s3_param_controls(self, parent_frame): 
        s3_specs = self._get_s3_param_specs()
        ordered_s3 = ['s3_illumination', 's3_thresholding_mode'] + [p for p in s3_specs if p not in ['s3_illumination', 's3_thresholding_mode']]
        for name in ordered_s3:
            if name in self.default_params:
                self._create_generic_param_control(parent_frame, name, s3_specs[name], self.current_params.get(name, self.default_params[name]))
                
    def reset_parameters(self): 
        logger.info("Resetting parameters.");
        if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
        for name, default_val in self.default_params.items():
            if name in self.param_vars: self.param_vars[name].set(default_val)
        self.on_mode_change() 

    def dispatch_load_action(self):
        if self.active_visualization_mode.get() == "S1": self._load_s1_directory()
        else: self._load_s3_s4_image()

    def _load_s1_directory(self): 
        dir_path = filedialog.askdirectory(title="Select S1 Image Directory")
        if not dir_path: self.status_label.config(text="S1 dir selection cancelled."); self.master.after(10, self.on_mode_change); return
        self.s1_image_directory_path = dir_path
        self.status_label.config(text=f"Loading S1 dir: {os.path.basename(dir_path)}..."); self.master.update_idletasks()
        try:
            parsed_df = _parse_image_filenames(dir_path);
            if parsed_df is None or parsed_df.empty: raise ValueError("No valid images.")
            self.s1_image_file_list_df = _filter_image_df(parsed_df); self.s1_image_file_list_df.reset_index(drop=True, inplace=True)
            if self.s1_image_file_list_df.empty: raise ValueError("No images after filtering.")
            filenames = self.s1_image_file_list_df['filename'].tolist()
            self.s1_fg_image_selector['values'] = filenames
            if filenames: self.s1_selected_fg_filepath_var.set(filenames[0])
            logger.info(f"S1 Dir loaded: {len(self.s1_image_file_list_df)} images."); self.status_label.config(text=f"S1 Dir: {os.path.basename(dir_path)}. Select FG.")
        except Exception as e:
            logger.exception(f"Fail S1 dir: {dir_path}"); messagebox.showerror("S1 Load Error", f"{e}")
            self.status_label.config(text="Fail S1 load."); self.s1_image_directory_path=None; self.s1_image_file_list_df=pd.DataFrame(); self.s1_fg_image_selector['values']=[]; self.s1_selected_fg_filepath_var.set("")
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
            logger.info(f"Img loaded: {path}"); self.status_label.config(text=f"Loaded: {os.path.basename(path)}.")
        except Exception as e:
            logger.exception(f"Fail img load: {path}"); messagebox.showerror("Img Load Error", f"{e}")
            self.status_label.config(text="Fail img load."); self.image_path=None; self.image_array_gray_float=None
        finally: 
            self.master.after(10, self.on_mode_change)
            if self.image_path: self.schedule_update()

    def update_raw_image_display(self): 
        self.ax_raw.clear(); mode = self.active_visualization_mode.get(); title = "Raw Input"; img_show = None
        if mode == "S1" and self.s1_image_directory_path and not self.s1_image_file_list_df.empty:
            fn = self.s1_selected_fg_filepath_var.get()
            if fn:
                path = os.path.join(self.s1_image_directory_path, fn)
                if os.path.exists(path): img_show_raw_uint8 = core_load_image(path, 'gray'); img_show = skimage.util.img_as_float(img_show_raw_uint8); title = f"S1 FG: {fn}"
                else: title = "S1 FG: Not found"
            else: title = "S1: Select FG"
        elif self.image_array_gray_float is not None: img_show = self.image_array_gray_float; title = f"Input: {os.path.basename(self.image_path or 'No Img')}"
        
        if img_show is not None: self.ax_raw.imshow(img_show, cmap='gray', vmin=0.0, vmax=1.0)
        else: self.ax_raw.imshow(np.zeros((100,100)), cmap='gray') 

        self.ax_raw.set_title(title, fontsize=8); self.ax_raw.set_axis_off(); self.canvas_raw.draw()

    def collect_current_params(self):
        gui_params = {name: var.get() for name, var in self.param_vars.items()}
        
        self.current_params['s1_subtracted_effective'] = {
            k.replace('s1_sub_', ''):v for k,v in gui_params.items() if k.startswith('s1_sub_')
        }
        self.current_params['s1_subtracted_effective']['convolution_kernel'] = self.s1_convolution_kernel_list
        self.current_params['s1_bg_offset_effective'] = gui_params.get('s1_bg_offset_val', self.default_params['s1_bg_offset_val'])

        self.current_params['s1_well_mask_effective'] = {
            k.replace('s1_well_mask_', ''):v for k,v in gui_params.items() if k.startswith('s1_well_mask_')
        }
        
        self.current_params['s1_unsubtracted_effective'] = {
            k.replace('s1_unsub_', ''):v for k,v in gui_params.items() if k.startswith('s1_unsub_')
        }
        
        self.current_params['s1_comparison_effective'] = {
            k.replace('s1_comp_', ''):v for k,v in gui_params.items() if k.startswith('s1_comp_')
        }
        
        self.current_params['s3_core_effective'] = {
            k.replace('s3_', ''):v for k,v in gui_params.items() if k.startswith('s3_')
        }
        return self.current_params


    def process_and_display_safe(self): 
        active_mode, valid_input = self.active_visualization_mode.get(), False
        if active_mode == "S1":
            if self.s1_image_directory_path and not self.s1_image_file_list_df.empty and self.s1_selected_fg_filepath_var.get(): valid_input = True
            else: self.status_label.config(text="S1: Load dir & select FG.")
        elif self.image_array_gray_float is not None: valid_input = True
        else: self.status_label.config(text="S3/S4: Load image.")
        
        if not valid_input:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None
            self.fig_dynamic_vis.clear(); self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111); self.ax_dynamic_vis.set_axis_off(); self.canvas_dynamic_vis.draw()
            return

        self.status_label.config(text="Processing..."); self.master.update_idletasks()
        try: self.process_and_display(); self.status_label.config(text=f"Ready. Mode: {active_mode}.")
        except DataValidationError as dve: logger.error(f"Data Validation Error during S1 processing: {dve}"); messagebox.showerror("S1 Data Error", f"{dve}"); self.status_label.config(text=f"S1 Data Error: {dve}")
        except Exception as e: logger.exception("Proc Error"); messagebox.showerror("Error", f"{e}"); self.status_label.config(text=f"Error: {e}")
        finally:
            if self.update_job_id: self.master.after_cancel(self.update_job_id); self.update_job_id = None

    def process_and_display(self): 
        params = self.collect_current_params()
        s1_sub_eff = params['s1_subtracted_effective']
        s1_wm_eff = params['s1_well_mask_effective']
        s1_unsub_eff = params['s1_unsubtracted_effective']
        s1_comp_eff = params['s1_comparison_effective']
        s1_bg_off = params['s1_bg_offset_effective']
        s3_eff = params['s3_core_effective']
        
        mode = self.active_visualization_mode.get(); self.fig_dynamic_vis.clear() 

        if mode == "S1":
            self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111) 
            title_dir = os.path.basename(self.s1_image_directory_path or "No Dir")
            self.fig_dynamic_vis.suptitle(f"S1 Fused Output: {title_dir}", fontsize=10)
            
            fg_fn = self.s1_selected_fg_filepath_var.get()
            fg_row = self.s1_image_file_list_df[self.s1_image_file_list_df['filename'] == fg_fn].iloc[0]
            fg_idx = fg_row.name 
            num_f = len(self.s1_image_file_list_df); bg_idx = (fg_idx + s1_bg_off) % num_f
            if bg_idx < 0: bg_idx += num_f
            bg_row = self.s1_image_file_list_df.iloc[bg_idx]
            
            img_fg_orig_uint8 = core_load_image(fg_row['full_path'],'gray')
            img_bg_orig_uint8 = core_load_image(bg_row['full_path'],'gray')
            if img_fg_orig_uint8 is None or img_bg_orig_uint8 is None: raise ValueError("S1 FG/BG image load fail for viewer.")

            well_mask_uint8 = None 
            if s1_wm_eff.get('create_mask', False):
                _, well_mask_uint8 = mask_well_area(img_fg_orig_uint8.copy(), s1_wm_eff, self.s1_system_config_from_file)
                if well_mask_uint8 is None:
                    logger.warning("S1 Viewer: Well mask creation failed. Proceeding without mask.")

            # --- Subtracted Analysis (df_ref) ---
            img_fg_sub_proc = img_fg_orig_uint8.copy()
            img_bg_sub_proc = img_bg_orig_uint8.copy()
            if s1_sub_eff.get('illumination') == 'collimated':
                img_fg_sub_proc = cv2.bitwise_not(img_fg_sub_proc)
                img_bg_sub_proc = cv2.bitwise_not(img_bg_sub_proc)
            
            subtracted_img = subtract_images(img_fg_sub_proc, img_bg_sub_proc)
            if subtracted_img is None: raise ValueError("S1 Subtraction fail for viewer.")

            img_for_sub_analysis = subtracted_img
            if well_mask_uint8 is not None: 
                img_for_sub_analysis = apply_mask(subtracted_img, well_mask_uint8)
            
            df_ref, _, _ = _analyze_single_subtracted_image(img_for_sub_analysis, s1_sub_eff)
            if df_ref is None: df_ref = pd.DataFrame() 

            # --- Unsubtracted Analysis (df_query) ---
            # The viewer prepares the input for _analyze_unsubtracted_fg_image
            # This mimics the preparation done in the main run_stage_01_initial_detection
            img_for_unsub_analysis_prepared = img_fg_orig_uint8.copy()
            if well_mask_uint8 is not None:
                if s1_unsub_eff.get('illumination') == 'collimated':
                    img_filled_outside_white = np.full_like(img_fg_orig_uint8, 255)
                    img_for_unsub_analysis_prepared = cv2.bitwise_and(img_fg_orig_uint8, img_fg_orig_uint8, mask=well_mask_uint8) + \
                                                      cv2.bitwise_and(img_filled_outside_white, img_filled_outside_white, mask=cv2.bitwise_not(well_mask_uint8))
                else: 
                    img_for_unsub_analysis_prepared = apply_mask(img_fg_orig_uint8.copy(), well_mask_uint8)
            
            df_query, _, binary_query_img_for_debug = _analyze_unsubtracted_fg_image(img_for_unsub_analysis_prepared, s1_unsub_eff)
            if df_query is None: df_query = pd.DataFrame()
            
            # --- Compare and Fuse ---
            df_fused = _compare_and_fuse_particle_dfs(df_ref, df_query, s1_comp_eff)
            
            # --- DEBUGGING PRINTS ---
            # (Keep the existing debug prints here)
            print("\n--- S1 VIEWER DEBUG ---")
            print(f"FG: {fg_fn}, BG: {bg_row['filename']}, BG Offset: {s1_bg_off}")
            print(f"Subtracted Analysis Illumination: {s1_sub_eff.get('illumination')}, Threshold: {s1_sub_eff.get('microorganism_threshold')}, Min Area: {s1_sub_eff.get('min_microorganism_area')}")
            print(f"Unsubtracted Analysis Illumination: {s1_unsub_eff.get('illumination')}, Blur: {s1_unsub_eff.get('blur_kernel_size')}, Threshold Mode: {s1_unsub_eff.get('thresholding_mode')}, Fixed Thresh: {s1_unsub_eff.get('fixed_threshold_value')}, Min Area: {s1_unsub_eff.get('min_particle_area')}")
            print(f"Comparison Search Radius Factor: {s1_comp_eff.get('search_radius_factor')}, Default Area: {s1_comp_eff.get('default_particle_area')}")
            
            print("\n--- df_ref (Subtracted Particles) ---")
            if df_ref.empty: print("DataFrame is empty.")
            else: print(df_ref.to_string(max_rows=10))

            print("\n--- df_query (Unsubtracted FG Particles) ---")
            if df_query.empty: print("DataFrame is empty.")
            else: print(df_query.to_string(max_rows=10))
            if binary_query_img_for_debug is not None:
                debug_unsub_bin_path = os.path.join(self.temp_dir.name, "debug_s1_unsub_binary_viewer.png")
                cv2.imwrite(debug_unsub_bin_path, binary_query_img_for_debug)
                print(f"DEBUG: Saved binary image from unsubtracted analysis to: {debug_unsub_bin_path}")

            print("\n--- df_fused (Fused Particles after Comparison) ---")
            if df_fused.empty: print("DataFrame is empty.")
            else: print(df_fused.to_string(max_rows=10))
            print("--- END S1 VIEWER DEBUG ---\n")
            # --- END DEBUGGING PRINTS ---

            # --- Visualization ---
            display_img_s1_final = cv2.cvtColor(img_fg_orig_uint8, cv2.COLOR_GRAY2BGR)
            particle_count_fused = 0
            if not df_fused.empty:
                particle_count_fused = len(df_fused)
                for _, p_row in df_fused.iterrows():
                    try: 
                        x,y,w,h = map(int, [p_row['bounding_x'],p_row['bounding_y'],p_row['bounding_w'],p_row['bounding_h']])
                        cv2.rectangle(display_img_s1_final, (x,y), (x+w,y+h), (0,255,255), 2) 
                    except (KeyError, TypeError, ValueError) as e_draw: 
                        logger.warning(f"S1 Viewer: Error drawing bbox for fused particle: {e_draw}")
            
            self.ax_dynamic_vis.imshow(display_img_s1_final)
            if well_mask_uint8 is not None:
                h_im, w_im = well_mask_uint8.shape; xc, yc = np.meshgrid(np.arange(w_im), np.arange(h_im))
                masked_out_areas = (well_mask_uint8 == 0) 
                self.ax_dynamic_vis.contourf(xc, yc, masked_out_areas, levels=[0.5,1.5], colors='red', hatches=['//'], alpha=0.3)
            
            title_s1 = f"FG: {fg_fn}\nBG: {bg_row['filename']} (Offset: {s1_bg_off})\nFused Particles: {particle_count_fused}"
            self.ax_dynamic_vis.set_title(title_s1, fontsize=8, loc='center'); self.ax_dynamic_vis.set_axis_off()

        elif mode in ["S3", "S4"]: 
            self.ax_dynamic_vis = self.fig_dynamic_vis.add_subplot(111); self.ax_dynamic_vis.set_axis_off()
            img_f = self.image_array_gray_float; title_b = os.path.basename(self.image_path or '')
            if mode == "S3":
                self.ax_dynamic_vis.set_title(f"S3: {title_b}", fontsize=8)
                s3f,s3mp,s3eap = extract_ml_features(self.image_path, s3_eff, img_f)
                if not (s3f or s3mp or s3eap): self.ax_dynamic_vis.imshow(img_f, cmap='gray'); self.ax_dynamic_vis.set_title("S3: Failed", fontsize=8)
                else:
                    tmp = os.path.join(self.temp_dir.name, "s3.png"); visualize_ml_features(img_f, s3mp, s3eap, tmp, s3_eff)
                    if os.path.exists(tmp): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp))
                    else: self.ax_dynamic_vis.imshow(img_f, cmap='gray'); logger.error("S3 Vis fail.")
            elif mode == "S4":
                self.ax_dynamic_vis.set_title(f"S4: {title_b}", fontsize=8)
                _, s3mp_s4, _ = extract_ml_features(self.image_path, s3_eff, img_f) 
                s4rm, s4fm, _ = reconstruct_main_body_masks(img_f, s3_eff) 
                if not (s4rm is not None or s4fm is not None) : self.ax_dynamic_vis.imshow(img_f, cmap='gray'); self.ax_dynamic_vis.set_title("S4: Failed", fontsize=8)
                else:
                    tmp = os.path.join(self.temp_dir.name, "s4.png"); visualize_detailed_morphology(img_f, s3mp_s4, s4rm, s4fm, tmp)
                    if os.path.exists(tmp): self.ax_dynamic_vis.imshow(skimage.io.imread(tmp))
                    else: self.ax_dynamic_vis.imshow(img_f, cmap='gray'); logger.error("S4 Vis fail.")
        self.canvas_dynamic_vis.draw(); logger.info(f"Vis updated: {mode}.")

def main():
    try: load_config(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml")); logger.info("Config accessible.")
    except Exception as e: logger.critical(f"Config error: {e}"); print(f"CRITICAL: {e}"); return
    root = tk.Tk(); FeatureViewerApp(root); root.mainloop(); logger.info("Tkinter exit.")

if __name__ == '__main__':
    main()