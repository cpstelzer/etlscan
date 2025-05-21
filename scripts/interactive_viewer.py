# interactive_viewer.py (to be placed in well_plate_analyzer/scripts/)

import sys
import os

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

try:
    from src.utils.config_loader import get_config
    from src.core.ml_feature_extractor import extract_ml_features
    from src.visualization.feature_visualization import visualize_ml_features
    from src.core.detailed_morphology_calculator import reconstruct_main_body_masks 
    from src.visualization.stage_04_visualization import visualize_detailed_morphology 
    from src.utils.common_exceptions import ConfigError 
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    print("Please ensure 'interactive_viewer.py' is in the 'scripts' directory", file=sys.stderr)
    print("and that the 'src' and 'config' directories are structured correctly within the project root (e.g., 'well_plate_analyzer/').", file=sys.stderr)
    print("You should typically run this script from the project root, e.g., 'python scripts/interactive_viewer.py'", file=sys.stderr)
    print("or from the 'scripts' directory, e.g., 'cd scripts; python interactive_viewer.py'.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

UPDATE_DEBOUNCE_MS = 300 

class FeatureViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Viewer (S3/S4 Toggle)") 
        master.geometry("1300x850") # Back to original width for two plots

        self.image_path = None
        self.image_array_gray_float = None 
        self.default_params = {}
        self.current_params = {}
        self.param_vars = {} 
        self.param_widgets = {} 
        self.update_job_id = None 

        # Variable to control which visualization (S3 or S4) is shown in the second panel
        self.active_visualization_mode = tk.StringVar(value="S3") # Default to S3

        try:
            cfg = get_config()
            feature_params_cfg = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params')
            if not feature_params_cfg:
                messagebox.showerror("Config Error", "Feature extraction parameters not found in config.")
                master.destroy()
                return
            self.default_params = feature_params_cfg.copy() 
            self.current_params = feature_params_cfg.copy()
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            messagebox.showerror("Config Error", f"Config file not found. Details: {e}")
            master.destroy()
            return
        except Exception as e:
            logger.exception("Error loading configuration.")
            messagebox.showerror("Config Error", f"Could not load configuration: {e}")
            master.destroy()
            return

        self.temp_dir = tempfile.TemporaryDirectory()
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_gui()
        self.update_threshold_param_visibility() 

    def on_closing(self):
        logger.info("Closing application and cleaning up resources...")
        if self.update_job_id:
            self.master.after_cancel(self.update_job_id)
        
        if hasattr(self, 'canvas_raw') and self.canvas_raw:
            try: self.canvas_raw.get_tk_widget().destroy()
            except tk.TclError: pass
        # Only one active canvas for S3/S4 now
        if hasattr(self, 'canvas_dynamic_vis') and self.canvas_dynamic_vis: 
            try: self.canvas_dynamic_vis.get_tk_widget().destroy()
            except tk.TclError: pass
        
        plt.close('all') 
            
        try:
            self.temp_dir.cleanup()
            logger.info("Temporary directory cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")

        self.master.destroy()
        logger.info("Application closed.")

    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Controls (Load image, Status, Visualization Toggle) ---
        top_controls_frame = ttk.Frame(main_frame)
        top_controls_frame.pack(fill=tk.X, pady=5)

        load_button = ttk.Button(top_controls_frame, text="Load Greyscale Image", command=self.load_image)
        load_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(top_controls_frame, text="Load an image to start.")
        self.status_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Visualization Toggle Radio Buttons
        toggle_frame = ttk.Frame(top_controls_frame)
        toggle_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(toggle_frame, text="Show:").pack(side=tk.LEFT)
        s3_radio = ttk.Radiobutton(toggle_frame, text="S3 Features", variable=self.active_visualization_mode, value="S3", command=self.schedule_update)
        s3_radio.pack(side=tk.LEFT)
        s4_radio = ttk.Radiobutton(toggle_frame, text="S4 Morphology", variable=self.active_visualization_mode, value="S4", command=self.schedule_update)
        s4_radio.pack(side=tk.LEFT)


        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Parameters Panel (left) ---
        controls_frame_outer = ttk.Labelframe(paned_window, text="S3/S4 Preprocessing Parameters", padding="10")
        paned_window.add(controls_frame_outer, weight=1) 

        canvas_params = tk.Canvas(controls_frame_outer)
        scrollbar_params = ttk.Scrollbar(controls_frame_outer, orient="vertical", command=canvas_params.yview)
        self.controls_frame_inner = ttk.Frame(canvas_params)

        self.controls_frame_inner.bind(
            "<Configure>",
            lambda e: canvas_params.configure(scrollregion=canvas_params.bbox("all"))
        )
        canvas_params.create_window((0, 0), window=self.controls_frame_inner, anchor="nw")
        canvas_params.configure(yscrollcommand=scrollbar_params.set)

        controls_frame_outer.grid_rowconfigure(0, weight=1)
        controls_frame_outer.grid_columnconfigure(0, weight=1)
        canvas_params.grid(row=0, column=0, sticky="nsew")
        scrollbar_params.grid(row=0, column=1, sticky="ns")
        self.controls_frame_inner.bind('<Enter>', lambda e: self._bind_mousewheel(e, canvas_params))
        self.controls_frame_inner.bind('<Leave>', lambda e: self._unbind_mousewheel(e, canvas_params))
        self.create_param_controls(self.controls_frame_inner)

        param_buttons_frame = ttk.Frame(self.controls_frame_inner)
        param_buttons_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=5)
        reset_button = ttk.Button(param_buttons_frame, text="Reset Parameters", command=self.reset_parameters)
        reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        update_button = ttk.Button(param_buttons_frame, text="Update Manually", command=self.process_and_display_safe)
        update_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        # --- Plots Panel (right) ---
        plots_container_frame = ttk.Frame(paned_window) 
        paned_window.add(plots_container_frame, weight=3) # Adjusted weight for two plots

        # Raw Image Plot
        self.fig_raw, self.ax_raw = plt.subplots()
        self.ax_raw.set_title("Raw Image")
        self.ax_raw.set_axis_off()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=plots_container_frame)
        self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_raw.draw()

        # Dynamic Visualization Plot (S3 or S4)
        self.fig_dynamic_vis, self.ax_dynamic_vis = plt.subplots() # One figure/axis for dynamic content
        self.ax_dynamic_vis.set_title("S3: ML Feature Visualization") # Initial title
        self.ax_dynamic_vis.set_axis_off()
        self.canvas_dynamic_vis = FigureCanvasTkAgg(self.fig_dynamic_vis, master=plots_container_frame)
        self.canvas_dynamic_vis.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_dynamic_vis.draw()


    def _bind_mousewheel(self, event, canvas):
        canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))

    def _unbind_mousewheel(self, event, canvas):
        canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event, canvas):
        if event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")

    def schedule_update(self):
        if self.update_job_id:
            self.master.after_cancel(self.update_job_id)
        self.update_job_id = self.master.after(UPDATE_DEBOUNCE_MS, self.process_and_display_safe)

    def update_threshold_param_visibility(self, *args):
        mode = self.param_vars.get('thresholding_mode', tk.StringVar(value='otsu')).get()
        
        if 'otsu_threshold_multiplier' in self.param_widgets:
            container_otsu = self.param_widgets['otsu_threshold_multiplier']
            if mode == 'otsu': container_otsu.pack(fill=tk.X, pady=2, padx=5)
            else: container_otsu.pack_forget()
        
        if 'fixed_threshold_value' in self.param_widgets:
            container_fixed = self.param_widgets['fixed_threshold_value']
            if mode == 'fixed': container_fixed.pack(fill=tk.X, pady=2, padx=5)
            else: container_fixed.pack_forget()
        
        self.schedule_update()


    def create_param_controls(self, parent_frame):
        param_specs = {
            'gaussian_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float'},
            'thresholding_mode': {'options': ['otsu', 'fixed'], 'type': 'choice'},
            'otsu_threshold_multiplier': {'min': 0.1, 'max': 2.0, 'type': 'float'},
            'fixed_threshold_value': {'min': 0, 'max': 255, 'type': 'int'}, 
            'closing_selem_size': {'min': 0, 'max': 10, 'type': 'int'},
            'min_noise_size': {'min': 1, 'max': 1000, 'type': 'int'},
            'min_significant_area': {'min': 10, 'max': 2000, 'type': 'int'},
            'egg_posterior_pixel_fraction': {'min': 0.05, 'max': 0.95, 'type': 'float', 'res': 0.01}, 
            'egg_threshold_quantile': {'min': 0.5, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_min_area': {'min': 10, 'max': 5000, 'type': 'int'},
            'egg_max_area': {'min': 50, 'max': 20000, 'type': 'int'},
            'egg_min_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_max_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_min_eccentricity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'orientation_diff_tolerance_deg': {'min': 0, 'max': 90, 'type': 'int'}
        }
        
        ordered_param_names = ['thresholding_mode'] + \
                              [p for p in self.default_params.keys() if p != 'thresholding_mode']


        for name in ordered_param_names:
            default_val_cfg = self.default_params.get(name) 
            
            if default_val_cfg is None and name in param_specs:
                if param_specs[name]['type'] == 'choice':
                    default_val_cfg = param_specs[name]['options'][0]
                elif param_specs[name]['type'] == 'int':
                    default_val_cfg = param_specs[name].get('min', 0) 
                elif param_specs[name]['type'] == 'float':
                    default_val_cfg = param_specs[name].get('min', 0.0)
                if name not in self.current_params: 
                    self.current_params[name] = default_val_cfg
            elif name not in param_specs:
                logger.info(f"Parameter '{name}' from config has no defined spec for GUI control. Skipping.")
                continue
            
            spec = param_specs[name]
            param_type = spec['type']
            
            container = ttk.Frame(parent_frame)
            if name not in ['otsu_threshold_multiplier', 'fixed_threshold_value']:
                 container.pack(fill=tk.X, pady=2, padx=5)

            self.param_widgets[name] = container 

            label_text = name.replace('_', ' ').title() + ":"
            label = ttk.Label(container, text=label_text, width=30, anchor="w")
            label.pack(side=tk.LEFT, padx=(0,5))

            current_val_for_var = self.current_params.get(name, default_val_cfg)

            if param_type == 'choice':
                var = tk.StringVar(value=str(current_val_for_var))
                dropdown = ttk.OptionMenu(container, var, current_val_for_var, *spec['options'],
                                          command=self.update_threshold_param_visibility if name == 'thresholding_mode' else self.schedule_update)
                dropdown.pack(side=tk.LEFT, expand=True, fill=tk.X)
                self.param_vars[name] = var
            
            elif param_type == 'float':
                min_val, max_val = spec['min'], spec['max']
                resolution = spec.get('res', 0.1)
                var = tk.DoubleVar(value=float(current_val_for_var))
                slider = ttk.Scale(container, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
                val_label_var = tk.StringVar(value=f"{var.get():.{max(1, str(resolution).split('.')[-1].__len__())}f}")
                var.trace_add("write", lambda n,i,m, sv=var, sl_var=val_label_var, res=resolution, pname=name: (
                    sl_var.set(f"{sv.get():.{max(1, str(res).split('.')[-1].__len__())}f}"),
                    self.schedule_update() 
                ))
                slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
                val_label = ttk.Label(container, textvariable=val_label_var, width=7, anchor="e")
                val_label.pack(side=tk.RIGHT, padx=(5,0))
                self.param_vars[name] = var

            elif param_type == 'int': 
                min_val, max_val = spec['min'], spec['max']
                var = tk.IntVar(value=int(current_val_for_var))
                slider = ttk.Scale(container, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL,
                                   command=lambda val, v=var: v.set(int(float(val))))
                val_label_var = tk.StringVar(value=f"{var.get():d}")
                var.trace_add("write", lambda n,i,m, sv=var, sl_var=val_label_var, pname=name: (
                    sl_var.set(f"{sv.get():d}"),
                    self.schedule_update()
                ))
                slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
                val_label = ttk.Label(container, textvariable=val_label_var, width=7, anchor="e")
                val_label.pack(side=tk.RIGHT, padx=(5,0))
                self.param_vars[name] = var

    def reset_parameters(self):
        logger.info("Resetting parameters to default values from config.")
        if self.update_job_id:
            self.master.after_cancel(self.update_job_id)
            self.update_job_id = None

        for name, default_val in self.default_params.items():
            if name in self.param_vars:
                self.param_vars[name].set(default_val)
        
        if 'thresholding_mode' in self.param_vars:
            if 'thresholding_mode' not in self.default_params:
                 self.param_vars['thresholding_mode'].set(param_specs['thresholding_mode']['options'][0]) 
        if 'fixed_threshold_value' in self.param_vars:
            if 'fixed_threshold_value' not in self.default_params:
                 self.param_vars['fixed_threshold_value'].set(param_specs['fixed_threshold_value']['min'])


        self.status_label.config(text="Parameters reset to defaults. Updating...")
        self.update_threshold_param_visibility()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Greyscale Image",
            filetypes=(("Image files", "*.png *.tif *.tiff *.jpg *.jpeg"), ("All files", "*.*"))
        )
        if not path: return

        self.image_path = path
        self.status_label.config(text=f"Loading {os.path.basename(path)}...")
        self.master.update_idletasks()
        try:
            img = skimage.io.imread(self.image_path, as_gray=True)
            if img.ndim == 3 and img.shape[2] in [3, 4]:
                img = skimage.color.rgb2gray(img[:,:,:3])
            self.image_array_gray_float = skimage.util.img_as_float(img) 
            logger.info(f"Image loaded: {self.image_path}")
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}. Processing...")
            self.update_raw_image_display()
            self.update_threshold_param_visibility() 
        except Exception as e:
            logger.exception(f"Failed to load image {self.image_path}")
            messagebox.showerror("Image Load Error", f"Failed to load or process image: {e}")
            self.status_label.config(text="Failed to load image.")
            self.image_path = None
            self.image_array_gray_float = None

    def update_raw_image_display(self):
        if self.image_array_gray_float is not None:
            self.ax_raw.clear()
            self.ax_raw.imshow(self.image_array_gray_float, cmap='gray')
            self.ax_raw.set_title(f"Raw: {os.path.basename(self.image_path if self.image_path else 'No Image')}")
            self.ax_raw.set_axis_off()
            self.canvas_raw.draw()

    def collect_current_params(self):
        for name, var in self.param_vars.items():
            if name == 'fixed_threshold_value':
                try: self.current_params[name] = int(var.get())
                except ValueError: self.current_params[name] = self.default_params.get(name, 0) 
            else:
                self.current_params[name] = var.get()
        return self.current_params.copy()

    def process_and_display_safe(self):
        if self.image_array_gray_float is None:
            self.status_label.config(text="Please load an image first.")
            self.update_job_id = None 
            return

        self.status_label.config(text=f"Processing {os.path.basename(self.image_path)}...")
        self.master.update_idletasks()
        try:
            self.process_and_display()
            self.status_label.config(text=f"Ready. Displaying for {os.path.basename(self.image_path)}")
        except ConfigError as ce: 
            logger.error(f"Configuration error during processing: {ce}")
            messagebox.showerror("Configuration Error", f"Error in parameters: {ce}")
            self.status_label.config(text=f"Config error: {ce}")
        except Exception as e:
            logger.exception("Error during feature processing or display update.")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.status_label.config(text="Error during processing.")
        finally:
            self.update_job_id = None

    def process_and_display(self):
        params_for_processing = self.collect_current_params() 
        logger.debug(f"Processing with params: {params_for_processing}")
        
        # Always extract S3 features, as S4 might need main_props from it
        s3_features_dict, s3_main_props, s3_egg_area_props = extract_ml_features(
            image_path=self.image_path, 
            params=params_for_processing, 
            image_array_gray=self.image_array_gray_float 
        )

        self.ax_dynamic_vis.clear() # Clear the dynamic axis before drawing

        if self.active_visualization_mode.get() == "S3":
            self.ax_dynamic_vis.set_title(f"S3 Features ({os.path.basename(self.image_path if self.image_path else '')})")
            if s3_features_dict is None and s3_main_props is None and s3_egg_area_props is None:
                logger.warning("S3: Feature extraction returned no data.")
                self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')
                self.ax_dynamic_vis.set_title("S3: Feature Extraction Failed")
            else:
                temp_s3_vis_path = os.path.join(self.temp_dir.name, "temp_s3_visualization.png")
                visualize_ml_features(
                    image_path_or_array=self.image_array_gray_float, 
                    main_props=s3_main_props,
                    egg_area_props=s3_egg_area_props, 
                    output_visualization_path=temp_s3_vis_path,
                    ml_feature_params=params_for_processing 
                )
                if os.path.exists(temp_s3_vis_path):
                    try:
                        vis_s3_img = skimage.io.imread(temp_s3_vis_path)
                        self.ax_dynamic_vis.imshow(vis_s3_img)
                    except Exception as read_err:
                        logger.error(f"S3: Failed to read visualization image {temp_s3_vis_path}: {read_err}")
                        self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')
                else:
                    logger.error("S3: Visualization output file was not created.")
                    self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')

        elif self.active_visualization_mode.get() == "S4":
            self.ax_dynamic_vis.set_title(f"S4 Morphology ({os.path.basename(self.image_path if self.image_path else '')})")
            s4_raw_mask, s4_filled_mask, _ = reconstruct_main_body_masks(
                image_gray_original=self.image_array_gray_float, 
                s3_feature_extraction_params=params_for_processing 
            )
            if s4_raw_mask is None and s4_filled_mask is None:
                logger.warning("S4: Mask reconstruction returned no data.")
                self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')
                self.ax_dynamic_vis.set_title("S4: Mask Reconstruction Failed")
            else:
                temp_s4_vis_path = os.path.join(self.temp_dir.name, "temp_s4_visualization.png")
                visualize_detailed_morphology(
                    image_path_or_array=self.image_array_gray_float, 
                    main_props_s3=s3_main_props, # Pass S3 main_props for orientation
                    raw_body_mask_s4=s4_raw_mask,
                    filled_body_mask_s4=s4_filled_mask,
                    output_visualization_path=temp_s4_vis_path
                )
                if os.path.exists(temp_s4_vis_path):
                    try:
                        vis_s4_img = skimage.io.imread(temp_s4_vis_path)
                        self.ax_dynamic_vis.imshow(vis_s4_img)
                    except Exception as read_err:
                        logger.error(f"S4: Failed to read visualization image {temp_s4_vis_path}: {read_err}")
                        self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')
                else:
                    logger.error("S4: Visualization output file was not created.")
                    self.ax_dynamic_vis.imshow(self.image_array_gray_float, cmap='gray')
        
        self.ax_dynamic_vis.set_axis_off()
        self.canvas_dynamic_vis.draw()
        logger.info(f"Dynamic visualization updated to: {self.active_visualization_mode.get()}.")


def main():
    try:
        get_config()
        logger.info("Configuration file accessible.")
    except FileNotFoundError as e:
        logger.critical(f"Critical: pipeline_config.yml not found. {e}")
        print(f"CRITICAL ERROR: pipeline_config.yml not found. Ensure 'config/pipeline_config.yml' exists in 'well_plate_analyzer/config/'.\nDetails: {e}", file=sys.stderr)
        return
    except Exception as e:
        logger.critical(f"Critical: Error during initial configuration access. {e}")
        print(f"CRITICAL ERROR: Could not access configuration: {e}", file=sys.stderr)
        return

    root = tk.Tk()
    app = FeatureViewerApp(root)
    root.mainloop()
    logger.info("Tkinter mainloop exited.")

if __name__ == '__main__':
    main()