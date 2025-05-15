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

UPDATE_DEBOUNCE_MS = 300 # Milliseconds to wait after a param change before updating

class FeatureViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive ML Feature Viewer")
        master.geometry("1300x850")

        self.image_path = None
        self.image_array_gray_float = None
        self.default_params = {}
        self.current_params = {}
        self.param_vars = {}
        self.update_job_id = None # For debouncing updates

        try:
            cfg = get_config()
            feature_params_cfg = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params')
            if not feature_params_cfg:
                messagebox.showerror("Config Error", "Feature extraction parameters not found in config.")
                master.destroy()
                return
            self.default_params = feature_params_cfg.copy() # Store a copy for reset
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

    def on_closing(self):
        logger.info("Closing application and cleaning up resources...")
        if self.update_job_id:
            self.master.after_cancel(self.update_job_id)
        
        if hasattr(self, 'canvas_raw') and self.canvas_raw:
            try:
                self.canvas_raw.get_tk_widget().destroy()
            except tk.TclError: pass
        if hasattr(self, 'canvas_features') and self.canvas_features:
            try:
                self.canvas_features.get_tk_widget().destroy()
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

        top_controls_frame = ttk.Frame(main_frame)
        top_controls_frame.pack(fill=tk.X, pady=5)

        load_button = ttk.Button(top_controls_frame, text="Load Greyscale Image", command=self.load_image)
        load_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(top_controls_frame, text="Load an image to start.")
        self.status_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)

        controls_frame_outer = ttk.Labelframe(paned_window, text="Feature Extraction Parameters", padding="10")
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

        plots_frame = ttk.Frame(paned_window, padding="5")
        paned_window.add(plots_frame, weight=3)

        self.fig_raw, self.ax_raw = plt.subplots()
        self.ax_raw.set_title("Raw Image")
        self.ax_raw.set_axis_off()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=plots_frame)
        self.canvas_raw.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_raw.draw()

        self.fig_features, self.ax_features = plt.subplots()
        self.ax_features.set_title("Feature Visualization")
        self.ax_features.set_axis_off()
        self.canvas_features = FigureCanvasTkAgg(self.fig_features, master=plots_frame)
        self.canvas_features.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_features.draw()

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

    def create_param_controls(self, parent_frame):
        # Updated to match new config and ml_feature_extractor.py
        param_specs = {
            'gaussian_sigma': {'min': 0.1, 'max': 5.0, 'type': 'float'},
            'otsu_threshold_multiplier': {'min': 0.1, 'max': 2.0, 'type': 'float'},
            'closing_selem_size': {'min': 0, 'max': 10, 'type': 'int'},
            'min_noise_size': {'min': 1, 'max': 1000, 'type': 'int'},
            'min_significant_area': {'min': 10, 'max': 2000, 'type': 'int'},
            
            # New parameter for egg search constraint
            'egg_posterior_pixel_fraction': {'min': 0.05, 'max': 0.95, 'type': 'float', 'res': 0.01}, 
            
            'egg_threshold_quantile': {'min': 0.5, 'max': 1.0, 'type': 'float', 'res': 0.01},
            
            # Renamed egg parameters (formerly egg_pair_...)
            'egg_min_area': {'min': 10, 'max': 5000, 'type': 'int'}, # Adjusted max based on config
            'egg_max_area': {'min': 50, 'max': 20000, 'type': 'int'}, # Adjusted min based on config
            'egg_min_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_max_solidity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            'egg_min_eccentricity': {'min': 0.0, 'max': 1.0, 'type': 'float', 'res': 0.01},
            
            'orientation_diff_tolerance_deg': {'min': 0, 'max': 90, 'type': 'int'}
            # egg_opening_selem_size removed as it's not in the new config/feature_extractor
        }

        # Iterate over params from loaded config to ensure we only show sliders for what's in the config
        # and has a spec here.
        sorted_params_from_config = sorted(self.default_params.items())

        for name, default_val_cfg in sorted_params_from_config:
            if name not in param_specs:
                logger.info(f"Parameter '{name}' from config has no defined spec for GUI control. Skipping.")
                continue

            spec = param_specs[name]
            min_val, max_val, param_type = spec['min'], spec['max'], spec['type']
            resolution = spec.get('res', 0.1 if param_type == 'float' else 1) # Default resolution

            container = ttk.Frame(parent_frame)
            container.pack(fill=tk.X, pady=2, padx=5)

            label_text = name.replace('_', ' ').title() + ":"
            label = ttk.Label(container, text=label_text, width=30, anchor="w") # Increased width
            label.pack(side=tk.LEFT, padx=(0,5))

            current_val_for_var = self.current_params.get(name, default_val_cfg)

            if param_type == 'float':
                var = tk.DoubleVar(value=float(current_val_for_var))
                # For ttk.Scale, resolution is more about display than strict steps.
                # The command for Int Scale helps enforce steps. For float, it's generally continuous.
                slider = ttk.Scale(container, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
                val_label_var = tk.StringVar(value=f"{var.get():.{max(2, str(resolution).split('.')[-1].__len__())}f}") # Dynamic precision
                var.trace_add("write", lambda n,i,m, sv=var, sl_var=val_label_var, res=resolution: (
                    sl_var.set(f"{sv.get():.{max(2, str(res).split('.')[-1].__len__())}f}"),
                    self.schedule_update()
                ))
            elif param_type == 'int':
                var = tk.IntVar(value=int(current_val_for_var))
                slider = ttk.Scale(container, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL,
                                   command=lambda val, v=var: v.set(int(float(val)))) # Ensure int values
                val_label_var = tk.StringVar(value=f"{var.get():d}")
                var.trace_add("write", lambda n,i,m, sv=var, sl_var=val_label_var: (
                    sl_var.set(f"{sv.get():d}"),
                    self.schedule_update()
                ))

            slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
            val_label = ttk.Label(container, textvariable=val_label_var, width=7, anchor="e") # Increased width
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
        
        self.status_label.config(text="Parameters reset to defaults. Updating...")
        self.schedule_update()

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
            self.schedule_update()
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
            self.status_label.config(text=f"Ready. Displaying features for {os.path.basename(self.image_path)}")
        except Exception as e:
            logger.exception("Error during feature processing or display update.")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.status_label.config(text="Error during processing.")
        finally:
            self.update_job_id = None

    def process_and_display(self):
        params = self.collect_current_params()
        logger.debug(f"Processing with params: {params}")
        
        # Updated return variable name for egg properties
        features_dict, main_props, egg_area_props_result = extract_ml_features(
            image_path=self.image_path, params=params, image_array_gray=self.image_array_gray_float )

        self.ax_features.clear()
        if features_dict is None and main_props is None and egg_area_props_result is None:
            logger.warning("Feature extraction returned no data.")
            self.ax_features.imshow(self.image_array_gray_float, cmap='gray')
            self.ax_features.set_title("Feature Extraction Failed or No Features Found")
        else:
            temp_vis_file_path = os.path.join(self.temp_dir.name, "temp_visualization.png")
            
            # Updated argument name for egg properties
            visualize_ml_features(
                image_path_or_array=self.image_array_gray_float, 
                main_props=main_props,
                egg_area_props=egg_area_props_result, # Changed from egg_pair_props
                output_visualization_path=temp_vis_file_path,
                ml_feature_params=params 
            )

            if os.path.exists(temp_vis_file_path):
                try:
                    vis_img = skimage.io.imread(temp_vis_file_path)
                    self.ax_features.imshow(vis_img)
                    if not self.ax_features.get_title() or self.ax_features.get_title() == "Processed Image":
                         self.ax_features.set_title(f"Features ({os.path.basename(self.image_path if self.image_path else '')})")
                except Exception as read_err:
                    logger.error(f"Failed to read visualization image {temp_vis_file_path}: {read_err}")
                    self.ax_features.imshow(self.image_array_gray_float, cmap='gray')
                    self.ax_features.set_title("Error loading visualization")
            else:
                logger.error("Visualization output file was not created.")
                self.ax_features.imshow(self.image_array_gray_float, cmap='gray')
                self.ax_features.set_title("Visualization Generation Failed")

        self.ax_features.set_axis_off()
        self.canvas_features.draw()
        logger.info("Visualization updated.")

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