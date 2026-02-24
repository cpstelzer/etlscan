from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from .plate_layout import calculate_coords_24wellplate, calculate_coords_96wellplate


def well_selection_gui(plate_type, plate_settings, csv_filename):
    if plate_type == "24":
        df_plate = calculate_coords_24wellplate(plate_settings)
        gui_rows = ['A', 'B', 'C', 'D']
        gui_cols = range(1, 7)
        title = "24 Well Plate Selector"
    else:
        df_plate = calculate_coords_96wellplate(plate_settings)
        gui_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        gui_cols = range(1, 13)
        title = "96 Well Plate Selector"

    selected_wells = set()
    buttons = {}

    def toggle_selection(well_name_arg):
        if well_name_arg in selected_wells:
            selected_wells.remove(well_name_arg)
            buttons[well_name_arg].config(bg="SystemButtonFace")
        else:
            selected_wells.add(well_name_arg)
            buttons[well_name_arg].config(bg="yellow")

    def save_selection():
        if not selected_wells:
            messagebox.showinfo("No Selection", "Please select at least one well.")
            return
        selected_df = df_plate[df_plate['wellname'].isin(selected_wells)].sort_values(by='wellnr')
        selected_df.to_csv(csv_filename, index=False)
        messagebox.showinfo("Selected Well Coordinates", selected_df.to_string(index=False))
        root.destroy()

    root = tk.Tk()
    root.title(title)
    for i, r_label in enumerate(gui_rows):
        for j, c_label in enumerate(gui_cols):
            well_name_key = f"{r_label}{c_label}"
            btn = tk.Button(root, text=well_name_key, width=8, height=4,
                            command=lambda wn_cb=well_name_key: toggle_selection(wn_cb))
            btn.grid(row=i, column=j, padx=5, pady=5)
            buttons[well_name_key] = btn
    enter_btn = tk.Button(root, text="Enter", command=save_selection)
    enter_btn.grid(row=len(gui_rows), column=0, columnspan=len(gui_cols), pady=10)
    root.mainloop()
