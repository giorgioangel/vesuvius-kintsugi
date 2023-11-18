import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk
from PIL import Image, ImageTk
import numpy as np
import zarr
from collections import deque
import threading
import math
import os
import sys

class VesuviusKintsugi:
    def __init__(self):
        self.voxel_data = None
        self.photo_img = None
        self.resized_img = None
        self.z_index = 0
        self.pencil_size = 0
        self.click_coordinates = None
        self.threshold = 4
        self.log_text = None
        self.zoom_level = 1
        self.max_zoom_level = 15
        self.drag_start_x = None
        self.drag_start_y = None
        self.image_position_x = 0
        self.image_position_y = 0
        self.pencil_cursor = None  # Reference to the circle representing the pencil size
        self.flood_fill_active = False  # Flag to control flood fill
        self.history = []  # List to store a limited history of image states
        self.max_history_size = 3  # Maximum number of states to store
        self.mask_data = None
        self.show_mask = True  # Default to showing the mask
        self.show_image = True
        self.init_ui()

    def load_data(self):
        # Ask the user to select a directory containing Zarr data
        dir_path = filedialog.askdirectory(title="Select Zarr Directory")
        if dir_path:
            try:
                # Load the Zarr data into the voxel_data attribute
                self.voxel_data = np.array(zarr.open(dir_path, mode='r'))
                self.mask_data = np.zeros_like(self.voxel_data)
                self.z_index = 0
                self.update_display_slice()
                self.file_name = os.path.basename(dir_path)
                self.root.title(f"Vesuvius Kintsugi - {self.file_name}")
                self.update_log(f"Data loaded successfully.")
            except Exception as e:
                self.update_log(f"Error loading data: {e}")

    def load_mask(self):
        if self.voxel_data is None:
            self.update_log("No voxel data loaded. Load voxel data first.")
            return

        # Prompt to save changes if there are any unsaved changes
        if self.history:
            if not tk.messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Do you want to continue without saving?"):
                return

        # File dialog to select mask file
        mask_file_path = filedialog.askdirectory(
            title="Select Label Zarr File")
        

        if mask_file_path:
            try:
                loaded_mask = np.array(zarr.open(mask_file_path, mode='r'))
                if loaded_mask.shape == self.voxel_data.shape:
                    self.mask_data = loaded_mask
                    self.update_display_slice()
                    self.update_log("Label loaded successfully.")
                else:
                    self.update_log("Error: Label dimensions do not match the voxel data dimensions.")
            except Exception as e:
                self.update_log(f"Error loading mask: {e}")

    def save_image(self):
        if self.mask_data is not None:
            # Construct the default file name for saving
            base_name = os.path.splitext(os.path.basename(self.file_name))[0]
            default_save_file_name = f"{base_name}_label.zarr"
            parent_directory = os.path.join(self.file_name, os.pardir)
            # Open the file dialog with the proposed file name
            save_file_path = filedialog.asksaveasfilename(
                initialdir=parent_directory,
                title="Select Directory to Save Mask Zarr",
                initialfile=default_save_file_name,
                filetypes=[("Zarr files", "*.zarr")]
            )

            if save_file_path:
                try:
                    # Save the Zarr array to the chosen file path
                    zarr.save_array(save_file_path, self.mask_data)
                    self.update_log(f"Mask saved as Zarr in {save_file_path}")
                except Exception as e:
                    self.update_log(f"Error saving mask as Zarr: {e}")
        else:
            self.update_log("No mask data to save.")

    def update_threshold(self, val):
        try:
            self.threshold = int(float(val))
            self.bucket_threshold_var.set(f"{self.threshold}")
            self.update_log(f"Threshold set to {self.threshold}")
        except ValueError:
            self.update_log("Invalid threshold value.")

    def threaded_flood_fill(self):
        if self.click_coordinates and self.voxel_data is not None:
            # Run flood_fill_3d in a separate thread
            thread = threading.Thread(target=self.flood_fill_3d, args=(self.click_coordinates,))
            thread.start()
        else:
            self.update_log("No starting point or data for flood fill.")

    def flood_fill_3d(self, start_coord):
        self.flood_fill_active = True
        target_color = self.voxel_data[start_coord]
        queue = deque([start_coord])
        visited = set()

        counter = 0
        while self.flood_fill_active and queue:
            cz, cy, cx = queue.popleft()

            if (cz, cy, cx) in visited or not (0 <= cz < self.voxel_data.shape[0] and 0 <= cy < self.voxel_data.shape[1] and 0 <= cx < self.voxel_data.shape[2]):
                continue

            visited.add((cz, cy, cx))

            if abs(int(self.voxel_data[cz, cy, cx]) - int(target_color)) <= self.threshold:
                self.mask_data[cz, cy, cx] = 1
                counter += 1
                for dz in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dz == 0 and dx == 0 and dy == 0:
                                continue
                            queue.append((cz + dz, cy + dy, cx + dx))

            if counter % 10 == 0:
                self.root.after(1, self.update_display_slice)
        if self.flood_fill_active == True:
            self.flood_fill_active = False
            self.update_log("Flood fill ended.")

    def stop_flood_fill(self):
        self.flood_fill_active = False
        self.update_log("Flood fill stopped.")

    def save_state(self):
        # Save the current state of the image before modifying it
        if self.voxel_data is not None:
            if len(self.history) == self.max_history_size:
                self.history.pop(0)  # Remove the oldest state
            self.history.append((self.voxel_data.copy(), self.mask_data.copy()))

    def undo_last_action(self):
        if self.history:
            self.voxel_data, self.mask_data = self.history.pop() 
            self.update_display_slice()
            self.update_log("Last action undone.")
        else:
            self.update_log("No more actions to undo.")

    def on_canvas_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_canvas_drag(self, event):
        if self.drag_start_x is not None and self.drag_start_y is not None:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.image_position_x += dx
            self.image_position_y += dy
            self.update_display_slice()
            self.drag_start_x, self.drag_start_y = event.x, event.y

    def on_canvas_pencil_drag(self, event):
        if self.mode.get() == "pencil" or self.mode.get() == "eraser":
            self.save_state()
            self.color_pixel(self.calculate_image_coordinates(event))

    def on_canvas_release(self, event):
        self.drag_start_x = None
        self.drag_start_y = None

    def resize_with_aspect(self, image, target_width, target_height, zoom=1):
        original_width, original_height = image.size
        zoomed_width, zoomed_height = int(original_width * zoom), int(original_height * zoom)
        aspect_ratio = original_height / original_width
        new_height = int(target_width * aspect_ratio)
        new_height = min(new_height, target_height)
        return image.resize((zoomed_width, zoomed_height), Image.NEAREST)

    def update_display_slice(self):
        if self.voxel_data is not None:
            target_width_xy = self.canvas.winfo_width()
            target_height_xy = self.canvas.winfo_height()

            # Convert the current slice to an RGBA image
            if self.show_image:
                img = Image.fromarray(self.voxel_data[self.z_index, :, :].astype('uint16')).convert('RGBA')
            else:
                img = Image.fromarray(np.zeros_like(self.voxel_data[self.z_index, :, :]).astype('uint16')).convert('RGBA')

            # Only overlay the mask if show_mask is True
            if self.mask_data is not None and self.show_mask:
                mask = np.uint8(self.mask_data[self.z_index, :, :] * 255)
                yellow = np.zeros_like(mask, dtype=np.uint8)
                yellow[:, :] = 255  # Yellow color
                mask_img = Image.fromarray(np.stack([yellow, yellow, np.zeros_like(mask), mask], axis=-1), 'RGBA')

                # Overlay the mask on the original image
                img = Image.alpha_composite(img, mask_img)

            # Resize the image with aspect ratio
            img = self.resize_with_aspect(img, target_width_xy, target_height_xy, zoom=self.zoom_level)

            # Convert back to a format that can be displayed in Tkinter
            self.resized_img = img.convert('RGB')
            self.photo_img = ImageTk.PhotoImage(image=self.resized_img)
            self.canvas.create_image(self.image_position_x, self.image_position_y, anchor=tk.NW, image=self.photo_img)
            self.canvas.tag_raise(self.z_slice_text)
            self.canvas.tag_raise(self.cursor_pos_text)

    def update_info_display(self):
        self.canvas.itemconfigure(self.z_slice_text, text=f"Z-Slice: {self.z_index}")
        if self.click_coordinates:
            try:
                _, cursor_y, cursor_x = self.calculate_image_coordinates(self.click_coordinates)
            except:
                cursor_x, cursor_y = 0, 0
            self.canvas.itemconfigure(self.cursor_pos_text, text=f"Cursor Position: ({cursor_x}, {cursor_y})")



    def on_canvas_click(self, event):
        self.save_state()
        img_coords = self.calculate_image_coordinates(event)
        if self.mode.get() == "bucket":
            if self.flood_fill_active == True:
                self.update_log("Last flood fill hasn't finished yet.")
            else:
                # Assuming the flood fill functionality
                self.click_coordinates = img_coords
                self.update_log("Starting flood fill...")
                self.threaded_flood_fill()  # Assuming threaded_flood_fill is implemented for non-blocking UI
        elif self.mode.get() == "pencil":
            # Assuming the pencil (pixel editing) functionality
            self.color_pixel(img_coords)  # Assuming color_pixel is implemented

    def calculate_image_coordinates(self, input):
        if input is None:
            return 0, 0, 0  # Default values
        if isinstance(input, tuple):
                _, y, x = input
        elif hasattr(input, 'x') and hasattr(input, 'y'):
                x, y = input.x, input.y
        else:
            # Handle unexpected input types
            raise ValueError("Input must be a tuple or an event object")
        if self.voxel_data is not None:
            original_image_height, original_image_width = self.voxel_data[self.z_index].shape

            # Dimensions of the image at the current zoom level
            zoomed_width = original_image_width * self.zoom_level
            zoomed_height = original_image_height * self.zoom_level

            # Adjusting click position for panning
            pan_adjusted_x = x - self.image_position_x
            pan_adjusted_y = y - self.image_position_y

            # Calculate the position in the zoomed image
            zoomed_image_x = max(0, min(pan_adjusted_x, zoomed_width))
            zoomed_image_y = max(0, min(pan_adjusted_y, zoomed_height))

            # Scale back to original image coordinates
            img_x = int(zoomed_image_x / self.zoom_level)
            img_y = int(zoomed_image_y / self.zoom_level)

            # Debugging output
            #print(f"Clicked at: ({x}, {y}), Image Coords: ({img_x}, {img_y})")

            return self.z_index, img_y, img_x
    
    def color_pixel(self, img_coords):
        z_index, center_y, center_x = img_coords
        if self.voxel_data is not None:
            # Calculate the square bounds of the circle
            min_x = max(0, center_x - self.pencil_size)
            max_x = min(self.voxel_data.shape[2] - 1, center_x + self.pencil_size)
            min_y = max(0, center_y - self.pencil_size)
            max_y = min(self.voxel_data.shape[1] - 1, center_y + self.pencil_size)

            if self.mode.get() == "pencil":
                mask_value = 1
            elif self.mode.get() == "eraser":
                mask_value = 0
            else:
                self.update_log("Something wrong with pencil/eraser.")
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Check if the pixel is within the circle's radius
                    if math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= self.pencil_size:
                        self.mask_data[z_index, y, x] = mask_value
            self.update_display_slice()

    
    def update_pencil_size(self, val):
        self.pencil_size = int(float(val))
        self.pencil_size_var.set(f"{self.pencil_size}")
        self.update_log(f"Pencil size set to {self.pencil_size}")

    def update_pencil_cursor(self, event):
        # Remove the old cursor representation
        if self.pencil_cursor:
            self.canvas.delete(self.pencil_cursor)
            self.update_display_slice()

        if self.mode.get() == "pencil":
            color = "yellow"
        if self.mode.get() == "eraser":
            color = "white"
        if self.mode.get() == "eraser" or self.mode.get() == "pencil":
            radius = self.pencil_size * self.zoom_level  # Adjust radius based on zoom level
            self.pencil_cursor = self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2)
        self.click_coordinates = (self.z_index, event.y, event.x)
        self.update_info_display()
            
    def scroll_or_zoom(self, event):
        # Adjust for different platforms
        ctrl_pressed = False
        if sys.platform.startswith('win'):
            # Windows
            ctrl_pressed = event.state & 0x0004
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            # Linux or macOS
            ctrl_pressed = event.state & 4

        if ctrl_pressed:
            self.zoom(event)
        else:
            self.scroll(event)

    def scroll(self, event):
        if self.voxel_data is not None:
            # Update the z_index based on scroll direction
            delta = 1 if event.delta > 0 else -1
            self.z_index = max(0, min(self.z_index + delta, self.voxel_data.shape[0] - 1))
            self.update_display_slice()

    
    def zoom(self, event):
        zoom_amount = 0.1  # Adjust the zoom sensitivity as needed
        if event.delta > 0:
            self.zoom_level = min(self.max_zoom_level, self.zoom_level + zoom_amount)
        else:
            self.zoom_level = max(1, self.zoom_level - zoom_amount)
        self.update_display_slice()

    def toggle_mask(self):
        # Toggle the state
        self.show_mask = not self.show_mask
        # Update the variable for the Checkbutton
        self.show_mask_var.set(self.show_mask)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Label {'shown' if self.show_mask else 'hidden'}.\n")

    def toggle_image(self):
        # Toggle the state
        self.show_image = not self.show_image
        # Update the variable for the Checkbutton
        self.show_image_var.set(self.show_image)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Image {'shown' if self.show_image else 'hidden'}.\n")

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Info")
        help_window.geometry("300x700")  # Adjust size as necessary
        help_window.resizable(False, False)

        info_text = """Vesuvius Kintsugi: A tool for labeling 3D Zarr images for the Vesuvius Challenge (scrollprize.org).

        Pour the gold into the crackles!

        Commands Overview:
        - Icons (Left to Right): 
        1. Open Zarr 3D Image
        2. Open Zarr 3D Label
        3. Save Zarr 3D Label
        4. Undo Last Action
        5. Brush Tool
        6. Eraser Tool
        7. 'Pour Gold' (3D Bucket)
        8. STOP (Halts Gold Pouring)
        9. Pencil Size Selector
        10. Gold Pouring Threshold
        11. Info

        - Bottom
        1. Toggle Label/Image (Side-by-side buttons for showing/hiding Labeling or Image)

        Usage Tips:
        - Pouring Gold: Propagates gold in 3D based on neighbor voxel values and threshold.
        - Navigation: Drag with left mouse button.
        - Tools: Use right mouse button for brush, eraser and gold pouring.
        - Zoom: CTRL+Scroll. Change Z-axis slice with mouse wheel.
        - Toggle Buttons: The 'Toggle Label' and 'Toggle Image' buttons appear pressed when active, indicating visibility of labels or the image respectively.

        Created by Dr. Giorgio Angelotti, Vesuvius Kintsugi is designed for efficient 3D voxel image labeling. Released under the MIT license."""

        label = tk.Label(help_window, text=info_text, wraplength=250)
        label.pack(pady=10, padx=10)

        close_button = tk.Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=10)

    def update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Scroll to the bottom

    @staticmethod
    def create_tooltip(widget, text):
        # Implement a simple tooltip
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        tooltip.withdraw()

        label = tk.Label(tooltip, text=text, background="#FFFFE0", relief='solid', borderwidth=1, padx=1, pady=1)
        label.pack(ipadx=1)

        def enter(event):
            x = y = 0
            x, y, cx, cy = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def init_ui(self):
        self.root = tk.Tk()
        self.root.iconbitmap("./icons/favicon.ico")
        self.root.title("Vesuvius Kintsugi")

        # Use a ttk.Style object to configure style aspects of the application
        style = ttk.Style()
        style.configure('TButton', padding=5)  # Add padding around buttons
        style.configure('TFrame', padding=5)  # Add padding around frames

        # Create a toolbar frame at the top with some padding
        toolbar_frame = ttk.Frame(self.root, padding="5 5 5 5")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # Create a drawing tools frame
        drawing_tools_frame = tk.Frame(toolbar_frame)
        drawing_tools_frame.pack(side=tk.LEFT, padx=5)

        # Load and set icons for buttons (icons need to be added)
        load_icon = PhotoImage(file='./icons/open-64.png')  # Replace with actual path to icon
        save_icon = PhotoImage(file='./icons/save-64.png')  # Replace with actual path to icon
        undo_icon = PhotoImage(file='./icons/undo-64.png')  # Replace with actual path to icon
        brush_icon = PhotoImage(file='./icons/brush-64.png')  # Replace with actual path to icon
        eraser_icon = PhotoImage(file='./icons/eraser-64.png')  # Replace with actual path to icon
        bucket_icon = PhotoImage(file='./icons/bucket-64.png')
        stop_icon = PhotoImage(file='./icons/stop-60.png')
        help_icon = PhotoImage(file='./icons/help-48.png')
        load_mask_icon = PhotoImage(file='./icons/ink-64.png')  # Replace with the actual path to icon

        self.mode = tk.StringVar(value="bucket")

        # Add buttons with icons and tooltips to the toolbar frame
        load_button = ttk.Button(toolbar_frame, image=load_icon, command=self.load_data)
        load_button.image = load_icon
        load_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_button, "Open Zarr 3D Image")

        load_mask_button = ttk.Button(toolbar_frame, image=load_mask_icon, command=self.load_mask)
        load_mask_button.image = load_mask_icon
        load_mask_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_mask_button, "Load Ink Label")

        save_button = ttk.Button(toolbar_frame, image=save_icon, command=self.save_image)
        save_button.image = save_icon
        save_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(save_button, "Save Zarr 3D Label")

        undo_button = ttk.Button(toolbar_frame, image=undo_icon, command=self.undo_last_action)
        undo_button.image = undo_icon
        undo_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(undo_button, "Undo Last Action")

        # Brush tool button
        brush_button = ttk.Radiobutton(toolbar_frame, image=brush_icon, variable=self.mode, value="pencil")
        brush_button.image = brush_icon
        brush_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(brush_button, "Brush Tool")

        # Eraser tool button
        eraser_button = ttk.Radiobutton(toolbar_frame, image=eraser_icon, variable=self.mode, value="eraser")
        eraser_button.image = eraser_icon
        eraser_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(eraser_button, "Eraser Tool")

        # Bucket tool button
        bucket_button = ttk.Radiobutton(toolbar_frame, image=bucket_icon, variable=self.mode, value="bucket")
        bucket_button.image = bucket_icon
        bucket_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(bucket_button, "Flood Fill Tool")

        # Stop tool button
        stop_button = ttk.Button(toolbar_frame, image=stop_icon, command=self.stop_flood_fill)
        stop_button.image = stop_icon
        stop_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(stop_button, "Stop Flood Fill")

        # Help button
        help_button = ttk.Button(toolbar_frame, image=help_icon, command=self.show_help)
        help_button.image = help_icon
        help_button.pack(side=tk.RIGHT, padx=2)
        self.create_tooltip(help_button, "Info")

        self.pencil_size_var = tk.StringVar(value="0")  # Default pencil size
        pencil_size_label = ttk.Label(toolbar_frame, text="Pencil Size:")
        pencil_size_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        pencil_size_slider = ttk.Scale(toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_pencil_size)
        pencil_size_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(pencil_size_slider, "Adjust Pencil Size")

        pencil_size_value_label = ttk.Label(toolbar_frame, textvariable=self.pencil_size_var)
        pencil_size_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Bucket Threshold Slider
        self.bucket_threshold_var = tk.StringVar(value="4")  # Default threshold
        bucket_threshold_label = ttk.Label(toolbar_frame, text="Bucket Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        bucket_threshold_slider = ttk.Scale(toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold)
        bucket_threshold_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(bucket_threshold_slider, "Adjust Bucket Threshold")

        bucket_threshold_value_label = ttk.Label(toolbar_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # The canvas itself remains in the center
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.z_slice_text = self.canvas.create_text(10, 10, anchor=tk.NW, text=f"Z-Slice: {self.z_index}", fill="red")

        self.cursor_pos_text = self.canvas.create_text(10, 30, anchor=tk.NW, text="Cursor Position: (0, 0)", fill="red")


        # Bind event handlers
        self.canvas.bind("<Motion>", self.update_pencil_cursor)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_press)
        self.canvas.bind("<B3-Motion>", self.on_canvas_pencil_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_click)  # Assuming on_canvas_click is implemented
        self.canvas.bind("<MouseWheel>", self.scroll_or_zoom)  # Assuming scroll_or_zoom is implemented

        # Variables for toggling states
        self.show_mask_var = tk.BooleanVar(value=self.show_mask)
        self.show_image_var = tk.BooleanVar(value=self.show_image)

        # Create a frame to hold the toggle buttons
        toggle_frame = tk.Frame(self.root)
        toggle_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        # Create toggle buttons for mask and image visibility
        toggle_mask_button = ttk.Checkbutton(toggle_frame, text="Toggle Label", command=self.toggle_mask, variable=self.show_mask_var)
        toggle_mask_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_image_button = ttk.Checkbutton(toggle_frame, text="Toggle Image", command=self.toggle_image, variable=self.show_image_var)
        toggle_image_button.pack(side=tk.LEFT, padx=5, anchor='s')

        # Create a frame for the log text area and scrollbar
        log_frame = tk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create the log text widget
        self.log_text = tk.Text(log_frame, height=4, width=50)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create the scrollbar and associate it with the log text widget
        log_scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = log_scrollbar.set

        self.root.mainloop()

if __name__ == "__main__":
    editor = VesuviusKintsugi()
