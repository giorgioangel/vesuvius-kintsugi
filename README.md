# Vesuvius Kintsugi

## Introduction

Vesuvius Kintsugi is a tool designed for the Vesuvius Challenge (https://scrollprize.org/) aimed at facilitating labeling 3D voxel data extracted from the Herculaneum scrolls. This tool alleviates the complexities of annotating 3D segments by providing an intuitive interface for manual segmentation and labeling of the "ink crackles", crucial for the creation of ground truth datasets necessary for machine learning ink detection models.

POUR GOLD INTO THE CRACKLES!

![ pour gold](example.gif)
## Features
![my screenshot](screenshot.jpg)
### Interactive UI
- A graphical user interface that enables users to navigate through 3D Zarr data.
- Tools for labeling including brush, eraser, and flood fill (3D bucket).
- Toggle visibility of labels, barrier masks, and original images.
- Adjust the transparency of label and barrier overlays.
- Undo feature to revert the last action.
- Switch between editing labels and barrier masks.

### Advanced Labeling Features
- A 3D flood fill algorithm to segment contiguous structures within the data.
- Configurable settings such as intensity threshold, bucket layer selection, and maximum propagation steps for precise control.
- Edit barrier masks to prevent the flood fill algorithm from propagating into restricted areas.

### Efficient Data Handling
- Loading and saving of Zarr formatted data directly from and to disk.

### Accessibility and Assistance
- Tooltips provide guidance for each control.
- A help window offers detailed tool usage instructions.
- The interface is designed to be user-friendly and intuitive.

### Letter dataset
- A small Zarr dataset of cropped letters from Scroll 1 to practice labeling

## Installation

Before using Vesuvius Kintsugi, ensure that the following dependencies are installed:

```console
pip install numpy zarr Pillow tk ttkthemes
```

## Usage
Run the main Python script to launch the Vesuvius Kintsugi interface:
```console
python kintsugi.py
```
Use the provided buttons to load Zarr data, navigate through slices, label regions, and save your work. The interface supports various annotation strategies suitable for different types of 3D structures.

## Customization
Directly customize the pencil size, flood fill threshold, and maximum propagation steps from the UI to tailor the tool's behavior to your specific data and labeling needs.

## Contributing
We encourage contributions to enhance the functionality of Vesuvius Kintsugi.

## Author
Dr. Giorgio Angelotti

For any inquiries or further information, please contact me at giorgio.angelotti@isae-supaero.fr

## License
This project is licensed under the MIT License - see the LICENSE file for details.


