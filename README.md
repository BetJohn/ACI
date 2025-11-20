````markdown
# 🏎️ Interactive F1 Track Segmentation Tool

## 🏁 Project Goal: Clean Track Binarization

This script is a specialized, interactive tool designed for the **binarization stage** of a larger project aiming to accurately **recreate a clean F1 track mask** from raw aerial or satellite imagery.

The primary goal is to **isolate the asphalt track surface** and output a pristine, black-and-white image where the track is one color (e.g., white) and everything else is the opposite (e.g., black). This **clean binary output** is crucial for subsequent stages, such as path planning, measurement, or simulation.

The tool achieves this by combining:

1.  **Color Inclusion:** Selecting the specific BGR colors of the track asphalt.
2.  **Morphological Filtering:** Removing noise, smoothing edges, and bridging small gaps.
3.  **Flood Fill Refinement:** Ensuring complete track coverage while excluding large surrounding areas.

---

## ⚙️ Prerequisites

- Python 3.x
- **OpenCV (`cv2`)**
- **NumPy**

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy
```
````

## 🖼️ Configuration

The default image file is set in the `FILENAME` variable. You should ensure this file exists or update the path:

```python
FILENAME = "./images/raw_image_3_formula_1.png" # <--- Update this path!
```

## 🚀 How to Run

1.  Save the code as a Python file (e.g., `track_segmentor.py`).

2.  Ensure your target image is accessible at the path defined in `FILENAME`.

3.  Run the script:

    ```bash
    python track_segmentor.py
    ```

Multiple interactive windows will open:

- **`Image` (Original Image)**
- **`Track Preview` (Result Preview)**
- **`Selected Colors` (Control Panel)**
- **`Flood Preview`**
- **`Flood Points` (Flood Control)**

---

## 🖱️ Interactive Controls

### **Image Window (`Image`)**

| Action            | Control          | Description                                                                                    |
| :---------------- | :--------------- | :--------------------------------------------------------------------------------------------- |
| **Include Color** | **Left Click**   | Click a pixel on the track to add its BGR value to the list of included colors.                |
| **Select Region** | **Shift + Drag** | Draw a rectangle to define a **Region of Interest (ROI)**, restricting the initial color mask. |

### **Control Panels & Preview Windows**

| Window                | Action                | Control                          | Function                                                                                             |
| :-------------------- | :-------------------- | :------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **`Selected Colors`** | **Remove Color**      | **Left Click** on a color swatch | Removes a previously included color.                                                                 |
| **`Track Preview`**   | **Add Flood Seed**    | **Left Click** anywhere          | Adds a point as a seed for the **Flood Fill** operation, which runs on the **color-filtered** image. |
| **`Flood Points`**    | **Remove Flood Seed** | **Left Click** on a seed swatch  | Removes a selected flood seed point.                                                                 |

### **Trackbars (Fine-Tuning Parameters)**

| Trackbar Name | Window          | Function                                                                        |
| :------------ | :-------------- | :------------------------------------------------------------------------------ |
| `Tol Inc`     | `Track Preview` | Adjusts the **Tolerance** for color inclusion (`tol_include`).                  |
| `Open Ker`    | `Track Preview` | Adjusts the kernel size for the initial **Opening** operation (`ker_open`).     |
| `Close Ker`   | `Track Preview` | Adjusts the kernel size for the initial **Closing** operation (`ker_close`).    |
| `Flood Tol`   | `Flood Preview` | Adjusts the **Tolerance** for the flood-fill operation (`flood_tolerance`).     |
| `Flood Open`  | `Flood Preview` | Adjusts the kernel size for **Opening** _after_ flood fill (`flood_ker_open`).  |
| `Flood Close` | `Flood Preview` | Adjusts the kernel size for **Closing** _after_ flood fill (`flood_ker_close`). |

### **Keyboard Shortcuts**

| Key     | Action                     | Description                                                                                                                                              |
| :------ | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`o`** | **Toggle Morph Order**     | Switches the initial morphological operation order between `open_close` and `close_open`.                                                                |
| **`r`** | **Toggle Region Restrict** | Switches the region-of-interest restriction **ON/OFF**.                                                                                                  |
| **`f`** | **Toggle Flood Mode**      | Switches the flood-fill mode between **Fixed** and **Floating** range.                                                                                   |
| **`s`** | **Save Output**            | Saves the final, refined binary mask to a file named **`track_binary.png`**. This final step uses **Otsu's Thresholding** for a clean black/white image. |
| **`q`** | **Quit**                   | Closes all windows and exits the program.                                                                                                                |

```

```
