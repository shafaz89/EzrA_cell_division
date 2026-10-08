import os
import numpy as np
import pandas as pd
from skimage import io, morphology
import cv2
from scipy import ndimage  # for binary_fill_holes

# --- CONFIG ---
folder_path = "."  # current directory
output_csv = "theta_lengths_ellipse.csv"

results = []

# --- Helper functions ---
def ellipse_arc_length(a, b):
    """Approximate perimeter of an ellipse using Ramanujan formula."""
    return np.pi * (3*(a+b) - np.sqrt((3*a + b)*(a + 3*b)))

def get_largest_contour(binary_img):
    """Get the largest contour of a binary image, filling gaps and smoothing."""
    closed = morphology.binary_closing(binary_img, morphology.disk(1)).astype(np.uint8)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.convexHull(contour)
    return contour

# --- Process images ---
for fname in sorted(os.listdir(folder_path)):
    if not fname.lower().endswith((".png", ".jpg", ".tif")):
        continue

    img_path = os.path.join(folder_path, fname)
    img = io.imread(img_path)

    # Convert to binary if necessary
    if img.max() > 1:
        img = (img > 0).astype(np.uint8)

    contour = get_largest_contour(img)
    if contour is None:
        print(f"No contour found for {fname}, skipping")
        continue

    # Fit ellipse if enough points
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (xc, yc), (d1, d2), angle = ellipse
        a, b = d1/2, d2/2  # semi-major and semi-minor axes
        arc_length = ellipse_arc_length(a, b)
        crossbar_length = 2 * min(a, b)
    else:
        # Fallback: fit circle
        (xc, yc), radius = cv2.minEnclosingCircle(contour)
        a = b = radius
        arc_length = 2 * np.pi * radius
        crossbar_length = 2 * radius

    results.append([fname, arc_length, crossbar_length, a, b])

# --- Save results ---
df = pd.DataFrame(results, columns=["image", "arc_length_px", "crossbar_length_px", "semi_major_px", "semi_minor_px"])
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")
