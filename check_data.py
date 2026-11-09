
# Import random so the same sample of images is picked every run.
import random

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import OpenCV to draw on the images and save the contact sheets.
import cv2

# Import NumPy to arrange the images into a single sheet.
import numpy as np

# Import the data loader so we check exactly what the model will be given.
from dataset import FomoData, load_manifest


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Set the input size used by the model.
size = 96

# Set how many images to check.
number_of_images = 20

# Set how many images are placed in each row of the sheet.
images_per_row = 5

# Set how much to enlarge each 96 pixel image so it is easy to see.
zoom = 3

# Set the folder where the sheets are saved.
output_dir = Path(__file__).parent / "checks"


# --------------------------------------------------
# 2. DRAWING
# --------------------------------------------------

def draw(image, boxes, target, grid):
    """Draw the grid, person cells, boxes and centres on one enlarged image."""

    # Enlarge the image and convert it from RGB to BGR for OpenCV.
    canvas = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Calculate the width of one grid cell on the enlarged image.
    cell = size * zoom / grid

    # Shade every cell the model will be trained to mark as person in yellow.
    overlay = canvas.copy()
    for row, col in zip(*np.nonzero(target[..., 1])):
        top_left = (int(col * cell), int(row * cell))
        bottom_right = (int((col + 1) * cell), int((row + 1) * cell))
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 255), -1)
    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

    # Draw the grid lines in grey.
    for line in range(1, grid):
        position = int(line * cell)
        cv2.line(canvas, (position, 0), (position, size * zoom), (128, 128, 128), 1)
        cv2.line(canvas, (0, position), (size * zoom, position), (128, 128, 128), 1)

    # Draw each person's box in green and its centre point in red.
    for x, y, w, h in boxes:
        cv2.rectangle(
            canvas,
            (int(x * zoom), int(y * zoom)),
            (int((x + w) * zoom), int((y + h) * zoom)),
            (0, 255, 0), 2,
        )
        centre = (int((x + w / 2) * zoom), int((y + h / 2) * zoom))
        cv2.circle(canvas, centre, 5, (0, 0, 255), -1)

    # Return the finished image.
    return canvas


def make_sheet(data, indices, title):
    """Draw the chosen images and arrange them into one sheet."""

    # Draw every chosen image.
    tiles = []
    for index in indices:
        image, boxes, target = data.load_item(index)
        tiles.append(draw(image, boxes, target, data.grid))

    # Join the images into rows, then stack the rows into one sheet.
    rows = [
        np.hstack(tiles[start:start + images_per_row])
        for start in range(0, len(tiles), images_per_row)
    ]
    sheet = np.vstack(rows)

    # Save the sheet as a PNG file.
    path = output_dir / f"{title}.png"
    cv2.imwrite(str(path), sheet)
    print(f"Saved {path}")


# --------------------------------------------------
# 3. MAKE THE SHEETS
# --------------------------------------------------

# Create the folder for the sheets.
output_dir.mkdir(exist_ok=True)

# Load the training manifest.
items = load_manifest("train")

# Pick the same random images every run.
indices = random.Random(0).sample(range(len(items)), number_of_images)

# Sheet 1: the images as the model sees them during validation (stretched to 96 x 96).
make_sheet(FomoData(items, size=size, train=False), indices, "data_check_plain")

# Sheet 2: the same images after the random training augmentation.
make_sheet(FomoData(items, size=size, train=True), indices, "data_check_augmented")
