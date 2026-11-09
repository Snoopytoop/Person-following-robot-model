
# Import os so we can set environment variables before the libraries are imported.
import os

# Use legacy Keras (tf_keras) so quantisation-aware training remains possible later.
# This must be set before TensorFlow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Stop Albumentations checking online for a newer version every time it is imported.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Import json so we can read the manifest files.
import json

# Import math to round up the number of validation batches.
import math

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import Albumentations to augment images while keeping the boxes in the right place.
import albumentations as A

# Import OpenCV to read images from disk.
import cv2

# Import NumPy to build the image batches and target grids.
import numpy as np

# Import TensorFlow, which provides the Keras Sequence class used below.
import tensorflow as tf

# Use the legacy Keras API provided through TensorFlow.
keras = tf.keras


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Get the Dataset folder, which sits next to this script.
# The image paths in the manifests are stored relative to this folder.
dataset_dir = Path(__file__).parent / "Dataset"

# Set how much the backbone shrinks the image.
# MobileNetV2 cut at block_6_expand_relu reduces the image by 8 times,
# so a 96 pixel input gives a 12 x 12 output grid.
model_stride = 8

# Set the size of the central region of each person marked as "person",
# as a fraction of the box width and height. 1/3 marks the middle third.
centre_region = 1 / 3


# --------------------------------------------------
# 2. LOADING THE MANIFEST
# --------------------------------------------------

def load_manifest(split):
    """Load the manifest for "train" or "val" and return its list of entries.

    Each entry is {"path": full image path, "boxes": [[x, y, w, h], ...]},
    with each box written as fractions of the image width and height.
    """

    # Construct the path to the manifest created by filter_people.py.
    manifest_path = dataset_dir / "prepared" / split / f"{split}_manifest.json"

    # Open the manifest file in read mode.
    with open(manifest_path, "r") as file:

        # Load the list of entries.
        items = json.load(file)

    # Convert each relative image path into a full path.
    for item in items:
        item["path"] = str(dataset_dir / item["path"])

    # Return the list of entries.
    return items


# --------------------------------------------------
# 3. BUILDING THE TARGET GRID
# --------------------------------------------------

def boxes_to_grid(boxes, size, grid):
    """Convert pixel boxes on a size x size image into a grid x grid x 2 target.

    Channel 0 is background and channel 1 is person. Every cell is background
    except the cells in the central region of each person's box. Marking a region
    rather than a single cell means neighbouring cells on the same body agree,
    which is much easier to learn for large people.
    """

    # Create the target grid and mark every cell as background.
    target = np.zeros((grid, grid, 2), dtype=np.float32)
    target[..., 0] = 1

    # Calculate the pixel position of the middle of every grid cell.
    cell_size = size / grid
    cell_middles = (np.arange(grid) + 0.5) * cell_size

    # Loop through each person's box in COCO pixel format [x, y, width, height].
    for x, y, width, height in boxes:

        # Find the centre of the box in pixels.
        centre_x = x + width / 2
        centre_y = y + height / 2

        # Find the edges of the central region of the box.
        half_width = width * centre_region / 2
        half_height = height * centre_region / 2

        # Select the columns and rows whose cell middles lie inside the central region.
        cols = np.nonzero(np.abs(cell_middles - centre_x) <= half_width)[0]
        rows = np.nonzero(np.abs(cell_middles - centre_y) <= half_height)[0]

        # Mark those cells as person: background 0, person 1.
        target[rows[:, None], cols[None, :]] = [0, 1]

        # Always mark the cell containing the centre, in case the region is
        # smaller than one cell. min() keeps an edge centre inside the grid.
        col = min(int(centre_x / cell_size), grid - 1)
        row = min(int(centre_y / cell_size), grid - 1)
        target[row, col] = [0, 1]

    # Return the finished target grid.
    return target


# --------------------------------------------------
# 4. AUGMENTATION
# --------------------------------------------------

def make_augmentation(size, train):
    """Create the image transform used for training or validation."""

    # Describe how the boxes are passed to Albumentations.
    box_params = A.BboxParams(

        # Boxes are in COCO pixel format [x, y, width, height].
        format="coco",

        # Drop a box if less than 30% of the person remains visible after cropping.
        min_visibility=0.3,

        # Name of the list holding one label per box (all people, so all 0).
        label_fields=["labels"],
    )

    # Validation images are only stretched to a square, with no randomness.
    # The board must stretch camera frames in exactly the same way.
    if not train:
        return A.Compose([A.Resize(size, size)], bbox_params=box_params)

    # Training images receive random changes so the model generalises better.
    return A.Compose([

        # Crop a random 40-100% of the image and stretch it to size x size.
        # This shows people at different distances and positions in the frame.
        A.RandomResizedCrop(size=(size, size), scale=(0.4, 1.0), ratio=(0.75, 1.33)),

        # Mirror the image left to right half of the time.
        A.HorizontalFlip(p=0.5),

        # Change the brightness and contrast to imitate different lighting.
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),

        # Change the colours slightly to imitate different cameras and lighting.
        A.HueSaturationValue(p=0.3),

        # Sometimes blur the image or add noise, like a cheap or moving camera.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.4),

        # Sometimes add JPEG compression artefacts.
        A.ImageCompression(quality_range=(40, 90), p=0.3),

    ], bbox_params=box_params)


# --------------------------------------------------
# 5. THE DATA LOADER
# --------------------------------------------------

class FomoData(keras.utils.Sequence):
    """Supplies batches of (images, target grids) to Keras during training."""

    def __init__(self, items, size=96, batch_size=64, train=True, stride=model_stride):

        # Initialise the parent Keras Sequence class.
        super().__init__()

        # Store the manifest entries, input size, batch size and mode.
        self.items = items
        self.size = size
        self.batch_size = batch_size
        self.train = train

        # Calculate the size of the output grid, e.g. 96 / 8 = 12.
        self.grid = size // stride

        # Create the training or validation transform.
        self.augmentation = make_augmentation(size, train)

        # Set the order in which the images will be used.
        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches in one epoch."""

        # Training drops the last incomplete batch so every batch has the same size.
        if self.train:
            return len(self.items) // self.batch_size

        # Validation keeps the last incomplete batch so every image is evaluated.
        return math.ceil(len(self.items) / self.batch_size)

    def on_epoch_end(self):
        """Shuffle the training images at the end of every epoch."""

        # Use a random order for training and a fixed order for validation.
        if self.train:
            self.order = np.random.permutation(len(self.items))
        else:
            self.order = np.arange(len(self.items))

    def load_item(self, index):
        """Load one image, transform it and return (image, pixel boxes, target)."""

        # Retrieve the manifest entry.
        item = self.items[index]

        # Read the image. OpenCV loads colours as BGR, so convert them to RGB.
        image = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)

        # Retrieve the image's height and width in pixels.
        height, width = image.shape[:2]

        # Create a list for the boxes converted to pixels.
        pixel_boxes = []

        # Loop through each box, written as fractions of the image size.
        for x, y, w, h in item["boxes"]:

            # Convert the box corners to pixels and clip them to the image edges.
            x0 = max(0.0, x * width)
            y0 = max(0.0, y * height)
            x1 = min(float(width), (x + w) * width)
            y1 = min(float(height), (y + h) * height)

            # Keep the box only if it is still at least one pixel wide and tall.
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                pixel_boxes.append([x0, y0, x1 - x0, y1 - y0])

        # Apply the transform to the image and its boxes together.
        result = self.augmentation(
            image=image, bboxes=pixel_boxes, labels=[0] * len(pixel_boxes)
        )

        # Retrieve the boxes after the transform as a plain list.
        boxes = [list(box) for box in result["bboxes"]]

        # Build the target grid from the transformed boxes.
        target = boxes_to_grid(boxes, self.size, self.grid)

        # Return the transformed image, its boxes and its target grid.
        return result["image"], boxes, target

    def load_batch(self, batch_index):
        """Return batch number batch_index as (images, targets, boxes).

        boxes is a list holding each image's pixel boxes, used by the evaluation.
        """

        # Select the image indices belonging to this batch.
        start = batch_index * self.batch_size
        indices = self.order[start:start + self.batch_size]

        # Load every image in the batch.
        loaded = [self.load_item(index) for index in indices]

        # Stack the images into one array with pixel values 0-255.
        # The model rescales the pixels itself, so no normalisation is done here.
        images = np.stack([image for image, _, _ in loaded]).astype(np.float32)

        # Stack the target grids into one array.
        targets = np.stack([target for _, _, target in loaded])

        # Collect each image's boxes.
        boxes = [image_boxes for _, image_boxes, _ in loaded]

        # Return the batch.
        return images, targets, boxes

    def __getitem__(self, batch_index):
        """Return batch number batch_index as (images, targets) for Keras."""

        # Load the batch and leave out the boxes, which Keras does not use.
        images, targets, _ = self.load_batch(batch_index)
        return images, targets
