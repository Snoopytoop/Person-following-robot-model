# Import json so we can read and write COCO annotation files.
import json

# Import Path to make it easier to work with file and folder paths.
from pathlib import Path

# Import NumPy to create and save the training target arrays.
import numpy as np


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Get the folder containing this Python script.
# This should be your Dataset folder.
dataset_dir = Path(__file__).parent

# Set the path to the folder containing the COCO validation images.
images_dir = dataset_dir / "val2017"

# Set the path to the original COCO validation annotations.
annotations_file = (
    dataset_dir / "annotations" / "instances_val2017.json"
)

# Create a folder to store our prepared data.
output_dir = dataset_dir / "prepared"

# Create the output folder if it doesn't already exist.
# exist_ok=True prevents an error if the folder already exists.
output_dir.mkdir(exist_ok=True)

# Set the width and height of the output grid.
# A value of 20 means the grid will contain 20 rows and 20 columns.
grid_size = 20


# --------------------------------------------------
# 2. LOAD COCO ANNOTATIONS
# --------------------------------------------------

# Open the original COCO annotation file in read mode.
with open(annotations_file, "r") as f:

    # Convert the JSON file into a Python dictionary.
    # This lets us access its images, annotations and categories.
    coco = json.load(f)

# Search the category list for the category named "person".
# COCO normally assigns person the category ID 1.
person_id = next(
    category["id"]
    for category in coco["categories"]
    if category["name"] == "person"
)

# Create a dictionary that will store annotations grouped by image ID.
# Each image ID will point to a list of its annotations.
annotations_by_image = {}

# Loop through every annotation in the COCO dataset.
for ann in coco["annotations"]:

    # Get the ID of the image that this annotation belongs to.
    image_id = ann["image_id"]

    # If this image ID has not been seen before, create an empty list.
    annotations_by_image.setdefault(image_id, [])

    # Add this annotation to the list for that image.
    annotations_by_image[image_id].append(ann)


# --------------------------------------------------
# 3. SELECT SUITABLE IMAGES
# --------------------------------------------------

# Create a list to hold the images we want to keep.
selected_images = []

# Create a list to hold the person annotations for those images.
selected_person_annotations = []

# Count the images containing exactly one annotated person.
positive_count = 0

# Count the images with no annotations.
negative_count = 0

# Examine every image listed in the COCO dataset.
for image in coco["images"]:

    # Find all annotations belonging to this image.
    # If there are none, return an empty list.
    annotations = annotations_by_image.get(image["id"], [])

    # If the image has no annotations, treat it as a background example.
    if len(annotations) == 0:

        # Keep this image in the prepared dataset.
        selected_images.append(image)

        # Increase the background-image count by one.
        negative_count += 1

        # Skip the remaining checks and move to the next image.
        continue

    # Check whether any annotation marks a crowd region.
    # Crowd annotations need special treatment in training.
    if any(ann.get("iscrowd", 0) == 1 for ann in annotations):

        # Exclude this image if any of its annotations are crowd regions.
        continue

    # Keep an image only if it has exactly one annotation,
    # and that annotation belongs to the person category.
    if (
        len(annotations) == 1
        and annotations[0]["category_id"] == person_id
    ):

        # Add the suitable image to our selected image list.
        selected_images.append(image)

        # Save the annotation describing the person in this image.
        selected_person_annotations.append(annotations[0])

        # Increase the positive-image count by one.
        positive_count += 1


# --------------------------------------------------
# 4. CREATE A FILTERED COCO ANNOTATION FILE
# --------------------------------------------------

# Copy the general dataset information from the original COCO file.
# Exclude the images, annotations and categories because we will replace them.
filtered_coco = {
    key: value
    for key, value in coco.items()
    if key not in {"images", "annotations", "categories"}
}

# Store only the selected images in the new annotation file.
filtered_coco["images"] = selected_images

# Store only the person annotations we selected.
filtered_coco["annotations"] = selected_person_annotations

# Store only the person category.
filtered_coco["categories"] = [
    category
    for category in coco["categories"]
    if category["id"] == person_id
]

# Open a new JSON file for writing the filtered annotations.
with open(
    output_dir / "instances_val2017_simple.json", "w"
) as f:

    # Write the filtered dataset to the file.
    # The indentation makes the JSON easier for humans to read.
    json.dump(filtered_coco, f, indent=2)


# --------------------------------------------------
# 5. GENERATE FOMO-STYLE GRID TARGETS
# --------------------------------------------------

# Create a NumPy array with one target grid per selected image.
# Its shape is:
# (number of images, grid rows, grid columns, number of classes).
# We use two channels: background and person.
targets = np.zeros(
    (len(selected_images), grid_size, grid_size, 2),
    dtype=np.uint8,
)

# Initially mark every grid cell as background.
# Channel 0 represents background; channel 1 represents a person.
targets[:, :, :, 0] = 1

# Loop through the selected images in order.
# The index i links each image to its matching target grid.
for i, image in enumerate(selected_images):

    # Retrieve this image's annotations.
    annotations = annotations_by_image.get(image["id"], [])

    # If the image has no annotations, leave the entire grid as background.
    if len(annotations) == 0:
        continue

    # This image has exactly one person annotation.
    # Retrieve that annotation.
    ann = annotations[0]

    # Get the bounding box as x, y, width and height.
    # x and y are the top-left corner in original image pixels.
    x, y, width, height = ann["bbox"]

    # Calculate the horizontal centre of the person.
    center_x = x + width / 2

    # Calculate the vertical centre of the person.
    center_y = y + height / 2

    # Convert the horizontal centre into a grid column.
    # Dividing by the image width gives a relative position.
    # Multiplying by grid_size converts it into a grid coordinate.
    col = int(center_x / image["width"] * grid_size)

    # Convert the vertical centre into a grid row.
    row = int(center_y / image["height"] * grid_size)

    # Ensure the column index is within the valid grid boundaries.
    col = min(max(col, 0), grid_size - 1)

    # Ensure the row index is within the valid grid boundaries.
    row = min(max(row, 0), grid_size - 1)

    # Change the selected cell from background to person.
    # Set its background value to 0.
    targets[i, row, col, 0] = 0

    # Set its person value to 1.
    targets[i, row, col, 1] = 1


# --------------------------------------------------
# 6. SAVE THE IMAGE PATHS AND TARGETS
# --------------------------------------------------

# Create a list of relative paths to the selected images.
# Each path corresponds to the target at the same array index.
image_paths = [
    str(Path("val2017") / image["file_name"])
    for image in selected_images
]

# Save all target grids in a NumPy file.
# The .npy format preserves the array's shape and data type.
np.save(output_dir / "val_targets.npy", targets)

# Open a JSON file to save the image paths.
with open(output_dir / "val_image_paths.json", "w") as f:

    # Save the paths as a readable JSON list.
    json.dump(image_paths, f, indent=2)


# --------------------------------------------------
# 7. PRINT A SUMMARY
# --------------------------------------------------

# Print how many images contain exactly one annotated person.
print("Images containing exactly one person:", positive_count)

# Print how many images contain no annotations.
print("Background-only images:", negative_count)

# Print the total number of selected images.
print("Total selected images:", len(selected_images))

# Print the shape of the target array.
print("Target shape:", targets.shape)

# Print the folder where the generated files were saved.
print("Output directory:", output_dir)
