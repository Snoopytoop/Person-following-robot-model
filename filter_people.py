
# Import json so we can read and write COCO annotation files and save image paths.
import json

# Import defaultdict so we can automatically create an empty list for each new image ID.
from collections import defaultdict

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import NumPy to create and save the arrays containing our training targets.
import numpy as np


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Get the directory containing this Python script.
# All dataset paths below will be constructed relative to this directory.
dataset_dir = Path(__file__).parent

# Set the height and width of the target grid to 20 cells.
# This must match the output grid expected by the model we eventually train.
grid_size = 20

# Set the maximum number of annotated people allowed in a positive image.
# Images containing more people than this will be excluded.
max_people = 5

# Create a dictionary containing the configuration for each dataset split.
datasets = {
    # Configure the training dataset.
    "train": {
        # Set the directory containing the COCO training images.
        "image_dir": dataset_dir / "train2017",

        # Set the path to the COCO training annotation file.
        "annotation_file": dataset_dir / "annotations" / "instances_train2017.json",

        # Set the directory where the prepared training files will be saved.
        "output_dir": dataset_dir / "prepared" / "train",
    },

    # Configure the validation dataset, which is named "test" here.
    "test": {
        # Set the directory containing the COCO validation images.
        "image_dir": dataset_dir / "val2017",

        # Set the path to the COCO validation annotation file.
        "annotation_file": dataset_dir / "annotations" / "instances_val2017.json",

        # Set the directory where the prepared validation files will be saved.
        "output_dir": dataset_dir / "prepared" / "test",
    },
}


# --------------------------------------------------
# 2. PROCESS EACH DATASET
# --------------------------------------------------

# Loop through the training and validation dataset configurations.
# dataset_name will be "train" or "test", and config will contain its paths.
for dataset_name, config in datasets.items():

    # Print which dataset is currently being processed.
    print(f"\nProcessing {dataset_name} dataset...")

    # Retrieve the directory containing this dataset's images.
    image_dir = config["image_dir"]

    # Retrieve the path to this dataset's COCO annotation file.
    annotation_file = config["annotation_file"]

    # Retrieve the directory where this dataset's output files will be saved.
    output_dir = config["output_dir"]

    # Create the output directory and any missing parent directories.
    # exist_ok=True prevents an error if the directory already exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the COCO annotation file in read mode.
    with open(annotation_file, "r") as file:

        # Load the JSON contents and convert them into Python dictionaries and lists.
        coco = json.load(file)

    # Find the category whose name is "person".
    # COCO assigns a numeric category ID to each object category.
    person_categories = [
        category
        for category in coco["categories"]
        if category["name"] == "person"
    ]

    # Stop execution if the annotation file does not contain a person category.
    if not person_categories:

        # Raise an error explaining which annotation file has the problem.
        raise ValueError(
            f"No person category found in {annotation_file}"
        )

    # Retrieve the dictionary describing the person category.
    # COCO's person category ID is commonly 1, but we retrieve it rather than assuming.
    person_category = person_categories[0]

    # Store the numeric category ID used for people in this dataset.
    person_id = person_category["id"]

    # Create a dictionary that will group annotations according to their image ID.
    # defaultdict(list) automatically creates an empty list for a new image ID.
    annotations_by_image = defaultdict(list)

    # Loop through every object annotation in the original COCO dataset.
    for annotation in coco["annotations"]:

        # Get the ID of the image containing this annotated object.
        image_id = annotation["image_id"]

        # Add this annotation to the list belonging to its image.
        annotations_by_image[image_id].append(annotation)

    # Create a list that will contain the image records we decide to keep.
    selected_images = []

    # Create a list that will contain the person annotations for selected images.
    selected_person_annotations = []

    # Create counters to record why images are included or excluded.
    counts = {

        # Count images with no annotations.
        "negative": 0,

        # Count images containing between one and max_people people.
        "positive": 0,

        # Count images containing more than max_people annotated people.
        "too_many_people": 0,

        # Count images containing at least one crowd annotation.
        "crowd": 0,

        # Count other images that do not meet the selection rules.
        "other": 0,
    }

    # --------------------------------------------------
    # 3. SELECT SUITABLE IMAGES
    # --------------------------------------------------

    # Loop through every image record listed in the COCO annotation file.
    # This processes the images listed in the JSON, not every file physically in the folder.
    for image in coco["images"]:

        # Retrieve the unique ID assigned to the current image.
        image_id = image["id"]

        # Retrieve all annotations belonging to this image.
        # If the image has no annotations, return an empty list.
        annotations = annotations_by_image.get(image_id, [])

        # Check whether the image has no annotations at all.
        if len(annotations) == 0:

            # Keep the image as a background-only training example.
            selected_images.append(image)

            # Increase the number of selected background images by one.
            counts["negative"] += 1

            # Move to the next image because no further checks are needed.
            continue

        # Check whether any annotation in this image is marked as a crowd.
        # The COCO iscrowd field is normally 1 for crowd annotations and 0 otherwise.
        has_crowd = any(
            annotation.get("iscrowd", 0) == 1
            for annotation in annotations
        )

        # Exclude the entire image if at least one crowd annotation is present.
        if has_crowd:

            # Increase the number of excluded crowd images by one.
            counts["crowd"] += 1

            # Skip the remaining processing for this image.
            continue

        # Create a list containing only annotations belonging to people.
        # Other categories, such as cars or bicycles, are ignored.
        person_annotations = [
            annotation
            for annotation in annotations
            if annotation["category_id"] == person_id
        ]

        # Count the number of annotated people in the current image.
        number_of_people = len(person_annotations)

        # Keep images containing at least one person and no more than max_people.
        if 1 <= number_of_people <= max_people:

            # Add the image record to the list of selected images.
            selected_images.append(image)

            # Add every person's annotation from this image to our annotation list.
            # This is important because an image may contain several people.
            selected_person_annotations.extend(
                person_annotations
            )

            # Increase the number of selected positive images by one.
            counts["positive"] += 1

        # Check whether the image contains too many annotated people.
        elif number_of_people > max_people:

            # Record that this image was excluded for having too many people.
            counts["too_many_people"] += 1

        # Handle images that have annotations but no person annotations.
        else:

            # Record that this image was excluded for not meeting the person criteria.
            # For example, it might contain only a car or a bicycle.
            counts["other"] += 1

    # --------------------------------------------------
    # 4. CREATE FILTERED COCO ANNOTATIONS
    # --------------------------------------------------

    # Create a new COCO dictionary containing only the selected images and people.
    filtered_coco = {

        # Preserve the general dataset information from the original COCO file.
        "info": coco.get("info", {}),

        # Preserve the dataset's licence information, if available.
        "licenses": coco.get("licenses", []),

        # Store only the image records selected by our filtering rules.
        "images": selected_images,

        # Store only person annotations belonging to selected positive images.
        "annotations": selected_person_annotations,

        # Store only the person category because other categories are not used.
        "categories": [person_category],
    }

    # Construct the output path for the filtered COCO annotation file.
    filtered_json_path = (
        output_dir / f"{dataset_name}_filtered_annotations.json"
    )

    # Open the output JSON file in write mode.
    with open(filtered_json_path, "w") as file:

        # Write the filtered COCO dictionary into the output file.
        json.dump(filtered_coco, file)

    # --------------------------------------------------
    # 5. GENERATE FOMO CENTRE-GRID TARGETS
    # --------------------------------------------------

    # Count the number of selected images.
    number_of_images = len(selected_images)

    # Create a NumPy array for the model's training targets.
    # Its dimensions are:
    # number of images x grid rows x grid columns x two classes.
    # Channel 0 represents background, and channel 1 represents person centres.
    # uint8 stores each target value as an unsigned 8-bit integer.
    targets = np.zeros(
        (number_of_images, grid_size, grid_size, 2),
        dtype=np.uint8,
    )

    # Initially mark every grid cell as background.
    # Setting channel 0 to 1 represents the background class.
    targets[:, :, :, 0] = 1

    # Create a dictionary mapping each selected image ID to its array index.
    # This allows us to find the correct target array for each annotation.
    image_index = {
        image["id"]: index
        for index, image in enumerate(selected_images)
    }

    # Create a dictionary that records the grid cells already assigned
    # to person centres in each image.
    # Sets allow us to check efficiently whether a cell has been used.
    centre_cells_by_image = defaultdict(set)

    # Initialise a counter for cases where two or more people map to the same cell.
    overlapping_centres = 0

    # Loop through every selected person annotation.
    for annotation in selected_person_annotations:

        # Retrieve the ID of the image containing this person.
        img_id = annotation["image_id"]

        # Find the index of that image in the selected_images list.
        index = image_index[img_id]

        # Retrieve the person's bounding box in COCO format:
        # [x-coordinate, y-coordinate, width, height].
        x, y, width, height = annotation["bbox"]

        # Calculate the horizontal centre of the bounding box.
        centre_x = x + width / 2

        # Calculate the vertical centre of the bounding box.
        centre_y = y + height / 2

        # Retrieve the image's width and height.
        # These values are used to scale the centre into grid coordinates.
        image = selected_images[index]

        # Convert the horizontal pixel coordinate into a grid column.
        # For example, a centre halfway across the image maps near column 10.
        col = int(centre_x / image["width"] * grid_size)

        # Convert the vertical pixel coordinate into a grid row.
        # The calculation is scaled relative to the image's height.
        row = int(centre_y / image["height"] * grid_size)

        # Ensure the column is within the valid grid range.
        # Valid column indices are from 0 to grid_size - 1.
        col = max(0, min(grid_size - 1, col))

        # Ensure the row is within the valid grid range.
        # Valid row indices are from 0 to grid_size - 1.
        row = max(0, min(grid_size - 1, row))

        # Represent the person's centre cell as a tuple of (row, column).
        cell = (row, col)

        # Check whether another person in this image already uses this cell.
        if cell in centre_cells_by_image[img_id]:

            # Increase the collision counter if the cell is already occupied.
            # A single cell cannot represent two distinct person centres.
            overlapping_centres += 1

        # Record that this cell contains a person centre in this image.
        centre_cells_by_image[img_id].add(cell)

        # Set the background channel to 0 at the person's centre cell.
        # This removes the background label from that cell.
        targets[index, row, col, 0] = 0

        # Set the person channel to 1 at the person's centre cell.
        # The resulting target at this cell is [0, 1].
        targets[index, row, col, 1] = 1

    # --------------------------------------------------
    # 6. SAVE TARGETS AND IMAGE PATHS
    # --------------------------------------------------

    # Construct the output path for the NumPy training-target array.
    targets_path = output_dir / f"{dataset_name}_targets.npy"

    # Save the target array so it can be loaded later during model training.
    np.save(targets_path, targets)

    # Create a list containing the file path of each selected image.
    # The list order matches the first dimension of the targets array.
    image_paths = [
        str(image_dir / image["file_name"])
        for image in selected_images
    ]

    # Construct the output path for the list of image paths.
    image_paths_path = (
        output_dir / f"{dataset_name}_image_paths.json"
    )

    # Open the image-path JSON file in write mode.
    with open(image_paths_path, "w") as file:

        # Save the image paths with indentation to make the JSON easier to read.
        json.dump(image_paths, file, indent=2)

    # --------------------------------------------------
    # 7. PRINT SUMMARY
    # --------------------------------------------------

    # Print a heading identifying the dataset that was processed.
    print(f"\n{dataset_name.upper()} SUMMARY")

    # Print a line to visually separate the heading from the results.
    print("-" * 40)

    # Print the total number of selected images.
    print(f"Selected images:          {len(selected_images)}")

    # Print the number of positive images containing one to max_people people.
    print(f"  - Single/multi-person:  {counts['positive']}")

    # Print the number of selected background-only images.
    print(f"  - Background-only:      {counts['negative']}")

    # Print the number of images excluded because they contained crowd annotations.
    print(f"Excluded crowd images:    {counts['crowd']}")

    # Print the number of images excluded for containing too many people.
    print(f"Excluded over {max_people} people: {counts['too_many_people']}")

    # Print the number of images excluded for containing no annotated people.
    print(f"Excluded other images:    {counts['other']}")

    # Print the total number of person annotations retained.
    # This may be greater than the number of positive images.
    print(f"Person annotations:       {len(selected_person_annotations)}")

    # Print the number of person-centre collisions detected in the grid.
    print(f"Centre-cell collisions:   {overlapping_centres}")

    # Print the dimensions of the saved target array.
    print(f"Targets shape:            {targets.shape}")

    # Print the output directory containing the generated files.
    print(f"Saved to:                 {output_dir}")
