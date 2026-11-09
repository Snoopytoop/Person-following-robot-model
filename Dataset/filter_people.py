
# Import json so we can read COCO annotation files and write the manifests.
import json

# Import math for the square root used in the person-size filter.
import math

# Import random so the background images can be sampled reproducibly.
import random

# Import defaultdict so we can automatically create an empty list for each new image ID.
from collections import defaultdict

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import OpenCV to read, resize and save the selected images.
import cv2

# Import tqdm to show a progress bar while the images are resized.
from tqdm import tqdm


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Get the directory containing this Python script.
# All dataset paths below will be constructed relative to this directory.
dataset_dir = Path(__file__).parent

# Set the maximum number of annotated people allowed in a positive image.
# Images containing more people than this will be excluded.
max_people = 5

# Set the minimum size of a person, measured as sqrt(box area / image area).
# 0.10 is roughly one grid cell at a 96 pixel input.
# Images containing any person smaller than this are excluded, because the model
# cannot see them at our input size and would learn to ignore people.
min_person_size = 0.10

# Set the fraction of the final dataset that should be background-only images.
negative_fraction = 0.15

# Set the length of the longest side of each saved image, in pixels.
# 192 is large enough to later train at 96, 128 or 160 pixel inputs.
max_side = 192

# Set the JPEG quality used when saving the resized images.
jpeg_quality = 92

# Set the random seed so the same background images are chosen every run.
seed = 0

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

    # Configure the validation dataset.
    # Our own robot recordings will be the test set later, so COCO val2017 is "val".
    "val": {
        # Set the directory containing the COCO validation images.
        "image_dir": dataset_dir / "val2017",

        # Set the path to the COCO validation annotation file.
        "annotation_file": dataset_dir / "annotations" / "instances_val2017.json",

        # Set the directory where the prepared validation files will be saved.
        "output_dir": dataset_dir / "prepared" / "val",
    },
}


# --------------------------------------------------
# 2. PROCESS EACH DATASET
# --------------------------------------------------

# Loop through the training and validation dataset configurations.
# dataset_name will be "train" or "val", and config will contain its paths.
for dataset_name, config in datasets.items():

    # Print which dataset is currently being processed.
    print(f"\nProcessing {dataset_name} dataset...")

    # Retrieve the directory containing this dataset's images.
    image_dir = config["image_dir"]

    # Retrieve the path to this dataset's COCO annotation file.
    annotation_file = config["annotation_file"]

    # Retrieve the directory where this dataset's output files will be saved.
    output_dir = config["output_dir"]

    # Set the directory where the resized images will be saved.
    resized_dir = output_dir / "images"

    # Create the output directories and any missing parent directories.
    # exist_ok=True prevents an error if the directory already exists.
    resized_dir.mkdir(parents=True, exist_ok=True)

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

    # Store the numeric category ID used for people in this dataset.
    # COCO's person category ID is commonly 1, but we retrieve it rather than assuming.
    person_id = person_categories[0]["id"]

    # Create a dictionary that will group person annotations according to their image ID.
    # Other categories, such as cars or bicycles, are not needed and are ignored.
    # defaultdict(list) automatically creates an empty list for a new image ID.
    people_by_image = defaultdict(list)

    # Loop through every object annotation in the original COCO dataset.
    for annotation in coco["annotations"]:

        # Keep only annotations belonging to people.
        if annotation["category_id"] == person_id:

            # Add this person annotation to the list belonging to its image.
            people_by_image[annotation["image_id"]].append(annotation)

    # Create a list that will contain the positive manifest entries.
    positives = []

    # Create a list that will contain every candidate background-only image.
    negative_candidates = []

    # Create counters to record why images are included or excluded.
    counts = {

        # Count images listed in the annotation file but missing from the image folder.
        "missing_file": 0,

        # Count images containing at least one crowd of people.
        "crowd": 0,

        # Count images containing more than max_people annotated people.
        "too_many_people": 0,

        # Count images containing at least one person below min_person_size.
        "tiny_person": 0,
    }

    # --------------------------------------------------
    # 3. SELECT SUITABLE IMAGES
    # --------------------------------------------------

    # Loop through every image record listed in the COCO annotation file.
    for image in coco["images"]:

        # Skip images listed in the annotations whose file is not in the image folder,
        # for example because the download or unzip did not finish.
        if not (image_dir / image["file_name"]).exists():

            # Record that this image file is missing.
            counts["missing_file"] += 1

            # Skip the remaining processing for this image.
            continue

        # Retrieve the image's width and height.
        # These are used to normalise the boxes and to measure person size.
        width, height = image["width"], image["height"]

        # Retrieve all person annotations belonging to this image.
        # If the image contains no people, return an empty list.
        people = people_by_image.get(image["id"], [])

        # Check whether the image contains no people at all.
        # This includes images showing only other objects, such as cars or animals,
        # which are useful examples of what is NOT a person.
        if len(people) == 0:

            # Keep the image as a candidate background-only example.
            negative_candidates.append(image)

            # Move to the next image because no further checks are needed.
            continue

        # Exclude the image if any person annotation is marked as a crowd.
        # Crowd annotations do not mark individual people, so their centres are unknown.
        if any(person["iscrowd"] == 1 for person in people):

            # Increase the number of excluded crowd images by one.
            counts["crowd"] += 1

            # Skip the remaining processing for this image.
            continue

        # Exclude the image if it contains too many annotated people.
        if len(people) > max_people:

            # Record that this image was excluded for having too many people.
            counts["too_many_people"] += 1

            # Skip the remaining processing for this image.
            continue

        # Exclude the image if any person is too small for the model to see.
        # Box width and height are at positions 2 and 3 of the COCO bbox.
        if any(
            math.sqrt(person["bbox"][2] * person["bbox"][3] / (width * height))
            < min_person_size
            for person in people
        ):

            # Record that this image was excluded for containing a tiny person.
            counts["tiny_person"] += 1

            # Skip the remaining processing for this image.
            continue

        # Convert each COCO box from pixels to normalised [x, y, width, height].
        # Normalised values (0 to 1) stay correct after the image is resized.
        boxes = [
            [x / width, y / height, w / width, h / height]
            for x, y, w, h in (person["bbox"] for person in people)
        ]

        # Add the image and its boxes to the list of positive examples.
        positives.append({"image": image, "boxes": boxes})

    # --------------------------------------------------
    # 4. SAMPLE BACKGROUND-ONLY IMAGES
    # --------------------------------------------------

    # Shuffle the background candidates using a fixed seed so the result is repeatable.
    random.Random(seed).shuffle(negative_candidates)

    # Calculate how many negatives are needed for them to make up negative_fraction
    # of the final dataset. For 15%, this is about 0.176 negatives per positive.
    number_of_negatives = int(
        len(positives) * negative_fraction / (1 - negative_fraction)
    )

    # Select the background images, limited by how many candidates exist.
    negatives = [
        {"image": image, "boxes": []}
        for image in negative_candidates[:number_of_negatives]
    ]

    # Combine positives and negatives, then shuffle them into a single list.
    selected = positives + negatives
    random.Random(seed).shuffle(selected)

    # --------------------------------------------------
    # 5. RESIZE IMAGES AND BUILD THE MANIFEST
    # --------------------------------------------------

    # Create the list of manifest entries that the training data loader will read.
    # The person-centre grid is NOT built here. It is built in the data loader after
    # augmentation, because cropping and flipping move the people in the image,
    # and because the grid size depends on the model's input size.
    manifest = []

    # Loop through every selected image, showing a progress bar.
    for item in tqdm(selected, desc=f"Resizing {dataset_name}"):

        # Retrieve the COCO file name of the image, e.g. "000000000139.jpg".
        file_name = item["image"]["file_name"]

        # Construct the path where the resized copy will be saved.
        resized_path = resized_dir / file_name

        # Only resize the image if it was not already resized in a previous run.
        if not resized_path.exists():

            # Read the original full-size image from the COCO folder.
            original = cv2.imread(str(image_dir / file_name))

            # Calculate the scale that makes the longest side equal to max_side.
            scale = max_side / max(original.shape[:2])

            # Shrink the image. INTER_AREA gives the best quality when downscaling.
            resized = cv2.resize(
                original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

            # Save the resized image as a JPEG.
            cv2.imwrite(
                str(resized_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )

        # Add the entry to the manifest.
        # The path is stored relative to this script's folder so the dataset can be
        # moved, for example to Colab, without editing the manifest.
        manifest.append({
            "path": str(resized_path.relative_to(dataset_dir)),
            "boxes": item["boxes"],
        })

    # Construct the output path for the manifest.
    manifest_path = output_dir / f"{dataset_name}_manifest.json"

    # Open the manifest file in write mode.
    with open(manifest_path, "w") as file:

        # Write the manifest entries into the output file.
        json.dump(manifest, file)

    # --------------------------------------------------
    # 6. PRINT SUMMARY
    # --------------------------------------------------

    # Count the total number of person boxes kept.
    number_of_boxes = sum(len(item["boxes"]) for item in manifest)

    # Print a heading identifying the dataset that was processed.
    print(f"\n{dataset_name.upper()} SUMMARY")

    # Print a line to visually separate the heading from the results.
    print("-" * 40)

    # Print the total number of selected images.
    print(f"Selected images:          {len(manifest)}")

    # Print the number of positive images containing one to max_people people.
    print(f"  - With people:          {len(positives)}")

    # Print the number of selected background-only images and their share.
    print(
        f"  - Background-only:      {len(negatives)} "
        f"({len(negatives) / len(manifest):.1%}, from {len(negative_candidates)} available)"
    )

    # Print the number of images skipped because their file was not found.
    print(f"Missing image files:      {counts['missing_file']}")

    # Print the number of images excluded because they contained crowd annotations.
    print(f"Excluded crowd images:    {counts['crowd']}")

    # Print the number of images excluded for containing too many people.
    print(f"Excluded over {max_people} people: {counts['too_many_people']}")

    # Print the number of images excluded for containing a tiny person.
    print(f"Excluded tiny people:     {counts['tiny_person']}")

    # Print the total number of person boxes retained.
    print(f"Person boxes:             {number_of_boxes}")

    # Print the path of the manifest.
    print(f"Saved to:                 {manifest_path}")
