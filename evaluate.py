
# Import argparse so the model file and threshold can be chosen from the command line.
import argparse

# Import random so the same example images are drawn every run.
import random

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import OpenCV to draw the example predictions.
import cv2

# Import NumPy for the calculations.
import numpy as np

# Import SciPy's image tools to group neighbouring person cells together.
from scipy import ndimage

# Import the data loader.
from dataset import FomoData, keras, load_manifest


# --------------------------------------------------
# 1. TURNING GRIDS INTO PERSON CENTRES
# --------------------------------------------------

def decode(prediction, threshold=0.5):
    """Turn one predicted grid into a list of person centres (row, col) in cells.

    Cells whose person probability is above the threshold are marked. Marked cells
    that touch each other are treated as one person, and the centre of that group,
    weighted by probability, is returned.
    """

    # Retrieve the person probability of every cell.
    person = prediction[..., 1]

    # Mark the cells above the threshold.
    mask = person > threshold

    # Give each group of touching marked cells its own number.
    groups, number_of_groups = ndimage.label(mask)

    # Return no people if nothing is marked.
    if number_of_groups == 0:
        return []

    # Find the probability-weighted centre of each group.
    centres = ndimage.center_of_mass(person, groups, range(1, number_of_groups + 1))

    # Add 0.5 so a centre is measured from the middle of its cell, like the labels.
    return [(row + 0.5, col + 0.5) for row, col in centres]


def label_centres(boxes, size, grid):
    """Return the centre (row, col) in cells of each labelled person's pixel box."""

    # Convert each box centre from pixels to grid cells.
    return [
        ((y + h / 2) / size * grid, (x + w / 2) / size * grid)
        for x, y, w, h in boxes
    ]


# --------------------------------------------------
# 2. MATCHING PREDICTIONS TO PEOPLE
# --------------------------------------------------

def match(predicted, labelled, radius=1.5):
    """Strict score: count true positives, false positives and false negatives.

    A prediction is correct if it is within radius cells of a labelled person's
    centre who has not already been matched to another prediction.
    """

    # Copy the labels so matched people can be removed.
    remaining = list(labelled)

    # Count the correct predictions.
    true_positives = 0

    # Check each prediction against the labelled people still unmatched.
    for row, col in predicted:
        if remaining:

            # Measure the distance to each remaining labelled person.
            distances = [np.hypot(row - r, col - c) for r, c in remaining]

            # Find the closest one.
            closest = int(np.argmin(distances))

            # Count a correct prediction if it is close enough, and remove that person.
            if distances[closest] <= radius:
                true_positives += 1
                remaining.pop(closest)

    # Predictions not matched are false alarms; people not matched were missed.
    return true_positives, len(predicted) - true_positives, len(remaining)


def match_inside_box(predicted, boxes, size, grid):
    """Box score: count true positives, false positives and false negatives.

    A prediction is correct if it lies inside a labelled person's box that has not
    already been matched. For following, this is what matters: the person was found
    and their position is known, even if the dot is on their head rather than torso.
    """

    # Copy the boxes so matched people can be removed.
    remaining = list(boxes)

    # Count the correct predictions.
    true_positives = 0

    # Check each prediction against the boxes still unmatched.
    for row, col in predicted:

        # Convert the prediction from grid cells to pixels.
        px, py = col / grid * size, row / grid * size

        # Find the unmatched boxes that contain the prediction.
        containing = [
            i for i, (x, y, w, h) in enumerate(remaining)
            if x <= px <= x + w and y <= py <= y + h
        ]

        # If several boxes overlap, match the one whose centre is closest.
        if containing:
            closest = min(
                containing,
                key=lambda i: np.hypot(
                    px - (remaining[i][0] + remaining[i][2] / 2),
                    py - (remaining[i][1] + remaining[i][3] / 2),
                ),
            )
            true_positives += 1
            remaining.pop(closest)

    # Predictions not matched are false alarms; people not matched were missed.
    return true_positives, len(predicted) - true_positives, len(remaining)


def scores(tp, fp, fn):
    """Return precision, recall and F1 from the three counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-9)


# --------------------------------------------------
# 3. EVALUATING A WHOLE DATASET
# --------------------------------------------------

def evaluate(predict_fn, data, threshold=0.5):
    """Evaluate a model over every batch of data and return a dictionary of metrics.

    predict_fn takes a batch of images and returns the predicted grids. Passing a
    function means the same code can later test the int8 TFLite model.
    """

    # Count true positives, false positives and false negatives for both scores.
    strict = np.zeros(3, dtype=int)
    box = np.zeros(3, dtype=int)

    # Count images with people, and how many of them had at least one person found.
    frames_with_people = frames_with_people_found = 0

    # Count images without people, and how many of them had a false detection.
    empty_frames = empty_frames_false_alarm = 0

    # Loop through every batch.
    for batch_index in range(len(data)):

        # Load the images and boxes, and predict the grids.
        images, _, batch_boxes = data.load_batch(batch_index)
        predictions = predict_fn(images)

        # Compare the prediction for each image with its labels.
        for prediction, boxes in zip(predictions, batch_boxes):
            predicted = decode(prediction, threshold)
            strict += match(predicted, label_centres(boxes, data.size, data.grid))
            box_counts = match_inside_box(predicted, boxes, data.size, data.grid)
            box += box_counts

            # Update the per-image counts, using the box score.
            if boxes:
                frames_with_people += 1
                frames_with_people_found += box_counts[0] > 0
            else:
                empty_frames += 1
                empty_frames_false_alarm += len(predicted) > 0

    # Calculate precision, recall and F1 for both scores.
    precision, recall, f1 = scores(*strict)
    box_precision, box_recall, box_f1 = scores(*box)

    # Return the metrics.
    return {
        # Strict score: prediction within 1.5 cells of the box centre.
        "precision": precision,
        "recall": recall,
        "f1": f1,

        # Box score: prediction anywhere inside the person's box.
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": box_f1,

        # Share of images with people where at least one person was found.
        "detection_rate": frames_with_people_found / max(frames_with_people, 1),

        # Share of empty images where the model wrongly detected someone.
        "false_alarm_rate": empty_frames_false_alarm / max(empty_frames, 1),
    }


# --------------------------------------------------
# 4. EXAMPLE PREDICTION IMAGES
# --------------------------------------------------

def save_examples(model, data, path, threshold=0.5, number_of_images=20, zoom=3):
    """Save a sheet of images with labelled boxes and centres (green) and predictions (red)."""

    # Pick the same random validation images every run.
    indices = random.Random(0).sample(range(len(data.items)), number_of_images)

    # Create a list for the drawn images.
    tiles = []

    # Draw each chosen image.
    for index in indices:

        # Load the image and its boxes, and predict its grid.
        image, boxes, _ = data.load_item(index)
        prediction = model(image[None].astype(np.float32), training=False).numpy()[0]

        # Enlarge the image and convert it from RGB to BGR for OpenCV.
        canvas = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        # Calculate the width of one grid cell on the enlarged image.
        cell = canvas.shape[0] / data.grid

        # Draw the labelled boxes as thin green rectangles.
        for x, y, w, h in boxes:
            cv2.rectangle(canvas, (int(x * zoom), int(y * zoom)),
                          (int((x + w) * zoom), int((y + h) * zoom)), (0, 255, 0), 1)

        # Draw the labelled centres as green rings.
        for row, col in label_centres(boxes, data.size, data.grid):
            cv2.circle(canvas, (int(col * cell), int(row * cell)), 10, (0, 255, 0), 2)

        # Draw the predicted centres as red dots.
        for row, col in decode(prediction, threshold):
            cv2.circle(canvas, (int(col * cell), int(row * cell)), 5, (0, 0, 255), -1)

        # Add the drawn image to the list.
        tiles.append(canvas)

    # Arrange the images in rows of five and save the sheet.
    sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
    cv2.imwrite(str(path), sheet)


# --------------------------------------------------
# 5. EVALUATE A SAVED MODEL WHEN RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    # Read the command-line options.
    parser = argparse.ArgumentParser(description="Evaluate a trained FOMO model on COCO val.")
    parser.add_argument("--model", default="models/fomo_96_a035_best.h5")
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()

    # Load the trained model. compile=False skips the custom loss, which is not needed.
    model_path = Path(__file__).parent / args.model
    model = keras.models.load_model(model_path, compile=False)

    # Load the validation data.
    data = FomoData(load_manifest("val"), size=args.size, train=False)

    # Use the model directly, which is faster than model.predict for single batches.
    predict_fn = lambda images: model(images, training=False).numpy()

    # Try a range of thresholds and print the metrics for each.
    print(f"\n{'':5} {'strict (within 1.5 cells)':^20} {'box (inside the box)':^20}")
    print(f"{'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6} {'prec':>6} {'rec':>6} {'f1':>6}"
          f" {'detect':>7} {'f.alarm':>8}")
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.3, 0.91, 0.1):
        m = evaluate(predict_fn, data, threshold)
        print(f"{threshold:5.1f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
              f" {m['box_precision']:6.3f} {m['box_recall']:6.3f} {m['box_f1']:6.3f}"
              f" {m['detection_rate']:7.3f} {m['false_alarm_rate']:8.3f}")
        if m["box_f1"] > best_f1:
            best_threshold, best_f1 = threshold, m["box_f1"]

    # Report the best threshold, chosen by the box score.
    print(f"\nBest threshold: {best_threshold:.1f} (box F1 {best_f1:.3f})")

    # Save example predictions at the best threshold, including any failures.
    examples_path = Path(__file__).parent / "checks" / f"{model_path.stem}_val_predictions.png"
    examples_path.parent.mkdir(exist_ok=True)
    save_examples(model, data, examples_path, best_threshold)
    print(f"Saved {examples_path}")
