
# Import argparse so the training settings can be chosen from the command line.
import argparse

# Import Path to create and manage file and folder paths.
from pathlib import Path

# Import the data loader, the model and the evaluation code.
from dataset import FomoData, keras, load_manifest
from evaluate import evaluate
from model import build_fomo, weighted_cross_entropy


# --------------------------------------------------
# 1. COMMAND-LINE OPTIONS
# --------------------------------------------------

# Describe the options that can be changed when running the script.
parser = argparse.ArgumentParser(description="Train the FOMO person detector on COCO.")
parser.add_argument("--size", type=int, default=96, help="input size in pixels")
parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width")
parser.add_argument("--centre-region", type=float, default=1 / 3,
                    help="fraction of each person's box marked as person (1 = whole box)")
parser.add_argument("--stride", type=int, default=8, choices=[8, 16],
                    help="how many times the backbone shrinks the image (16 = deeper cut)")
parser.add_argument("--batch", type=int, default=64, help="images per batch")
parser.add_argument("--stage1-epochs", type=int, default=5, help="epochs with frozen backbone")
parser.add_argument("--stage2-epochs", type=int, default=40, help="epochs with full model")
parser.add_argument("--person-weight", type=float, default=10.0, help="loss weight of person cells")
parser.add_argument("--workers", type=int, default=6, help="threads loading images")
parser.add_argument("--quick", action="store_true", help="tiny run to check everything works")
args = parser.parse_args()


# --------------------------------------------------
# 2. DATA
# --------------------------------------------------

# Make the random choices repeatable (weights of the new layers, shuffling).
keras.utils.set_random_seed(0)

# Load the training and validation manifests.
train_items = load_manifest("train")
val_items = load_manifest("val")

# For a quick test run, use only a small part of each dataset and one epoch per stage.
if args.quick:
    train_items, val_items = train_items[:640], val_items[:256]
    args.stage1_epochs, args.stage2_epochs = 1, 1

# Create the data loaders: augmented for training, plain for validation.
train_data = FomoData(train_items, size=args.size, batch_size=args.batch, train=True,
                      stride=args.stride, region=args.centre_region)
val_data = FomoData(val_items, size=args.size, batch_size=args.batch, train=False,
                    stride=args.stride, region=args.centre_region)

# Print the dataset sizes.
print(f"Training images: {len(train_items)}, validation images: {len(val_items)}")


# --------------------------------------------------
# 3. MODEL AND OUTPUT FILES
# --------------------------------------------------

# Build the FOMO model and keep a handle on its backbone.
model, trunk = build_fomo(args.size, args.alpha, stride=args.stride)

# Create the folder where models and logs are saved.
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

# Name the output files after the input size and width, e.g. fomo_96_a035.
# The person weight is part of the name so runs with different weights don't overwrite each other.
# Runs with a non-default centre region get it in the name too, e.g. _c100 for the whole box.
region_tag = "" if abs(args.centre_region - 1 / 3) < 1e-6 else f"_c{round(args.centre_region * 100)}"
name = (f"fomo_{args.size}_a{int(args.alpha * 100):03d}_s{args.stride}_w{args.person_weight:g}{region_tag}"
        + ("_quick" if args.quick else ""))
best_model_path = models_dir / f"{name}_best.h5"
log_path = models_dir / f"{name}_log.csv"


# --------------------------------------------------
# 4. VALIDATION CALLBACK
# --------------------------------------------------

class ValidationF1(keras.callbacks.Callback):
    """After each epoch, measure F1 on the validation set and save the best model.

    The best model is chosen by the frame score: the lower of the detection rate
    (images with people where a person was found inside a box) and the rejection
    rate (empty images with no detection). Both must be high for the score to be
    high, so a model that reports a person everywhere scores close to 0.
    The target is 0.9: 90% of people found and at most 10% false alarms.
    """

    def __init__(self):
        super().__init__()

        # Remember the best frame score seen so far, across both stages.
        self.best_score = -1.0

    def on_epoch_end(self, epoch, logs=None):

        # Evaluate the model on the validation set with a threshold of 0.5.
        metrics = evaluate(
            lambda images: self.model(images, training=False).numpy(), val_data, 0.5
        )

        # Add the metrics to the epoch's log so they appear in the CSV and other callbacks.
        for key, value in metrics.items():
            logs[f"val_{key}"] = value

        # Frame score: the worse of detection rate and rejection rate (1 - false alarms).
        score = min(metrics["detection_rate"], 1.0 - metrics["false_alarm_rate"])
        logs["val_frame_score"] = score

        # Print the main metrics.
        print(f" - val box F1 {metrics['box_f1']:.3f} (precision {metrics['box_precision']:.3f}, "
              f"recall {metrics['box_recall']:.3f}, strict F1 {metrics['f1']:.3f}, "
              f"detection {metrics['detection_rate']:.3f}, "
              f"false alarms {metrics['false_alarm_rate']:.3f}, frame score {score:.3f})")

        # Save the model if this is the best frame score so far.
        if score > self.best_score:
            self.best_score = score
            self.model.save(best_model_path, include_optimizer=False)
            print(f" - new best model saved to {best_model_path.name}")


# Create one validation callback, shared by both stages so "best" covers the whole run.
validation = ValidationF1()

# Write each epoch's loss and validation metrics to a CSV file.
csv_logger = keras.callbacks.CSVLogger(str(log_path), append=False)

# Stop training straight away if the loss ever becomes NaN, instead of wasting epochs.
stop_on_nan = keras.callbacks.TerminateOnNaN()


# --------------------------------------------------
# 5. STAGE 1: TRAIN THE HEAD WITH THE BACKBONE FROZEN
# --------------------------------------------------

print("\nSTAGE 1: frozen backbone")

# Freeze the pretrained backbone so only the new head layers learn.
trunk.trainable = False

# Prepare the model for training. The legacy Adam optimizer runs faster on Apple Silicon.
model.compile(
    # clipnorm limits the size of each update, which prevents the loss blowing up to NaN.
    optimizer=keras.optimizers.legacy.Adam(1e-3, clipnorm=1.0),
    loss=weighted_cross_entropy(args.person_weight),
)

# Train the head.
model.fit(
    train_data,
    epochs=args.stage1_epochs,
    callbacks=[validation, csv_logger, stop_on_nan],
    workers=args.workers,
)


# --------------------------------------------------
# 6. STAGE 2: FINE-TUNE THE WHOLE MODEL
# --------------------------------------------------

print("\nSTAGE 2: full model")

# Unfreeze the backbone so all layers learn.
trunk.trainable = True

# Recompile with a smaller learning rate so the pretrained weights change gently.
# Recompiling is required for the change in trainable layers to take effect.
model.compile(
    optimizer=keras.optimizers.legacy.Adam(1e-4, clipnorm=1.0),
    loss=weighted_cross_entropy(args.person_weight),
)

# Write to the same CSV file, adding to the stage 1 rows.
csv_logger = keras.callbacks.CSVLogger(str(log_path), append=True)

# Halve the learning rate if the validation frame score has not improved for 4 epochs.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_frame_score", mode="max", factor=0.5, patience=4, min_lr=1e-6, verbose=1
)

# Stop early if the validation frame score has not improved for 10 epochs.
early_stop = keras.callbacks.EarlyStopping(monitor="val_frame_score", mode="max", patience=10)

# Train the whole model. The validation callback must come first so val_frame_score is in the logs.
model.fit(
    train_data,
    initial_epoch=args.stage1_epochs,
    epochs=args.stage1_epochs + args.stage2_epochs,
    callbacks=[validation, csv_logger, reduce_lr, early_stop, stop_on_nan],
    workers=args.workers,
)

# Report where the best model was saved.
print(f"\nBest validation frame score: {validation.best_score:.3f}")
print(f"Best model: {best_model_path}")
print(f"Training log: {log_path}")
