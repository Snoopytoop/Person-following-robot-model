
# Import NumPy to compare the predictions with the targets.
import numpy as np

# Import the data loader and the model.
from dataset import FomoData, load_manifest, keras
from model import build_fomo, weighted_cross_entropy


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Set the input size used by the model.
size = 96

# Set how many images the model should memorise.
number_of_images = 32

# Set how many times the model is trained on those images.
steps = 300


# --------------------------------------------------
# 2. PREPARE A SINGLE FIXED BATCH
# --------------------------------------------------

# Take the first 32 training images that contain at least one person.
items = [item for item in load_manifest("train") if item["boxes"]][:number_of_images]

# Load them as one batch with no augmentation, so the model sees identical images every step.
images, targets = FomoData(items, size=size, batch_size=number_of_images, train=False)[0]


# --------------------------------------------------
# 3. BUILD THE MODEL
# --------------------------------------------------

# Build the FOMO model.
model, trunk = build_fomo(size)

# Freeze only the batch-normalisation layers in the backbone.
# With the same 32 images every step, they would otherwise learn statistics that
# do not match how the model behaves when predicting, hiding whether training worked.
for layer in trunk.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

# Prepare the model for training with the weighted loss.
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=weighted_cross_entropy(100.0))


# --------------------------------------------------
# 4. TRAIN ON THE BATCH
# --------------------------------------------------

# Record the loss before training.
start_loss = model.evaluate(images, targets, verbose=0)

# Train on the same batch over and over.
history = model.fit(images, targets, epochs=steps, batch_size=number_of_images, verbose=0)

# Record the loss after training.
end_loss = history.history["loss"][-1]


# --------------------------------------------------
# 5. CHECK THE PREDICTIONS
# --------------------------------------------------

# Predict the grid for each image and find the class chosen for each cell.
predicted_person = model.predict(images, verbose=0).argmax(-1) == 1

# Find the cells that really contain a person centre.
true_person = targets.argmax(-1) == 1

# Count person cells found, person cells missed and background cells wrongly marked.
found = int((predicted_person & true_person).sum())
missed = int((~predicted_person & true_person).sum())
false_alarms = int((predicted_person & ~true_person).sum())

# Print the results.
print("\nSANITY CHECK")
print("-" * 40)
print(f"Loss before training:      {start_loss:.4f}")
print(f"Loss after training:       {end_loss:.4f}")
print(f"Person cells found:        {found} / {found + missed}")
print(f"Background cells marked:   {false_alarms} / {(~true_person).sum()}")

# Pass if the model found almost every person cell with almost no false alarms.
passed = found >= 0.95 * (found + missed) and false_alarms <= 0.05 * (found + missed)
print("PASSED" if passed else "FAILED - check the labels, grid and loss before training")
