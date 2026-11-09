
# Import os so we can set environment variables before TensorFlow is imported.
import os

# Use legacy Keras (tf_keras) so quantisation-aware training remains possible later.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import certifi, which provides an up-to-date list of trusted certificates.
import certifi

# Point Python at certifi's certificates so the ImageNet weights can be downloaded.
# Without this, this Python installation fails with CERTIFICATE_VERIFY_FAILED.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Import TensorFlow.
import tensorflow as tf

# Use the legacy Keras API provided through TensorFlow.
keras = tf.keras
layers = tf.keras.layers


# --------------------------------------------------
# 1. THE FOMO MODEL
# --------------------------------------------------

# Where to cut MobileNetV2 for each output stride (how many times the image is shrunk).
# Stride 8 keeps the early layers only: each grid cell sees a small patch of the image.
# Stride 16 keeps more layers: each cell sees most of a person, at more RAM and flash.
CUT_LAYERS = {8: "block_6_expand_relu", 16: "block_13_expand_relu"}


def build_fomo(size=96, alpha=0.35, head_channels=32, stride=8):
    """Build the FOMO person detector.

    Returns (model, trunk). The model takes a size x size x 3 image with pixel
    values 0-255 and outputs a (size / 8) x (size / 8) x 2 grid of
    [background, person] probabilities. trunk is the MobileNetV2 backbone,
    returned so training can freeze and unfreeze it.
    """

    # Create the input layer for a size x size RGB image.
    inputs = keras.Input(shape=(size, size, 3))

    # Convert pixels from 0-255 to the -1 to 1 range expected by MobileNetV2.
    # Doing this inside the model means the board can pass raw camera pixels.
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)

    # Create MobileNetV2 with weights pretrained on ImageNet.
    # alpha sets the width of the network: 0.35 uses 35% of the normal channels.
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )

    # Keep only the layers up to the cut point for the chosen stride.
    # Stride 8: 96 -> 12 x 12 grid. Stride 16: 160 -> 10 x 10 grid.
    # The deeper layers are dropped because they cost memory and are not needed.
    trunk = keras.Model(
        base.input, base.get_layer(CUT_LAYERS[stride]).output, name="trunk"
    )

    # Pass the rescaled image through the backbone.
    x = trunk(x)

    # Add a small 1 x 1 convolution that mixes the backbone features in each cell.
    x = layers.Conv2D(head_channels, 1, activation="relu", name="head")(x)

    # Output two probabilities per cell: background and person.
    outputs = layers.Conv2D(2, 1, activation="softmax", name="person_grid")(x)

    # Combine the layers into the full model.
    model = keras.Model(inputs, outputs, name=f"fomo_{size}")

    # Return the model and the backbone.
    return model, trunk


# --------------------------------------------------
# 2. THE LOSS FUNCTION
# --------------------------------------------------

def weighted_cross_entropy(person_weight=100.0):
    """Cross-entropy loss where person cells count person_weight times more.

    About 99% of cells are background, so without this weighting the model
    could score well by predicting background everywhere.
    """

    def loss(y_true, y_pred):

        # Give background cells a weight of 1 and person cells a weight of person_weight.
        weights = y_true[..., 0] + y_true[..., 1] * person_weight

        # Keep the probabilities away from exactly 0 and 1 so log() never gives inf/NaN.
        # The previous run crashed to a NaN loss at epoch 28 without this.
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Calculate the normal cross-entropy for every cell.
        cell_loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        # Return the average of the weighted cell losses.
        return tf.reduce_mean(weights * cell_loss)

    # Return the loss function so Keras can use it.
    return loss


# --------------------------------------------------
# 3. PRINT A SUMMARY WHEN RUN DIRECTLY
# --------------------------------------------------

# Build the model and print its layers when this file is run on its own.
if __name__ == "__main__":
    model, trunk = build_fomo()
    model.summary()
