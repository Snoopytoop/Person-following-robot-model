
# Import re to find the progress numbers in the training output.
import re

# Import sys to read the optional output file path from the command line.
import sys

# Import time to wait between refreshes.
import time

# Import Path to create and manage file and folder paths.
from pathlib import Path


# Set the training output file to watch. A different file can be passed on the command line.
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "models" / "train_output.txt"

# Set the width of the progress bar in characters.
bar_width = 40

# Record when watching started, to estimate the time remaining.
started = time.time()
start_progress = None

# Refresh the display every two seconds until stopped with Ctrl+C.
while True:

    # Read the output. Keras rewrites its progress line with carriage returns.
    text = output_path.read_text(errors="ignore").replace("\r", "\n")

    # Find the current epoch and total epochs, e.g. "Epoch 7/45".
    epochs = re.findall(r"^Epoch (\d+)/(\d+)", text, re.M)

    # Find the current batch and total batches, e.g. "348/666 [====".
    batches = re.findall(r"^\s*(\d+)/(\d+) \[", text, re.M)

    # Find every validation F1 printed so far.
    f1_scores = [float(f1) for f1 in re.findall(r"val box F1 ([\d.]+)", text)]

    # Read the frame score and false-alarm rate printed after each epoch (new training script).
    frame_scores = [float(v) for v in re.findall(r"frame score ([\d.]+)", text)]
    false_alarms = [float(v) for v in re.findall(r"false alarms ([\d.]+)", text)]
    detections = [float(v) for v in re.findall(r"detection ([\d.]+)", text)]

    # Wait if training has not printed its first epoch yet.
    if not epochs or not batches:
        print("\rWaiting for training to start...", end="", flush=True)
        time.sleep(2)
        continue

    # Convert the latest numbers to integers.
    epoch, total_epochs = map(int, epochs[-1])
    batch, total_batches = map(int, batches[-1])

    # During stage 1 Keras only counts stage 1's epochs, so add stage 2's default 40.
    if "STAGE 2" not in text:
        total_epochs += 40

    # Calculate the overall progress across all epochs, from 0 to 1.
    progress = ((epoch - 1) + batch / total_batches) / total_epochs

    # Estimate the time remaining from the progress made since watching started.
    if start_progress is None:
        start_progress = progress
    rate = (progress - start_progress) / max(time.time() - started, 1)
    minutes_left = (1 - progress) / rate / 60 if rate > 0 else float("nan")

    # Build the progress bar.
    filled = int(progress * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)

    # Describe the latest and best validation F1.
    scores = f"box F1 last {f1_scores[-1]:.3f} best {max(f1_scores):.3f}" if f1_scores else "box F1 -"

    # Prefer the new metrics when the run prints them.
    if frame_scores:
        scores = (f"frame score {frame_scores[-1]:.3f} (best {max(frame_scores):.3f})  "
                  f"found {detections[-1]:.0%}  false alarms {false_alarms[-1]:.0%}")

    # Show whether training has finished.
    finished = "Best model:" in text
    status = "DONE" if finished else f"~{minutes_left:.0f} min left"

    # Print the line, overwriting the previous one.
    print(
        f"\r[{bar}] {progress:6.1%}  epoch {epoch}/{total_epochs}  "
        f"batch {batch}/{total_batches}  {scores}  {status}     ",
        end="", flush=True,
    )

    # Stop once training has finished.
    if finished:
        print()
        break

    # Wait before refreshing.
    time.sleep(2)
