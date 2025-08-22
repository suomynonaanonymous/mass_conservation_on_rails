model_name = "NN_aux"
print("Model - ", model_name)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

from configs import *
from metrics import *
from models import NN_aux

from utils import set_seed
set_seed(42)

### START TIMING ###
import time
start_time = time.time()  # Start timing after imports

### START TRACKING EXPERIMENT EMISSIONS ###
if TRACK_EMISSIONS_BOOL:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(project_name = model_name, output_dir = "results/_emissions/", output_file = f"{model_name}.csv")
    tracker.start()

# setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# overwrite if needed: # device = 'cpu'
print('Using device:', device)
print()

### TRAIN ###
train = torch.load("data/train_test_tensors/train_tensor.pt", weights_only = False)
train = train.T

# Columns of tensor t (train / test) (N × 9)
# t[:, 0] - x coordinate mapped to [0,1] using domain bounds (dimensionless)
# t[:, 1] - y coordinate mapped to [0,1] using domain bounds (dimensionless)
# t[:, 2] - surface elevation / surface_scale (e.g., 1000 m)
# t[:, 3] - (∂s/∂x) / sgrad_scale (e.g., m/km) → surface x-gradient
# t[:, 4] - (∂s/∂y) / sgrad_scale (e.g., m/km) → surface y-gradient
# t[:, 5] - x-component ice flux / flux_scale (sign preserved) - TARGET
# t[:, 6] - y-component ice flux / flux_scale (sign preserved) - TARGET
# t[:, 7] - x-component velocity (vx) / vel_scale
# t[:, 8] - y-component velocity (vy) / vel_scale

# NOTE: take (x, y) location inputs and (s)
x_train = train[:, [0, 1, 2]].to(device)
print("x_train shape:", x_train.shape)
y_train = train[:, [5, 6]].to(device)
print("y_train shape:", y_train.shape)
print()

### TEST ###
test = torch.load("data/train_test_tensors/test_tensor.pt", weights_only = False)
test = test.T

# NOTE: take (x, y) location inputs and (s)
x_test = test[:, [0, 1, 2]].to(device)
print("x_test shape:", x_test.shape)
y_test = test[:, [5, 6]].to(device)
print("y_test shape:", y_test.shape)
print()

########################
### LOOP over EPOCHS ###
########################

for run in range(NUM_RUNS):

    ### TRAINING ###
    # convert to DataLoader for batching
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = NN_aux().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = NN_LR, weight_decay = WEIGHT_DECAY)

    train_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
    test_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)

    # Early stopping variables
    best_epoch_loss = float('inf')
    # counter starts at 0
    epochs_no_improve = 0

    for epoch in range(MAX_NUM_EPOCHS):

        model.train()
        optimizer.zero_grad()

        # accumulate losses over batches for each epoch 
        train_losses_RMSE_over_batches = 0.0

        #########################
        ### LOOP over BATCHES ###
        #########################

        for x_batch, y_batch in dataloader:

            # Move batch to device and enable gradient tracking for x_batch
            x_batch = x_batch.to(device).requires_grad_(True)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            # Train on MSE
            loss = torch.mean((output - y_batch) ** 2)
            loss.backward()
            optimizer.step()

            # Add losses to the epoch loss (over batches)
            # MSE -> RMSE
            train_losses_RMSE_over_batches += torch.sqrt(loss).item()

        #############################
        ### END LOOP over BATCHES ###
        #############################
        
        avg_train_loss_RMSE_for_epoch = train_losses_RMSE_over_batches / len(dataloader)
        train_losses_RMSE_over_epochs[epoch] = avg_train_loss_RMSE_for_epoch

        # reset to max
        if epoch == 50:
            best_epoch_loss = float('inf')

        # Early stopping check
        if avg_train_loss_RMSE_for_epoch < best_epoch_loss:
            best_epoch_loss = avg_train_loss_RMSE_for_epoch
            epochs_no_improve = 0  # reset counter
            best_model_state = model.state_dict()  # save best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # EVALUATE CURRENT MODEL ON TEST SET
        # Do this every epoch so we can track performance
        model.eval()

        y_pred = model(x_test.requires_grad_(True).to(device))

        RMSE = compute_RMSE(y_test, y_pred)
        MAE = compute_MAE(y_test, y_pred)
        # MAD not needed 

        test_losses_RMSE_over_epochs[epoch] = RMSE.item()

        # Print loss every epoch
        if epoch % PRINT_FREQUENCY == 0 or epoch == MAX_NUM_EPOCHS - 1:
            print(f"Run {run} - {model_name} - Epoch {epoch} | Loss (Train RMSE): {avg_train_loss_RMSE_for_epoch:.3f}")
            print(f"Run {run} - {model_name} - Epoch {epoch} | Test RMSE: {RMSE:.3f}, Test MAE: {MAE:.3f}")

    ############################
    ### END LOOP over EPOCHS ###
    ############################

    # Save best model
    torch.save(best_model_state, f"trained_models/{model_name}/{model_name}_epochs_{MAX_NUM_EPOCHS}_run_{run + 1}.pth")
    print("Best model saved.")

    torch.save({
        "train_RMSE": train_losses_RMSE_over_epochs,
        "test_RMSE": test_losses_RMSE_over_epochs
    }, f"trained_models/{model_name}/{model_name}_loss_curves_epochs_{MAX_NUM_EPOCHS}_run_{run + 1}.pt")

    # Create a new instance of the model (with same architecture)
    trained_model = NN_aux().to(device)

    # Load the saved state dict
    trained_model.load_state_dict(best_model_state)

    # Set to eval mode
    trained_model.eval()

    y_pred = trained_model(x_test.requires_grad_(True).to(device))

    RMSE = compute_RMSE(y_test, y_pred)
    MAE = compute_MAE(y_test, y_pred)
    MAD = compute_divergence_field(y_pred, x_test).abs().mean()

    print(f"RMSE: {RMSE:.6f}, MAE: {MAE:.6f}, MAD: {MAD:.6f}")

    # Create results dictionary
    results = {
        "model_name": model_name,
        "run": run + 1,
        "epochs": epoch + 1,
        "RMSE": RMSE.item(),
        "MAE": MAE.item(),
        "MAD": MAD.item(),
        "lr": NN_LR # track which lr was used
    }

    # Convert to DataFrame (1-row)
    df = pd.DataFrame([results])

    # Ensure directory exists
    os.makedirs(f"results/{model_name}", exist_ok = True)

    # Save to CSV
    df.to_csv(f"results/{model_name}/{model_name}_metrics_run_{run}.csv", index = False, float_format = "%.5f")

##########################
### END LOOP over RUNS ###
##########################

#############################
### WALL time & GPU model ###
#############################

end_time = time.time()
# compute elapsed time
elapsed_time = end_time - start_time 
# convert elapsed time to minutes
elapsed_time_minutes = elapsed_time / 60

# also end emission tracking. Will be saved as emissions.csv
if TRACK_EMISSIONS_BOOL:
    tracker.stop()

if device == "cuda":
    # get name of GPU model
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "N/A"

print(f"Elapsed wall time: {elapsed_time:.4f} seconds")

# Define full path for the file
wall_time_and_gpu_path = os.path.join("results/_walltime/", model_name + "_run_" "wall_time.txt")

# Save to the correct folder with both seconds and minutes
with open(wall_time_and_gpu_path, "w") as f:
    f.write(f"Elapsed wall time: {elapsed_time:.4f} seconds\n")
    f.write(f"Elapsed wall time: {elapsed_time_minutes:.2f} minutes\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU model: {gpu_name}\n")

print(f"Wall time saved to {wall_time_and_gpu_path}.")