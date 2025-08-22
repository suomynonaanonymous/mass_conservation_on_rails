model_name = "dfNN_dir"
print("Model - ", model_name)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

from configs import *
from metrics import *
from models import dfNN

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

# NOTE: only take (x, y) location inputs
x_train = train[:, [0, 1]].to(device)
print("x_train shape:", x_train.shape)
y_train = train[:, [5, 6]].to(device)
print("y_train shape:", y_train.shape)
print()

### TEST ###
test = torch.load("data/train_test_tensors/test_tensor.pt", weights_only = False)
test = test.T

# NOTE: only take (x, y) location inputs
x_test = test[:, [0, 1]].to(device)
print("x_test shape:", x_test.shape)
y_test = test[:, [5, 6]].to(device)
print("y_test shape:", y_test.shape)
print()

### UNIT VELOCITIES OVER DOMAIN ###
# Actuall learn across full domain since train observations are sparse too
velocity_unit_norm = torch.load("data/directional_guidance/velocity_unit_norm.pt", 
                                     weights_only = False).to(device)

print("velocity_unit_norm shape:", velocity_unit_norm.shape)
# Extract number of velocities as upper bound
N_dg = velocity_unit_norm.shape[0]

########################
### LOOP over EPOCHS ###
########################

for run in range(NUM_RUNS):

    ### TRAINING ###
    # convert to DataLoader for batching
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = dfNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = dfNN_LR, weight_decay = WEIGHT_DECAY)

    train_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
    train_losses_DIR_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
    train_losses_COMBINED_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
    test_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)

    # Early stopping variables
    best_epoch_loss = float('inf')
    # counter starts at 0
    epochs_no_improve = 0

    for epoch in range(MAX_NUM_EPOCHS):

        model.train()

        # accumulate losses over batches for each epoch 
        train_losses_RMSE_over_batches = 0.0
        train_losses_DIR_over_batches = 0.0
        train_losses_COMBINED_over_batches = 0.0

        #########################
        ### LOOP over BATCHES ###
        #########################

        for x_batch, y_batch in dataloader:

            # Move batch to device and enable gradient tracking for x_batch
            x_batch = x_batch.to(device).requires_grad_(True)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(x_batch)
            # NOTE: Train on MSE, report RMSE
            mse_loss = torch.mean((output - y_batch) ** 2)

            ### DIRECTIONAL guidance ###
            # Select another bathc of random indicies
            idx = torch.randint(0, N_dg, (BATCH_SIZE * 5,), device = device)
            batch_dg = velocity_unit_norm[idx]
            # Input locations
            batch_dg_in = batch_dg[:, [0, 1]].requires_grad_().to(device)
            # "directions" (unit vectors) as outputs
            batch_dg_out = batch_dg[:, [2, 3]].to(device)

            dg_output = model(batch_dg_in)
            # cosine similarity per row, then turn into a loss
            cos = F.cosine_similarity(dg_output, batch_dg_out, dim = 1, eps = 1e-8)  # in [-1, 1]
            dg_loss = 1.0 - cos.mean() # 0 = same direction, 2 = opposite

            ### COMBINE LOSSES ###
            # weight and combine
            loss = (1 - W_DIRECTIONAL_GUIDANCE) * mse_loss + W_DIRECTIONAL_GUIDANCE * dg_loss

            loss.backward()
            optimizer.step()

            # Add losses to the epoch loss (over batches)
            # MSE -> RMSE
            train_losses_RMSE_over_batches += torch.sqrt(mse_loss).item()
            train_losses_DIR_over_batches += dg_loss.item()
            train_losses_COMBINED_over_batches += loss.item()


        #############################
        ### END LOOP over BATCHES ###
        #############################
        
        avg_train_loss_RMSE_for_epoch = train_losses_RMSE_over_batches / len(dataloader)
        avg_train_loss_DIR_for_epoch = train_losses_DIR_over_batches / len(dataloader)
        avg_train_loss_COMBINED_for_epoch = train_losses_COMBINED_over_batches / len(dataloader)
        
        train_losses_RMSE_over_epochs[epoch] = avg_train_loss_RMSE_for_epoch
        train_losses_DIR_over_epochs[epoch] = avg_train_loss_DIR_for_epoch
        train_losses_COMBINED_over_epochs[epoch] = avg_train_loss_COMBINED_for_epoch

        # Early stopping check
        # Keep early stopping on RMSE because combined has random component
        if avg_train_loss_RMSE_for_epoch < best_epoch_loss:
            best_epoch_loss = avg_train_loss_RMSE_for_epoch   # <-- update this one
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

        # Avoid leakage
        x_test_eval = x_test.detach().clone().requires_grad_(True).to(device)

        y_pred = model(x_test_eval)

        RMSE = compute_RMSE(y_test, y_pred)
        MAE = compute_MAE(y_test, y_pred)
        # MAD not needed 

        test_losses_RMSE_over_epochs[epoch] = RMSE.item()

        # Print loss every epoch
        if epoch % PRINT_FREQUENCY == 0 or epoch == MAX_NUM_EPOCHS - 1:
            print(f"Run {run} - {model_name} - Epoch {epoch} | Loss (comb.): {avg_train_loss_COMBINED_for_epoch:.3f} - Loss (dir.): {avg_train_loss_DIR_for_epoch:.3f} - Loss (Train RMSE): {avg_train_loss_RMSE_for_epoch:.3f}")
            print(f"Run {run} - {model_name} - Epoch {epoch} | Test RMSE: {RMSE:.3f}, Test MAE: {MAE:.3f}")

    ############################
    ### END LOOP over EPOCHS ###
    ############################

    # Save best model
    torch.save(best_model_state, f"trained_models/{model_name}/{model_name}_epochs_{MAX_NUM_EPOCHS}_run_{run + 1}.pth")
    print("Best model saved.")

    torch.save({
        "train_RMSE": train_losses_RMSE_over_epochs,
        "train_DIR": train_losses_DIR_over_epochs,
        "train_COMBINED": train_losses_COMBINED_over_epochs, 
        "test_RMSE": test_losses_RMSE_over_epochs
    }, f"trained_models/{model_name}/{model_name}_loss_curves_epochs_{MAX_NUM_EPOCHS}_run_{run + 1}.pt")

    # Create a new instance of the model (with same architecture)
    trained_model = dfNN().to(device)

    # Load the saved state dict
    trained_model.load_state_dict(best_model_state)

    # Set to eval mode
    trained_model.eval()

    x_test_final_eval = x_test.detach().clone().requires_grad_(True).to(device)

    y_pred = trained_model(x_test_final_eval)

    RMSE = compute_RMSE(y_test, y_pred)
    MAE = compute_MAE(y_test, y_pred)
    MAD = compute_divergence_field(y_pred, x_test_final_eval).abs().mean()

    print(f"RMSE: {RMSE:.6f}, MAE: {MAE:.6f}, MAD: {MAD:.6f}")

    # Create results dictionary
    results = {
        "model_name": model_name,
        "run": run + 1,
        "epochs": epoch + 1,
        "RMSE": RMSE.item(),
        "MAE": MAE.item(),
        "MAD": MAD.item(),
        "lr": dfNN_LR # track which lr was used
    }

    # Convert to DataFrame (1-row)
    df = pd.DataFrame([results])

    # Ensure directory exists
    os.makedirs(f"results/{model_name}", exist_ok = True)

    # Save to CSV
    df.to_csv(f"results/{model_name}/{model_name}_metrics_run_{run}.csv", index = False, float_format = "%.5f")

##########################
### END LOOP over RUNS ###
#########################

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
