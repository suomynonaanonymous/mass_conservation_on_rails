# HYPERPARAMETERS

# LRS
dfNN_LR = 0.0001 # 1x10^-3 
NN_LR = 0.0001
PINN_LR = 0.0005 # 1x10^-3 was too noisy

# TRAINING
NUM_RUNS = 5
MAX_NUM_EPOCHS = 3000
# Stop after {PATIENCE} epochs with no improvement
PATIENCE = 100
BATCH_SIZE = 1024
WEIGHT_DECAY = 0.001

# DIRECTIONAL GUIDANCE
W_DIRECTIONAL_GUIDANCE = 0.4

# PINN SPECIFIC
# NOTE: PINN-DIR has adjustments due to 2 additional loss terms. See script.
W_PINN_DIV_WEIGHT = 0.2
N_PINN_div_reduction = 1024 * 10 

# EXTRA
# Toggle emission tracking with codecarbon on or off
# TRACK_EMISSIONS_BOOL = False
TRACK_EMISSIONS_BOOL = True

# Define how often to print training progress
PRINT_FREQUENCY = 50