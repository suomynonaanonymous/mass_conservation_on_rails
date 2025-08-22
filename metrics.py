import torch

##############################
### Root Mean Square Error ###
##############################

def compute_RMSE(y_true, y_pred):
    # NOTE: Mean across all points and tasks before we take the square root.
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

############################
### Mean Absolute Error ###
############################

def compute_MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

########################
### Divergence field ###
########################

def compute_divergence_field(mean_pred, x_grad):
    """Generate the divergence field from the mean prediction and the input gradient.
    The output of this function is later used to compute MAD, the mean absolute divergence, which is a measure of how much the flow field deviates from being divergence-free.

    Args:
        mean_pred (torch.Size(N, 2)): 2D vector field predictions, where N is the number of points.
        x_grad (torch.Size(N, 2)): 2D input points, where N is the number of points.

    Returns:
        torch.Size(N, 1): The div field is scalar because we add the two components
    """
    # Because autograd computes gradients of the output w.r.t. the inputs...
    # ... we specify which component of the output you want the gradient of via masking
    # mean_pred is a vector values output
    u_indicator, v_indicator = torch.zeros_like(mean_pred), torch.zeros_like(mean_pred)

    # output mask
    u_indicator[:, 0] = 1.0 # output column u selected
    v_indicator[:, 1] = 1.0 # output column v selected

    # divergence field (positive and negative divergences in case of non-divergence-free models)
    # NOTE: We can imput a whole field because it only depends on the point input
    div_field = (torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = u_indicator,
        create_graph = True
        )[0][:, 0] + torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = v_indicator,
        create_graph = True
        )[0][:, 1])
    
    return div_field