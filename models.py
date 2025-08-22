import torch
import torch.nn as nn

################
### dfNN aux ###
################

class dfNN_aux(nn.Module):
    def __init__(self, coordinate_dims = 2, input_dim = 3, hidden_dim = 64, n_hidden_layers = 4):
        super().__init__()
        # Number of input coordinates (e.g., x, y)
        self.coordinate_dims = coordinate_dims

        # Number of input features (e.g., x, y, s) INCLUDING auxiliary inputs
        self.input_dim = input_dim
        self.output_dim = 1  # Scalar potential
        self.hidden_dim = hidden_dim

        # HACK: SiLu() worked much better than ReLU() for this model

        layers = [nn.Linear(self.input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, return_H = False):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, input_dim]  # Symplectic gradient
        """
        # Retrieve scalar potential
        H = self.net(x)

        # Compute full partials with respect to x coordinates, but only use coordinate dims
        partials = torch.autograd.grad(
                outputs = H.sum(), # we can sum here because every H row only depend on every x row
                inputs = x,
                create_graph = True
            )[0]
        
        # Only keep coordinate partials
        coord_partials = partials[:, :self.coordinate_dims]

        # Symplectic gradient
        # flip columns (last dim) for x2, x1 order. Multiply x2 by -1
        symp = coord_partials.flip(-1) * torch.tensor([1, -1], dtype = torch.float32, device = x.device)

        # return symp, H # NOTE: return H as well if we want to see what is going on
        if return_H:
            return symp, H
        else:
            return symp

############
### dfNN ###
############
  
class dfNN(nn.Module):
    def __init__(self, coordinate_dims = 2, hidden_dim = 64, n_hidden_layers = 4):
        super().__init__()
        # Number of input coordinates (e.g., x, y)
        self.coordinate_dims = coordinate_dims

        self.output_dim = 1  # Scalar potential
        self.hidden_dim = hidden_dim

        # HACK: SiLu() worked much better than ReLU() for this model

        layers = [nn.Linear(self.coordinate_dims, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, return_H = False):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, input_dim]  # Symplectic gradient
        """
        # Retrieve scalar potential
        H = self.net(x)

        # Compute full partials with respect to x coordinates, but only use coordinate dims
        partials = torch.autograd.grad(
                outputs = H.sum(), # we can sum here because every H row only depend on every x row
                inputs = x,
                create_graph = True
            )[0]

        # Symplectic gradient
        # flip columns (last dim) for x2, x1 order. Multiply x2 by -1
        symp = partials.flip(-1) * torch.tensor([1, -1], dtype = torch.float32, device = x.device)

        # return symp, H # NOTE: return H as well if we want to see what is going on
        if return_H:
            return symp, H
        else:
            return symp

########## 
### NN ###
##########  
 
class NN(nn.Module):
    def __init__(self, input_dims = 2, hidden_dim = 64, n_hidden_layers = 4):
        super().__init__()
        # Number of input dims
        self.input_dims = input_dims

        # NOTE: Here we directly output the vector field, so output_dim = 2
        self.output_dim = 2 
        self.hidden_dim = hidden_dim

        # TODO: Test various activation functions
        layers = [nn.Linear(self.input_dims, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, output_dim]
        """
        v = self.net(x)

        return v
    
##############
### NN aux ###
##############

class NN_aux(nn.Module):
    def __init__(self, input_dims = 3, hidden_dim = 64, n_hidden_layers = 4):
        super().__init__()
        # Number of input dims
        self.input_dims = input_dims

        # NOTE: Here we directly output the vector field, so output_dim = 2
        self.output_dim = 2 
        self.hidden_dim = hidden_dim

        # TODO: Test various activation functions
        layers = [nn.Linear(self.input_dims, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, output_dim]
        """
        v = self.net(x)

        return v

#####################
### PINN backbone ###
#####################
  
class PINN_backbone(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, hidden_dim = 64, n_hidden_layers = 4):
        super().__init__()
        # Number of input coordinates (e.g., x, y)
        self.input_dim = input_dim
        
        # Output dim follows input dims
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim

        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, output_dim]
        """
        # Return 2D vector field
        v = self.net(x)

        return v