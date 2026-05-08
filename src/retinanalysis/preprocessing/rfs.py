import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib.patches import Ellipse
from typing import Callable
from tqdm import trange
import matplotlib.pyplot as plt
import os


def matlab_style_gauss2D(sigma_r, sigma_c, c_row, c_col, theta=torch.tensor(0), shape=(5,5), device='cpu')->torch.Tensor:
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    Uses pytorch.
    """
    m, n = [(ss-1.)/2. for ss in shape]
    rows, cols = torch.arange(-m, m+1).to(device), torch.arange(-n, n+1).to(device)
    rows, cols = rows - c_row + m, cols - c_col + n
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
    
    # Rotation parameter
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=device)
    theta_rad = torch.deg2rad(theta)
    cos_theta, sin_theta = torch.cos(theta_rad), torch.sin(theta_rad)
    sin_2theta = torch.sin(2*theta_rad)

    # Compute Gaussian filter
    a = (cos_theta ** 2) / (2 * sigma_r ** 2) + (sin_theta ** 2) / (2 * sigma_c ** 2)
    b = -sin_2theta / (4 * sigma_r ** 2) + sin_2theta / (4 * sigma_c ** 2)
    c = (sin_theta ** 2) / (2 * sigma_r ** 2) + (cos_theta ** 2) / (2 * sigma_c ** 2)
    h = torch.exp(-(a * grid_rows ** 2 + 2 * b * grid_rows * grid_cols + c * grid_cols ** 2))
    
    # Normalize by 1/(2*pi*sigma_r*sigma_c)
    h = h / (2 * torch.pi * sigma_r * sigma_c + 1e-8)

    return h

class ParametrizedSpatialFilters(torch.nn.Module):
    def __init__(self, input_spatial_dims: np.ndarray,
                 b_OFF: bool=True, device: str='cpu',
                 b_1d: bool=False,
                 d_grad_flags: dict={},
                 b_update_filter: bool=True,
                 b_shared_params: bool=False,
                 b_verbose: bool=True,
                 **d_filt_params):
        """
        """
        super().__init__()
        self.device = device
        self.b_1d = b_1d
        self.d_grad_flags = d_grad_flags
        self.b_shared_params = b_shared_params
        if b_verbose:
            print(f'Sharing filter params: {self.b_shared_params}')
        
        self.row_coords = self._to_param(d_filt_params['row_coords'], 'row_coords')
        self.col_coords = self._to_param(d_filt_params['col_coords'], 'col_coords')
        self.c_row_sigmas = self._to_param(d_filt_params['c_row_sigmas'], 'c_row_sigmas')
        self.c_col_sigmas = self._to_param(d_filt_params['c_col_sigmas'], 'c_col_sigmas')
        self.s_row_sigmas = self._to_param(d_filt_params['s_row_sigmas'], 's_row_sigmas')
        self.s_col_sigmas = self._to_param(d_filt_params['s_col_sigmas'], 's_col_sigmas')
        self.s_amps = self._to_param(d_filt_params['s_amps'], 's_amps')
        self.thetas = self._to_param(d_filt_params['thetas'], 'thetas')

        self.b_OFF = b_OFF
        self.polarity = 1.0
        if self.b_OFF:
            self.polarity = -1.0
        
        self.n_filters = len(self.row_coords)
        self.input_spatial_dims = input_spatial_dims
        self.n_ht, self.n_wt = input_spatial_dims
        self.filter = self.compute_filter()
        self.b_update_filter = b_update_filter

    def get_ells(self, sd_mult=1.0, **kwargs):
        ells = []
        for i in range(self.n_filters):
            # Create ellipse for each filter
            ell = Ellipse(xy=(self.col_coords[i].item(), self.row_coords[i].item()),
                          width=2*self.c_col_sigmas[i].item()*sd_mult, height=2*self.c_row_sigmas[i].item()*sd_mult,
                          angle=self.thetas[i].item(), **kwargs)
            ells.append(ell)
        
        return ells

    def _to_param(self, param, str_param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float32, device=self.device)
        param = torch.nn.Parameter(param)
        b_grad = True
        if str_param in self.d_grad_flags.keys():
            b_grad = self.d_grad_flags[str_param]
        param.requires_grad = b_grad
        return param

    def compute_filter(self) -> torch.Tensor:
        filter = torch.zeros((self.n_filters, self.n_ht, self.n_wt), 
                             device=self.device, dtype=torch.float32)
        center = filter.clone()
        surround = filter.clone()
        
        for i in range(self.n_filters):
            if self.b_shared_params:
                center_h = matlab_style_gauss2D(self.c_row_sigmas[0], self.c_col_sigmas[0], 
                                                self.row_coords[i], self.col_coords[i], self.thetas[0],
                                                self.input_spatial_dims, device=self.device)
                surround_h = matlab_style_gauss2D(self.s_row_sigmas[0], self.s_col_sigmas[0],
                                                    self.row_coords[i], self.col_coords[i], self.thetas[0],
                                                    self.input_spatial_dims, device=self.device)
                surround_h =  - self.s_amps[0] * surround_h
                
            else:
                center_h = matlab_style_gauss2D(self.c_row_sigmas[i], self.c_col_sigmas[i], 
                                                self.row_coords[i], self.col_coords[i], self.thetas[i],
                                                self.input_spatial_dims, device=self.device)
                surround_h = matlab_style_gauss2D(self.s_row_sigmas[i], self.s_col_sigmas[i],
                                                    self.row_coords[i], self.col_coords[i], self.thetas[i],
                                                    self.input_spatial_dims, device=self.device)
            
                surround_h = - self.s_amps[i] * surround_h
                
            
            rf_filter = center_h + surround_h

            # Normalize by peak
            rf_filter = rf_filter / (torch.max(torch.abs(rf_filter)) + 1e-8)

            center[i] = center_h
            surround[i] = surround_h

            filter[i] = rf_filter

        filter = filter * self.polarity
        self.center = center * self.polarity
        self.surround = surround * self.polarity

        return filter

    def forward(self, x: torch.Tensor):
        # x is of shape [n_imgs, n_channels, n_ht, n_wt]
        # Output is of shape [n_imgs, n_filters]
        if len(x.shape) != 4:
            raise ValueError(f"Input x must be of shape [n_imgs, n_channels, n_ht, n_wt], but got {x.shape}")
        
        # Calculate filter
        if self.b_update_filter:
            self.filter = self.compute_filter().to(self.device).to(torch.float32)

        n_imgs = x.shape[0]

        # Make x [n_imgs, n_ht*n_wt]
        flat_x = torch.reshape(x, (n_imgs, -1)) 
        # Make filter [n_ht*n_wt, n_filters]
        flat_filter = torch.reshape(self.filter, (self.n_filters, -1)).T

        # Output will be [n_imgs, n_filters]
        self.output = torch.matmul(flat_x, flat_filter)

        return self.output

    def string(self):
        return f"ParametrizedSpatialFilters(filter: {self.filter.shape})"

    def get_params_np(self):
        d_filt_params = {
        'row_coords': self.row_coords.detach().cpu().numpy(),
        'col_coords': self.col_coords.detach().cpu().numpy(),
        'c_row_sigmas': self.c_row_sigmas.detach().cpu().numpy(),
        'c_col_sigmas': self.c_col_sigmas.detach().cpu().numpy(),
        's_row_sigmas': self.s_row_sigmas.detach().cpu().numpy(),
        's_col_sigmas': self.s_col_sigmas.detach().cpu().numpy(),
        's_amps': self.s_amps.detach().cpu().numpy(),
        'thetas': self.thetas.detach().cpu().numpy()
        }
        return d_filt_params

class Spatial_DoG(torch.nn.Module):
    def __init__(self, input_spatial_dims, device, d_filt_params, b_OFF):
        super().__init__()
        self.filt1 = ParametrizedSpatialFilters(input_spatial_dims=input_spatial_dims,
                                          device=device, b_OFF=b_OFF,
                                          **d_filt_params)
        self.n_cells = len(d_filt_params['row_coords'])
        self.gain = torch.nn.Parameter(torch.ones(self.n_cells).to(torch.float32).to(device))
        self.bias = torch.nn.Parameter(torch.zeros(self.n_cells).to(torch.float32).to(device))
        self.device = device
    
    def forward(self, x):
        filter = self.filt1.compute_filter()

        filter = torch.moveaxis(filter, 
                                (0,1,2),
                                (2,0,1))
        
        filter = filter * self.gain + self.bias

        filter = torch.moveaxis(filter, 
                                (2,0,1),
                                (0,1,2))
        
        return filter
    
    def constrain(self):
        with torch.no_grad():
            self.filt1.s_amps.data = torch.clamp(self.filt1.s_amps, min=0.0)


def fit_model_params(model: torch.nn.Module, train_loader: DataLoader,
                    test_loader: DataLoader = None, 
                    str_loss: str='mse', n_total_epochs: int=1000, 
                    n_print_every: int=10, n_lr: float=0.01,
                    n_save_every: int=1,
                    weight_decay: float=0.0, n_patience: int=100) -> dict:
    """
    Fit parameters of a mosaic. 

    Args:
        model
        train_x (torch.Tensor): Training data (n_samples, ... input shape ...)
        train_y (torch.Tensor): Training targets (n_samples, n_cells)
        test_x (torch.Tensor, optional): Test data (n_samples, ... input shape ...)
        test_y (torch.Tensor, optional): Test targets (n_samples, n_cells)
        str_loss (str, optional): Loss function. Defaults to 'mse'.
        n_total_epochs (int, optional): Total epochs. Defaults to 30.
        n_print_every (int, optional): Print loss every. Defaults to 10.
        n_lr (float, optional): _description_. Defaults to 0.1.

    Returns:
        dict: Dictionary tracking optimization:
            train_loss: training loss
            fit_params: fitted parameters
            test_loss: test loss
    """

    b_test = True
    if test_loader is None:
        b_test = False

    # Only update params that require grad.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=n_lr, weight_decay=weight_decay
    )
        
    if str_loss == 'mse':
        f_loss = torch.nn.functional.mse_loss
    elif str_loss == 'l1':
        f_loss = torch.nn.functional.l1_loss
    elif isinstance(str_loss, Callable):
        f_loss = str_loss

    ls_train_loss = []
    ls_test_loss = []
    ls_state_dicts = []
    ls_grad_norms = []
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in trange(n_total_epochs):
        model.train()
        # Forward pass
        e_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = batch[0]
            targets = batch[1]        
            
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            outputs = model(inputs)
            
            if len(batch) == 3:
                weights = batch[2]
                weights = weights.to(model.device)
                loss = f_loss(outputs, targets, weight=weights)
            else:
                loss = f_loss(outputs, targets)
            # If model has regularize(), add to loss
            if hasattr(model, 'regularize'):
                loss += model.regularize(inputs)

            e_loss += loss.item()

            # Backward pass
            loss.backward()
            
            # Calculate gradient norms
            norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    norm += param.grad.data.norm(2).item() ** 2
            norm = np.sqrt(norm)
            ls_grad_norms.append(norm)
            
            optimizer.step()
        
        ls_train_loss.append(e_loss / len(train_loader))
        train_r = np.corrcoef(outputs.detach().cpu().numpy().flatten(), targets.detach().cpu().numpy().flatten())[0,1]

        if b_test:
            with torch.no_grad():
                # Test set loss
                model.eval()
                e_test_loss = 0.0
                for batch in test_loader:
                    inputs = batch[0]
                    targets = batch[1]

                    inputs = inputs.to(model.device)
                    targets = targets.to(model.device)
                    if len(batch) == 3:
                        weights = batch[2]
                        weights = weights.to(model.device)

                    outputs = model(inputs)
                    if len(batch) == 3:
                        test_loss = f_loss(outputs, targets, weight=weights)
                    else:
                        test_loss = f_loss(outputs, targets)
                    e_test_loss += test_loss.item()
                ls_test_loss.append(e_test_loss / len(test_loader))
                test_r = np.corrcoef(outputs.detach().cpu().numpy().flatten(), targets.detach().cpu().numpy().flatten())[0,1]

        # Constrain parameters if model has constrain method
        if hasattr(model, 'constrain'):
            model.constrain()
        
        if (epoch+1) % n_print_every == 0:
            str_print = f"Epoch {epoch+1}/{n_total_epochs}, Train Loss: {ls_train_loss[-1]:.4f}"
            if b_test:
                str_print += f", Test Loss: {ls_test_loss[-1]:.4f}"    
            str_print += f", Grad Norm: {ls_grad_norms[-1]:.4f}"
            # Also print correlation between predictions and targets
            str_print += f"\nTrain R: {train_r:.4f}"
            if b_test:
                str_print += f", Test R: {test_r:.4f}"
            
            print(str_print)
        
        if (epoch+1) % n_save_every == 0:
            # Need to copy the state_dict to CPU before saving
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            state_dict = {
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                # 'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': ls_train_loss[-1]
            }
            if b_test:
                state_dict['test_loss'] = ls_test_loss[-1]
            ls_state_dicts.append(state_dict)

        # Check if loss is NaN
        if np.isnan(ls_train_loss[-1]):
            print('Training Loss is NaN. Stopping training.')
            break
        if b_test:
            if np.isnan(ls_test_loss[-1]):
                print('Test loss is NaN. Stopping training.')
                break

        # Determine current loss to monitor
        current_loss = ls_test_loss[-1] if b_test else ls_train_loss[-1]

        # Early stopping logic
        if current_loss < best_loss - 1e-6:  # small epsilon to avoid float issues
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement for {n_patience} epochs.")
            early_stop = True
            break

    d_track = {'train_loss': np.array(ls_train_loss), 'ls_state_dicts': ls_state_dicts, 'grad_norms': np.array(ls_grad_norms)}
    if b_test:
        d_track['test_loss'] = np.array(ls_test_loss)
        i_best = np.argmin(d_track['test_loss'])
        print(f'Best Test Loss: {d_track["test_loss"][i_best]:.4f} at Epoch {i_best+1}')
    else:
        i_best = np.argmin(d_track['train_loss'])
        print(f'Best Train Loss: {d_track["train_loss"][i_best]:.4f} at Epoch {i_best+1}')

    # Load best state dict
    best_state_dict = ls_state_dicts[i_best]['model_state_dict']
    model.load_state_dict(best_state_dict)
    model.eval()
    print("Loaded best state dict.")

    return d_track


def plot_spatial_dog_performance(model: Spatial_DoG, stas, str_type, cmap='jet', n_pad=15, str_save_dir=None, n_max_rows=50):
    # Plot spatial receptive fields
    n_rows = np.min([model.n_cells, n_max_rows])
    f, axs = plt.subplots(ncols=3, nrows=n_rows, figsize=(9, 3*n_rows))
    if n_rows == 1:
        axs = np.array([axs])
    # If stas is tensor, convert to numpy
    if torch.is_tensor(stas):
        stas = stas.detach().cpu().numpy()
    model_stas = model(None).detach().cpu().numpy()
    print(model_stas.shape)
    for i_cell in range(n_rows):
        # Crop around center pixel
        row = int(model.filt1.row_coords[i_cell].detach().cpu().numpy())
        col = int(model.filt1.col_coords[i_cell].detach().cpu().numpy())
        if row >= stas.shape[1] or col >= stas.shape[2]:
            print(f'Cell {i_cell}: Center ({row}, {col}) out of bounds for STA shape {stas.shape[1:3]}')
            continue

        ax = axs[i_cell, 0]
        vmin, vmax = stas[i_cell].min(), stas[i_cell].max()
        im=ax.imshow(stas[i_cell], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlim(col-n_pad, col+n_pad)
        ax.set_ylim(row-n_pad, row+n_pad)

        ax = axs[i_cell, 1]
        im = ax.imshow(model_stas[i_cell], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlim(col-n_pad, col+n_pad)
        ax.set_ylim(row-n_pad, row+n_pad)

        # Plot slice through center pixel
        ax = axs[i_cell, 2]
        ax.plot(stas[i_cell, row, :], label='STA', alpha=0.7)
        ax.plot(model_stas[i_cell, row, :], label='Model', alpha=0.7)
        ax.set_xlim(col-n_pad, col+n_pad)
        ax.legend(loc='lower left')
    
    axs[0, 0].set_title(f'{str_type} Spatial STA')
    axs[0, 1].set_title('DoG model fit')
    for ax in axs[:,:2].flatten():
        ax.axis('off')
    
    plt.tight_layout()
    if str_save_dir is not None:
        str_save = os.path.join(str_save_dir, f'{str_type}_spatial_dog_fits.png')
        plt.savefig(str_save)
        plt.close()

def plot_model_param_dists(model, str_type, str_save_dir=None):
    ls_params = ['c_row_sigmas', 'c_col_sigmas', 's_row_sigmas', 's_col_sigmas',
                 's_amps', 'thetas']
    ls_labels = ['Center SigmaY', 'Center SigmaX', 'Surround SigmaY', 'Surround SigmaX',
                 'Surround Amplitude', 'Theta']
    ncols, nrows = 3, 2
    f, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 4*nrows))
    for i, ax in enumerate(axs.flatten()):
        if i < len(ls_params):
            ax.grid()
            param = ls_params[i]
            label = ls_labels[i]
            vals = getattr(model.filt1, param).detach().cpu().numpy()
            ax.hist(vals)
            ax.set_title(label)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            
        else:
            ax.axis('off')
    plt.tight_layout()
    if str_save_dir is not None:
        str_save = os.path.join(str_save_dir, f'{str_type}_dog_param_dists.png')
        plt.savefig(str_save)
        plt.close()


def fit_spatial_dog(str_type: str,
                    str_save_dir: str=None, d_filt_params: dict=None,
                    stas: np.ndarray=None, n_total_epochs = 500, n_lr = 0.05):
    # Fit a Difference of Gaussian (DoG) filter to the data
    # data: input data object
    # str_type: type of the filter (e.g., 'DoG')
    
    n_cells, n_height, n_width = stas.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stas = torch.tensor(stas, device=device, dtype=torch.float32)
    print(f'Stas shape: {stas.shape}')
    
    # Initialize filter parameters
    input_spatial_dims = np.array([n_height, n_width])
    
    # Initialize DoG model
    b_OFF = True if 'off' in str_type.lower() else False
    model = Spatial_DoG(input_spatial_dims=input_spatial_dims, device=device, d_filt_params=d_filt_params, b_OFF=b_OFF)
    loss_fn = 'mse'
    
    # n_total_epochs = 20
    print(f'Fitting {str_type} model with {n_cells} cells')

    dataset = TensorDataset(torch.zeros(n_cells), stas)
    loader = DataLoader(dataset, batch_size=n_cells, shuffle=False)
    
    d_track = fit_model_params(
        model=model,
        train_loader=loader,
        str_loss=loss_fn,
        n_total_epochs=n_total_epochs,
        n_print_every=n_total_epochs//5,
        n_lr=n_lr, n_patience=n_total_epochs//10
    )
    i_best = np.argmin(d_track['train_loss'])
    print(f'Best iteration: {i_best}')
    
    # Load the best model
    model.load_state_dict(d_track['ls_state_dicts'][i_best]['model_state_dict'])
    
    # Plot loss
    f, ax = plt.subplots()
    ax.plot(d_track['train_loss'], label='Train Loss')
    ax.axvline(i_best, color='r', linestyle='--', label='Best Iteration')

    plot_spatial_dog_performance(model, stas, str_type, str_save_dir=str_save_dir)
    plot_model_param_dists(model, str_type, str_save_dir=str_save_dir)
    
    if str_save_dir is not None:
        str_save = os.path.join(str_save_dir, f'{str_type}_spatial_dog.pkl')
        save_filt_params(model, str_save)

    return model, d_track, stas.detach().cpu().numpy()