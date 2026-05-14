import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib.patches import Ellipse
from typing import Callable
from tqdm import trange
import matplotlib.pyplot as plt
import os
from skimage.segmentation import flood

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
                 device: str='cpu',
                 b_1d: bool=False,
                 d_grad_flags: dict={},
                 b_update_filter: bool=True,
                 **d_filt_params):
        """
        """
        super().__init__()
        self.device = device
        self.b_1d = b_1d
        self.d_grad_flags = d_grad_flags
        
        self.row_coords = self._to_param(d_filt_params['row_coords'], 'row_coords')
        self.col_coords = self._to_param(d_filt_params['col_coords'], 'col_coords')
        self.c_row_sigmas = self._to_param(d_filt_params['c_row_sigmas'], 'c_row_sigmas')
        self.c_col_sigmas = self._to_param(d_filt_params['c_col_sigmas'], 'c_col_sigmas')
        self.s_row_sigmas = self._to_param(d_filt_params['s_row_sigmas'], 's_row_sigmas')
        self.s_col_sigmas = self._to_param(d_filt_params['s_col_sigmas'], 's_col_sigmas')
        self.s_amps = self._to_param(d_filt_params['s_amps'], 's_amps')
        self.thetas = self._to_param(d_filt_params['thetas'], 'thetas')
        
        self.n_cells = len(self.row_coords)
        self.input_spatial_dims = input_spatial_dims
        self.n_ht, self.n_wt = input_spatial_dims
        self.filter = self.compute_filter()
        self.b_update_filter = b_update_filter

    def get_ells(self, sd_mult=1.0, **kwargs):
        ells = []
        for i in range(self.n_cells):
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
        filter = torch.zeros((self.n_cells, self.n_ht, self.n_wt), 
                             device=self.device, dtype=torch.float32)
        
        vec_gauss_fn = torch.vmap(matlab_style_gauss2D, in_dims=(0, 0, 0, 0, 0, None, None))
        centers = vec_gauss_fn(
            self.c_row_sigmas, self.c_col_sigmas, self.row_coords, self.col_coords, self.thetas,
            self.input_spatial_dims, self.device
        )
        surrounds = vec_gauss_fn(
            self.s_row_sigmas, self.s_col_sigmas, self.row_coords, self.col_coords, self.thetas,
            self.input_spatial_dims, self.device
        )
        surround = - self.s_amps.view(-1, 1, 1) * surrounds
        filter = centers + surround
        
        # Normalize by peak
        peaks = torch.max(torch.abs(filter).reshape(self.n_cells, -1), dim=1)[0].view(-1, 1, 1)
        peaks[peaks == 0] = 1
        filter = filter / peaks

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
        # Make filter [n_ht*n_wt, n_cells]
        flat_filter = torch.reshape(self.filter, (self.n_cells, -1)).T

        # Output will be [n_imgs, n_cells]
        self.output = torch.matmul(flat_x, flat_filter)

        return self.output

    def string(self):
        return f"ParametrizedSpatialFilters(filter: {self.filter.shape})"

    def get_params_np(self)-> dict:
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
    def __init__(self, input_spatial_dims, device, d_init_params):
        super().__init__()

        self.parametrized_filter = ParametrizedSpatialFilters(
            input_spatial_dims=input_spatial_dims,
            device=device,
            **d_init_params
        )
        self.n_cells = len(d_init_params['row_coords'])
        
        # Make [K, 1, 1] gain and bias for per cell scaling and shifting.
        gain = torch.ones(self.n_cells, device=device, dtype=torch.float32)
        bias = torch.zeros(self.n_cells, device=device, dtype=torch.float32)
        gain = gain.view(-1, 1, 1)
        bias = bias.view(-1, 1, 1)

        self.gain = torch.nn.Parameter(gain)
        self.bias = torch.nn.Parameter(bias)

        self.device = device
    
    def forward(self, x):
        """
        Input is unused cell index.
        Output is of shape [K cells]
        """
        filter = self.parametrized_filter.compute_filter()
        
        filter = filter * self.gain + self.bias
        
        return filter
    
    def constrain(self):
        with torch.no_grad():
            self.parametrized_filter.s_amps.data = torch.clamp(self.parametrized_filter.s_amps, min=0.0)


def fit_model_params(model: torch.nn.Module, train_loader: DataLoader,
                    n_total_epochs: int=1000, 
                    n_print_every: int=10, n_lr: float=0.01,
                    n_save_every: int=1,
                    weight_decay: float=0.0, n_patience: int=100) -> dict:
    """
    Fit parameters of a mosaic. 

    Args:
        model
        train_loader (DataLoader): DataLoader for training data. Should yield batches of (inputs, targets)
        n_total_epochs (int, optional): Total epochs. Defaults to 1000.
        n_print_every (int, optional): Print loss every. Defaults to 10.
        n_lr (float, optional): _description_. Defaults to 0.01.

    Returns:
        dict: Dictionary tracking optimization:
            train_loss: training loss
            fit_params: fitted parameters
    """

    # Only update params that require grad.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=n_lr, weight_decay=weight_decay
    )
        
    f_loss = torch.nn.functional.mse_loss

    ls_train_loss = []
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
            
            loss = f_loss(outputs, targets)

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

        # Constrain parameters if model has constrain method
        if hasattr(model, 'constrain'):
            model.constrain()
        
        if (epoch+1) % n_print_every == 0:
            str_print = f"Epoch {epoch+1}/{n_total_epochs}, Train Loss: {ls_train_loss[-1]:.4f}"
            str_print += f", Grad Norm: {ls_grad_norms[-1]:.4f}"
            # Also print correlation between predictions and targets
            str_print += f"\nTrain R: {train_r:.4f}"
            
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
            ls_state_dicts.append(state_dict)

        # Check if loss is NaN
        if np.isnan(ls_train_loss[-1]):
            print('Training Loss is NaN. Stopping training.')
            break

        # Determine current loss to monitor
        current_loss = ls_train_loss[-1]

        # Early stopping logic: if loss doesn't improve for n_patience epochs, stop training
        if current_loss < best_loss:  
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement for {n_patience} epochs.")
            break

    d_track = {'train_loss': np.array(ls_train_loss), 'ls_state_dicts': ls_state_dicts, 'grad_norms': np.array(ls_grad_norms)}

    i_best = np.argmin(d_track['train_loss'])
    d_track['i_best'] = i_best
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
        row = int(model.parametrized_filter.row_coords[i_cell].detach().cpu().numpy())
        col = int(model.parametrized_filter.col_coords[i_cell].detach().cpu().numpy())
        if row >= stas.shape[1] or col >= stas.shape[2]:
            print(f'Cell {i_cell}: Center ({row}, {col}) out of bounds for STA shape {stas.shape[1:3]}')
            continue

        ax = axs[i_cell, 0]
        vmin, vmax = stas[i_cell].min(), stas[i_cell].max()
        im=ax.imshow(stas[i_cell], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlim(col-n_pad, col+n_pad)
        ax.set_ylim(row-n_pad, row+n_pad)
        ax.set_ylabel(f'Cell idx {i_cell}')

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
        r = np.corrcoef(stas[i_cell].flatten(), model_stas[i_cell].flatten())[0,1]
        ax.set_title(f'r = {r:.2f}')
    
    axs[0, 0].set_title(f'{str_type} Spatial STA')
    axs[0, 1].set_title('DoG model fit')
    for ax in axs[:,:2].flatten():
        ax.axis('off')
    
    plt.tight_layout()
    if str_save_dir is not None:
        str_save = os.path.join(str_save_dir, f'{str_type}_spatial_dog_fits.png')
        plt.savefig(str_save)
        plt.close()

def plot_model_param_dists(model: Spatial_DoG, str_type, str_save_dir=None):
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
            vals = getattr(model.parametrized_filter, param).detach().cpu().numpy()
            ax.hist(vals, bins=100)
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


def fit_spatial_dog(
        spatial_stas: np.ndarray, 
        d_init_params: dict,
        str_save_dir: str=None,
        n_total_epochs = 500, n_lr = 0.05, n_patience=500
    )-> tuple[Spatial_DoG, dict]:
    # Fit a Difference of Gaussian (DoG) filter to the data
    # data: input data object
    # str_type: type of the filter (e.g., 'DoG')
    
    n_cells, n_height, n_width = spatial_stas.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spatial_stas = torch.tensor(spatial_stas, device=device, dtype=torch.float32)
    print(f'Stas shape: {spatial_stas.shape}')
    
    # Initialize filter parameters
    input_spatial_dims = np.array([n_height, n_width])
    
    # Initialize DoG model
    model = Spatial_DoG(
        input_spatial_dims=input_spatial_dims, device=device, 
        d_init_params=d_init_params
    )
    
    print(f'Fitting DoG model with {n_cells} cells')

    dataset = TensorDataset(torch.zeros(n_cells), spatial_stas)
    loader = DataLoader(dataset, batch_size=n_cells, shuffle=False)
    
    d_track = fit_model_params(
        model=model,
        train_loader=loader,
        n_total_epochs=n_total_epochs,
        n_print_every=n_total_epochs//5,
        n_lr=n_lr, n_patience=n_patience
    )

    # Plot loss
    f, axs = plt.subplots(ncols=2, figsize=(12, 5))
    ax = axs[0]
    ax.plot(d_track['train_loss'], label='Train Loss')
    ax.axvline(d_track['i_best'], color='r', linestyle='--', label='Best Iteration')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    # Histogram of correlations
    ax = axs[1]
    ls_r = []
    model_stas = model(None).detach().cpu().numpy()
    for i_cell in range(n_cells):
        r = np.corrcoef(spatial_stas[i_cell].cpu().numpy().flatten(), model_stas[i_cell].flatten())[0,1]
        ls_r.append(r)
    ax.hist(ls_r, bins=50)
    ax.set_title('Performance distribution')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    if str_save_dir is not None:
        str_save = os.path.join(str_save_dir, f'dog_training_loss.png')
        plt.savefig(str_save)
        plt.close()


    plot_spatial_dog_performance(model, spatial_stas, 'All', str_save_dir=str_save_dir)
    plot_model_param_dists(model, 'All', str_save_dir=str_save_dir)

    return model, d_track

def rf_fitting_pipeline(
        stas: np.ndarray, str_output_dir: str,
        n_baseline_frames: int=0
    )->dict:
    """
    Main function for estimating RF params from STAs.
    Step 1: estimates initial params by peak pixel and contiguous region.
    Step 2: fits DoG model to peak spatial frame.

    Args:
        stas (np.ndarray): [K cells, D depth, H height, W width, C channels]
        str_output_dir (str, optional): Directory to save outputs and plots.
        n_baseline_frames (int, optional): Number of baseline frames to subtract mean from.
    """

    if n_baseline_frames > 0:
        # Subtract baseline mean from STA.
        stas = stas - np.mean(stas[:, :n_baseline_frames], axis=1, keepdims=True)

    # Get peak index of each cell's STA
    n_cells, n_depth, n_height, n_width, n_channels = stas.shape
    peak_idxs = np.argmax(np.abs(stas).reshape(n_cells, -1), axis=1)
    peak_ts, peak_hs, peak_ws, peak_cs = np.unravel_index(peak_idxs, (n_depth, n_height, n_width, n_channels))

    # Collect center pixel timecourse for every channel for final output.
    timecourses = stas[np.arange(n_cells), :, peak_hs, peak_ws, :]
    
    # Get peak spatial frame [K, H, W].
    spatial_stas = stas[np.arange(n_cells), peak_ts, :, :, peak_cs]
    # Make them all "ON" i.e. positive peaks
    signs = np.sign(stas[np.arange(n_cells), peak_ts, peak_hs, peak_ws, peak_cs])
    spatial_stas = spatial_stas * signs[:, np.newaxis, np.newaxis]

    ## Estimate initial sigmas from mask.
    # Make high SNR mask, with threshold of top 10% of abs values in the peak frame
    thresholds = np.percentile(spatial_stas.reshape(n_cells, -1), 90, axis=1)
    thresholds = thresholds.reshape(-1, 1, 1)
    masks = np.where(spatial_stas > thresholds, 1, 0)
    
    # Compute contiguous region around peak, and estimate row and col sigmas.
    ls_row_sigmas = []
    ls_col_sigmas = []
    for i in range(n_cells):
        mask = masks[i]
        peak_h, peak_w = peak_hs[i], peak_ws[i]
        if mask[peak_h, peak_w] == 0:
            print(f'Warning: Peak pixel for cell {i} is not in high SNR mask. Consider adjusting threshold or checking data quality.')
            cnt_mask = mask
            continue
        
        # Use flood fill to find contiguous region around the peak
        cnt_mask = flood(mask, (peak_h, peak_w), connectivity=1)
        
        # Get bounding box of contiguous region
        h_inds, w_inds = np.where(cnt_mask)
        n_cnt_rows = w_inds.max() - w_inds.min()
        n_cnt_cols = h_inds.max() - h_inds.min()
        ls_row_sigmas.append(n_cnt_cols / 2.0)
        ls_col_sigmas.append(n_cnt_rows / 2.0)
    
    # Construct d_init_params for DoG model
    ls_row_sigmas = np.array(ls_row_sigmas)
    ls_col_sigmas = np.array(ls_col_sigmas)
    d_init_params = {
        'row_coords': peak_hs,
        'col_coords': peak_ws,
        'c_row_sigmas': ls_row_sigmas,
        'c_col_sigmas': ls_col_sigmas,
        's_row_sigmas': ls_row_sigmas * 1.5,
        's_col_sigmas': ls_col_sigmas * 1.5,
        's_amps': np.ones(n_cells) * 0.5,
        'thetas': np.zeros(n_cells)
    }
    
    str_plot_dir = None
    if os.path.isdir(str_output_dir):
        str_plot_dir = os.path.join(str_output_dir, 'plots')
        os.makedirs(str_plot_dir, exist_ok=True)
    else:
        raise ValueError(f'str_output_dir {str_output_dir} is not a valid directory.')
    
    model, d_track = fit_spatial_dog(
        spatial_stas=spatial_stas,
        str_save_dir=str_plot_dir,
        d_init_params=d_init_params,
        n_total_epochs=1000, n_lr=0.05,
        n_patience=500
    )

    # Get model params
    d_fit_params = model.parametrized_filter.get_params_np()
    
    # Add timecourses
    d_fit_params['timecourses'] = timecourses

    return d_fit_params