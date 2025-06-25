from stim import Stim, NoiseStim
import utils.vision_utils as vu
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

class Response:
    def __init__(self, stim: Stim, ss_version: str = 'kilosort2.5'):
        self.stim = stim
        self.ss_version = ss_version

class NoiseResponse(Response):
    def __init__(self, stim: NoiseStim, typing_file_name: str):
        super().__init__(stim, stim.ss_version)
        self.stim = stim
        vcd = vu.get_vcd(self.stim.exp_name, self.stim.chunk_name, self.ss_version,
                         ei = False, params = True, neurons = False)
        self.vcd = vcd
        self.typing_file_name = typing_file_name

    def get_rf_params(self, ls_cells: list=None):
        if ls_cells is None:
            ls_cells = self.vcd.get_cell_ids()
        all_rf_params = []
        for cell_id in ls_cells:
            rf_params = self.vcd.get_stafit_for_cell(cell_id)
            center_x = rf_params.center_x + self.stim.deltaXChecks
            center_y = rf_params.center_y + self.stim.deltaYChecks
            d_params = {
                'xy': (center_x, center_y),
                'width': rf_params.std_x,
                'height': rf_params.std_y,
                'angle': rf_params.rot,
            }
            all_rf_params.append(d_params)
        return all_rf_params
    
    def plot_rf_ells(self, ls_cells: list=None):
        all_rf_params = self.get_rf_params(ls_cells)
        all_rf_ells = [Ellipse(**params) for params in all_rf_params]
        f, ax = plt.subplots(figsize=(5, 5))
        alpha = 0.5
        ell_color = 'black'
        lw = 1.5
        for idx_ell, ell in enumerate(all_rf_ells):
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell.set_edgecolor(ell_color)
            ell.set_facecolor('none')
            ell.set_linewidth(lw)
        ax.set_xlim(right=self.stim.staXChecks)
        ax.set_ylim(top=self.stim.staYChecks)
        