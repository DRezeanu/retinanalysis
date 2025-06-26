import numpy as np
from response import ResponseBlock
from stim import StimBlock
from analysis_chunk import AnalysisChunk
import visionloader as vl

# Do we want a separate plotting class? Would it take a pipeline object as well?

# class MEAPlotter:

#     def __init__(self, stim_block: StimBlock, response_block: ResponseBlock, analysis_chunk: AnalysisChunk):
#         self.stim_block = stim_block
#         self.response_block = response_block
#         self.analysis_chunk = analysis_chunk

#     def get_ells(self):
#         pass

#     def plot_ells(self):
#         pass

#     def plot_psth(self):
#         pass

#     def plot_spike_count(self):
#         pass



class MEAPipeline:

    def __init__(self, stim_block: StimBlock, response_block: ResponseBlock, analysis_chunk: AnalysisChunk):
        self.stim_block = stim_block
        self.response_block = response_block
        self.analysis_chunk = analysis_chunk

        self.match_dict = self.cluster_match()

    def cluster_match(self, corr_cutoff: float = 0.8, method: str = 'full'):
        corr_arr: np.ndarray = self.ei_corr(self.analysis_chunk.vcd, self.response_block.vcd, method = method)
        # Methods: full, space, power. Space uses no time dimension, power squares the response

        match_dict = dict()
        match_count = 0
        bad_match_count = 0

        analysis_ids = self.analysis_chunk.cell_ids
        response_ids = self.response_block.cell_ids

        for idx, ref_cell in enumerate(analysis_ids):
            sorted_corr = np.sort(corr_arr[idx,:])
            sorted_corr = np.flip(sorted_corr)

            max_corr = sorted_corr[0]
            next_max_corr = sorted_corr[1]
            max_ind = np.argmax(corr_arr[idx,:])
            max_rev_ind = np.argmax(corr_arr[:,max_ind])

            if max_corr > corr_cutoff:
                if next_max_corr > (max_corr*0.90):
                    bad_match_count += 1
                elif analysis_ids[max_rev_ind] != ref_cell:
                    bad_match_count += 1
                else:
                    match_dict[ref_cell] = response_ids[max_ind]
                    match_count += 1    

            else:
                bad_match_count += 1
        
        percent_good = match_count/len(analysis_ids)
        percent_bad = bad_match_count/len(analysis_ids)

        print(f"\nRef clusters matched: {match_count}")
        print(f"Ref clusters unmatched: {bad_match_count}")
        print(f"{np.round(percent_good*100, 2)}% matched, {np.round(percent_bad*100, 2)}% unmatched.")
        
        self.match_dict = dict(sorted(match_dict.items()))

        return self.match_dict


    def get_type_dict(self):
        pass

    def ei_corr(self, analysis_vcd: vl.VisionCellDataTable, response_vcd: vl.VisionCellDataTable, method: str = 'full') -> np.ndarray:

        # Pull reference eis
        ref_ids = analysis_vcd.get_cell_ids()
        ref_eis = [analysis_vcd.get_ei_for_cell(cell).ei for cell in ref_ids]

        # Set any EI value where the ei is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(ref_eis):
            ref_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # Flatten 512 x 201 array into a vector
        ref_eis_flat = [ei.flatten() for ei in ref_eis]
        ref_eis = np.array(ref_eis_flat)

        # Pull test eis
        test_ids = response_vcd.get_cell_ids()
        test_eis = [response_vcd.get_ei_for_cell(cell).ei for cell in test_ids]

        # Set the EI value where the EI is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(test_eis):
            test_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # Flatten all the eis and turn them into numpy array
        test_eis_flat = [ei.flatten() for ei in test_eis]
        test_eis = np.array(test_eis_flat)

        num_pts = ref_eis.shape[1]

        # Calculate covariance and correlation
        c = test_eis @ ref_eis.T / num_pts
        d = np.mean(test_eis, axis = 1)[:,None] * np.mean(ref_eis, axis = 1)[:,None].T
        covs = c - d

        std_calc = np.std(test_eis, axis = 1)[:,None] * np.std(ref_eis, axis = 1)[:, None].T
        corr = covs / std_calc

        # Set nan values and infinite values to 0
        np.nan_to_num(corr, copy=False, nan = 0, posinf = 0, neginf = 0)

        return corr.T


    def purpura_spike_distance(self, spike_train_1, spike_train_2):
        pass

