import numpy as np
from response import ResponseBlock
from stim import StimBlock
from analysis_chunk import AnalysisChunk
import visionloader as vl
import utils.vision_utils as vu

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
        corr_arr: np.ndarray = vu.ei_corr(self.analysis_chunk.vcd, self.response_block.vcd, method = method)
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

    def purpura_spike_distance(self, spike_train_1, spike_train_2):
        pass

