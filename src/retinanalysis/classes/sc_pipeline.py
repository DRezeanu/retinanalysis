import numpy as np
from retinanalysis.classes.response import SCResponseBlock
from retinanalysis.classes.stim import StimBlock
import matplotlib.pyplot as plt
from retinanalysis.utils.regen import get_image_paths_across_epochs, get_df_dict_vals

class PresentImagesSplitter():
    def __init__(self, sb: StimBlock, rb: SCResponseBlock, **regen_kwargs):
        self.sb = sb
        if not hasattr(sb, 'stim_data'):
            self.sb.regenerate_stimulus(**regen_kwargs)
        self.rb = rb
        

        stage_frame_rates = get_df_dict_vals(self.sb.df_epochs, 'frameRate')
        if len(np.unique(stage_frame_rates)) > 1:
            raise ValueError(f'Multiple stage frame rates found: {np.unique(stage_frame_rates)}. Please ensure all epochs have the same stage_frame_rate.')
        self.stage_frame_rate = stage_frame_rates[0]
        self.split_responses_by_u_img()

    def split_responses_by_u_img(self):
        df_epochs = self.sb.df_epochs

        # Get timing info
        pre_time = get_df_dict_vals(df_epochs, 'preTime')[0]
        imgs_per_epoch = get_df_dict_vals(df_epochs, 'imagesPerEpoch')[0]
        imgs_per_epoch = int(imgs_per_epoch)
        
        epoch_flash_frames = get_df_dict_vals(df_epochs, 'flashFrames')
        epoch_gap_frames = get_df_dict_vals(df_epochs, 'gapFrames')
        print(f'Using {imgs_per_epoch} images per epoch, {epoch_flash_frames.mean()} flash frames, {epoch_gap_frames.mean()} gap frames.')

        pre_frames = np.floor(pre_time * 1e-3 * self.stage_frame_rate).astype(int)
        # Correct for LCR Stage error of missing first frame by subtracting 1 frame
        pre_frames -= 1

        n_epochs = len(df_epochs)
        all_image_paths = get_image_paths_across_epochs(df_epochs)
        u_image_paths, u_repeats = np.unique(all_image_paths, return_counts=True)
        repeats = np.unique(u_repeats)
        if len(repeats) > 1:
            raise NotImplementedError(f'Found images with different number of repeats: {repeats}. Please ensure all images have the same number of repeats.')
        repeats = repeats[0]

        # Ensure alignment with loaded u_image_paths
        if np.any(u_image_paths != self.sb.stim_data['u_image_paths']):
            raise ValueError('u_image_paths in StimBlock does not match the u_image_paths generated from the epochs. Please check the StimBlock.')
        
        # Convert frame times ms to samples
        frame_times = []
        avg_frame_rates = []
        for i in range(n_epochs):
            e_fts = self.rb.d_timing['frameTimesMs'][i] 
            e_fts = np.array(e_fts)
            e_fts *= self.rb.amp_sample_rate / 1000
            e_fts = np.round(e_fts).astype(int)
            frame_times.append(e_fts)

            avg_frame_rate = self.rb.frame_sample_rate / np.mean(np.diff(e_fts))
            avg_frame_rates.append(avg_frame_rate)
            
        
        # Compute average frame rate
        avg_frame_rate = np.mean(avg_frame_rates)
        print(f'Average frame rate: {avg_frame_rate} Hz')
        pre_time = pre_frames / avg_frame_rate
        pre_samples = int(np.round(pre_time * self.rb.amp_sample_rate))
        print(f'Pre time: {pre_time:.4f} s, {pre_frames} frames')
        avg_flash_time = epoch_flash_frames.mean() / avg_frame_rate
        avg_flash_samples = int(np.round(avg_flash_time * self.rb.amp_sample_rate))
        avg_gap_time = epoch_gap_frames.mean() / avg_frame_rate
        avg_gap_samples = int(np.round(avg_gap_time * self.rb.amp_sample_rate))
        avg_img_samples = pre_samples + avg_flash_samples + avg_gap_samples
        print(f'Average flash time: {avg_flash_time:.4f} s, gap time: {avg_gap_time:.4f} s, img samples: {avg_img_samples} samples')

        split_raw = []
        split_sts = []
        split_image_paths = []
        split_n_sps = []
        split_pre_n_sps = []
        epoch_pre_n_sps = []
        for i in range(n_epochs):
            sts = self.rb.spike_times[i]
            flash_frames = epoch_flash_frames[i].astype(int)
            gap_frames = epoch_gap_frames[i].astype(int)
            for j in range(imgs_per_epoch):
                # Indexing explanation
                # eg-if 15 pre_frames, index 14 gives onset of last pre frame, 
                # index 15 gives onset of first flash frame
                t_onset = frame_times[i][pre_frames + j * (flash_frames + gap_frames)]
                t_offset = frame_times[i][pre_frames + j * (flash_frames + gap_frames) + flash_frames]
                t_delta = t_offset - t_onset
                t_p_end = frame_times[i][pre_frames + (j + 1) * (flash_frames + gap_frames)]
                
                t_p_start = frame_times[i][j * (flash_frames + gap_frames)]
                img_sts = sts[(sts >= t_p_start) & (sts < t_p_end)]
                n_sps = len(img_sts[(img_sts > t_onset) & (img_sts <= t_offset)]) 
                n_pre_sps = len(img_sts[(img_sts <= t_onset) & (img_sts >= t_onset - t_delta)])
                img_sts = img_sts - t_p_start
                
                img_raw = self.rb.amp_data[i][t_p_start:t_p_start + avg_img_samples]
                
                split_raw.append(img_raw)
                split_sts.append(img_sts)
                split_n_sps.append(n_sps)
                split_pre_n_sps.append(n_pre_sps)
                split_image_paths.append(all_image_paths[i * imgs_per_epoch + j])
            epoch_pre_sps = len(sts[sts < frame_times[i][pre_frames - 1]])
            epoch_pre_n_sps.append(epoch_pre_sps)
        split_raw = np.array(split_raw)
        split_sts = np.array(split_sts, dtype=object)
        split_n_sps = np.array(split_n_sps).astype(int)
        split_pre_n_sps = np.array(split_pre_n_sps).astype(int)
        epoch_pre_n_sps = np.array(epoch_pre_n_sps).astype(int)
        split_image_paths = np.array(split_image_paths)
        

        # Make (u_img_idx, repeat, data) array
        u_image_paths = self.sb.stim_data['u_image_paths']
        reshaped_raw = np.zeros((len(u_image_paths), repeats, avg_img_samples))
        reshaped_sts = np.zeros((len(u_image_paths), repeats), dtype=object)
        reshaped_n_sps = np.zeros((len(u_image_paths), repeats), dtype=int)
        reshaped_pre_n_sps = np.zeros((len(u_image_paths), repeats), dtype=int)

        for i, u_img_path in enumerate(u_image_paths):
            img_indices = np.where(split_image_paths == u_img_path)[0]
            reshaped_raw[i] = split_raw[img_indices].reshape((repeats, avg_img_samples))
            reshaped_sts[i] = split_sts[img_indices]
            reshaped_n_sps[i] = split_n_sps[img_indices]
            reshaped_pre_n_sps[i] = split_pre_n_sps[img_indices]

        self.epoch_pre_n_sps = epoch_pre_n_sps
        self.split_sts = reshaped_sts
        self.split_n_sps = reshaped_n_sps
        self.split_pre_n_sps = reshaped_pre_n_sps
        self.split_raw = reshaped_raw
        self.u_image_paths = u_image_paths
        self.d_avg_timing = {
            'pre_time': pre_time,
            'pre_samples': pre_samples,
            'pre_frames': pre_frames,
            'avg_flash_samples': avg_flash_samples,
            'avg_gap_samples': avg_gap_samples,
            'avg_img_samples': avg_img_samples,
            'avg_frame_rate': avg_frame_rate,
            'avg_flash_time': avg_flash_time,
            'avg_gap_time': avg_gap_time,
        }

    def plot_eg_trace(self, i_u_img, i_repeat=0):
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        raw = self.split_raw[i_u_img, i_repeat]
        sts = self.split_sts[i_u_img, i_repeat]
        n_sps = self.split_n_sps[i_u_img, i_repeat]
        time = np.arange(len(raw)) / self.rb.amp_sample_rate
        ax.plot(time, raw, label='Raw trace')
        ax.scatter(sts/self.rb.amp_sample_rate, raw[sts], color='r')
        ax.axvline(self.d_avg_timing['pre_time'], c='grey' )
        ax.axvline(self.d_avg_timing['pre_time'] + self.d_avg_timing['avg_flash_time'], c='grey' )
        ax.set_title(f'{self.u_image_paths[i_u_img]} - {n_sps} spikes')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return ax

class ExpandingSpotsPipeline():
    def __init__(self, sb: StimBlock, rb: SCResponseBlock, b_sub_baseline: bool=False):
        self.sb = sb
        self.rb = rb
        self.spot_sizes = self.sb.df_epochs['currentSpotSize'].values
        if self.rb.b_spiking:
            self.get_stim_nspikes(b_sub_baseline=b_sub_baseline)

    def get_stim_nspikes(self, b_sub_baseline: bool=False):
        ls_nsps = []
        print('Getting stim spikes based on time, precise frametime not yet implement...')
        for e_idx in self.sb.df_epochs.index:
            pre_time = self.sb.df_epochs.at[e_idx, 'preTime']
            stim_time = self.sb.df_epochs.at[e_idx, 'stimTime']
            offset_time = pre_time + stim_time
            # Convert from ms to samples
            onset_samples = int(np.round(pre_time * self.rb.amp_sample_rate / 1000))
            offset_samples = int(np.round(offset_time * self.rb.amp_sample_rate / 1000))
            sts = self.rb.spike_times[e_idx]
            n_sps = len(sts[(sts >= onset_samples) & (sts <= offset_samples)])
            if b_sub_baseline:
                n_baseline = len(sts[(sts < onset_samples)])
                n_sps -= n_baseline
            ls_nsps.append(n_sps)
        self.stim_spikes = np.array(ls_nsps)
    
    def plot_rf(self):
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.grid()
        ax.scatter(self.spot_sizes, self.stim_spikes, c='k', s=30)
        ax.set_xlabel('Spot Diameter (um)')
        ax.set_ylabel('Number of Spikes')
        
    
    def plot_eg_trace(self, i_epoch):
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        raw = self.rb.amp_data[i_epoch]
        sts = self.rb.spike_times[i_epoch]
        n_sps = self.stim_spikes[i_epoch]
        time = np.arange(len(raw)) / self.rb.amp_sample_rate
        ax.plot(time, raw, label='Raw trace')
        ax.scatter(sts/self.rb.amp_sample_rate, raw[sts], color='r')
        pre_time = self.sb.df_epochs.at[i_epoch, 'preTime']
        stim_time = self.sb.df_epochs.at[i_epoch, 'stimTime']
        offset_time = pre_time + stim_time
        ax.axvline(pre_time / 1000, c='grey', label='Stim onset')
        ax.axvline(offset_time / 1000, c='grey', label='Stim offset')
        spot_size = self.sb.df_epochs.at[i_epoch, 'currentSpotSize']
        ax.set_title(f'Epoch {i_epoch} {spot_size}um spot: {n_sps} spikes')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
    

