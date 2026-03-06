import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import tqdm
import pandas as pd
import os
import cv2
from typing import Optional
from retinanalysis.classes.response import MEAResponseBlock, SCResponseBlock

def get_df_dict_vals(df, key, col_name='epoch_parameters'):
    vals = np.array([d[key] for d in df[col_name].values])
    return vals

def load_mp4(str_mp4, b_crop_screencap=True, display_dims=(600,800)):
    cap = cv2.VideoCapture(str_mp4)
    rec_frames = []
    while True:
        ret, rec_frame = cap.read()
        if not ret:
            break
        # Convert BGR frame to RGB
        rec_frame = cv2.cvtColor(rec_frame, cv2.COLOR_BGR2RGB)
        rec_frames.append(rec_frame)
    cap.release()

    rec_frames = np.array(rec_frames)
    print(f'Loaded {len(rec_frames)} frames from {str_mp4}')
    print(f'Frame shape: {rec_frames.shape}')
    
    if b_crop_screencap:
        n_rows, n_cols = display_dims
        rec_frames = np.array([r_frame[:n_rows, :n_cols, :] for r_frame in rec_frames])
        print(f'Cropped frame shape: {rec_frames.shape}')
    return rec_frames

def make_spatial_noise(df_epochs: pd.DataFrame, center_row: Optional[int]=None,
                       center_col: Optional[int]=None, n_pad: Optional[int]=None,
                       canvas_size: tuple=None):
    # Create noise movies by epochs
    ls_frames = []
    ls_steps = []
    for e_idx in tqdm.tqdm(df_epochs.index):
        fts = df_epochs.at[e_idx, 'frame_times_ms']
        pre_time = df_epochs.at[e_idx, 'preTime']
        unique_time = df_epochs.at[e_idx, 'epoch_parameters']['uniqueTime']
        repeat_time = df_epochs.at[e_idx, 'epoch_parameters']['repeatTime']
        
        # TODO get saved out unique_frames, and compute total frames with state.time logic, which should be equivalent to below method.
        unique_frames = len(np.where(np.logical_and((fts > pre_time),(fts <= pre_time+unique_time)))[0])
        repeat_frames = len(np.where(np.logical_and((fts > pre_time+unique_time), (fts <= pre_time+unique_time+repeat_time)))[0])
        
        d_e_params = df_epochs.at[e_idx, 'epoch_parameters']
        d_meta = {
            'numXStixels': d_e_params['numXStixels'],
            'numYStixels': d_e_params['numYStixels'],
            'numXChecks': d_e_params['numXChecks'],
            'numYChecks': d_e_params['numYChecks'],
            'gridSizeUm': d_e_params['gridSize'],
            'chromaticClass': d_e_params['chromaticClass'],
            'unique_frames': unique_frames,
            'repeat_frames': repeat_frames,
            'stepsPerStixel': d_e_params['stepsPerStixel'],
            'seed': d_e_params['seed'],
            'frameDwell': d_e_params['frameDwell']
        }
        if 'canvasSize' in d_e_params:
            d_meta['canvasSize'] = d_e_params['canvasSize']
        else:
            if canvas_size is None:
                # Default canvas size
                canvas_size = (1140, 1824)
                print(f'canvasSize not in epoch params and not provided, defaulting to {canvas_size}')
            d_meta['canvasSize'] = canvas_size
        if 'gaussianFilter' in d_e_params:
            d_meta['gaussianFilter'] = d_e_params['gaussianFilter']
        if 'filterSdStixels' in d_e_params:
            d_meta['filterSdStixels'] = d_e_params['filterSdStixels']
        if 'canvasSize' in d_e_params:
            # (x, y) to (rows, cols)
            d_meta['canvasSize'] = tuple(d_e_params['canvasSize'][::-1])  
        if 'micronsPerPixel' in d_e_params:
            d_meta['micronsPerPixel'] = d_e_params['micronsPerPixel']
        if 'repeating_seed' in d_e_params:
            d_meta['repeating_seed'] = d_e_params['repeating_seed']
        d_wn = get_spatial_noise_frames(**d_meta)
        ls_steps.append(d_wn['steps'])
        ls_frames.append(d_wn['stimulus'])
    frames = np.array(ls_frames)
    steps = np.array(ls_steps)

    # Checking all optional variables for more accurate static type checking
    if center_row is not None and center_col is not None and n_pad is not None:
        # Crop frames around the cell center
        frames = frames[:, :, center_row-n_pad:center_row+n_pad+1, center_col-n_pad:center_col+n_pad+1, :]
    d_out = {
        'frames': frames,
        'steps': steps,
        'canvas_size': canvas_size,
    }

    return d_out


def lcr_video_device_um_to_pix(um_value: float, micronsPerPixel: float) -> int:
    # Rounding like in LCRVideoDevice.m
    pix_value = np.round(um_value / micronsPerPixel).astype(int)
    return pix_value


def get_spatial_noise_frames(numXStixels: int, 
                        numYStixels: int, 
                        numXChecks: int, 
                        numYChecks: int, 
                        gridSizeUm: float,
                        chromaticClass: str, 
                        unique_frames: int, 
                        repeat_frames: int,
                        stepsPerStixel: int, 
                        seed: int, 
                        frameDwell: int,
                        gaussianFilter: bool=False,
                        filterSdStixels: float=1.0,
                        canvasSize: tuple=None,
                        micronsPerPixel: float=None,
                        repeating_seed: int=None):
    """
    Get the frame sequence for the FastNoiseStimulus.
    From symphony_data.py
    Parameters:
        numXStixels: number of stixels in the x direction.
        numYStixels: number of stixels in the y direction.
        numXChecks: number of checks in the x direction.
        numYChecks: number of checks in the y direction.
        chromaticClass: chromatic class of the stimulus.
        numFrames: number of frames in the stimulus.
        stepsPerStixel: number of steps per stixel.
        seed: seed for the random number generator.
        frameDwell: number of frames to dwell on each frame.

    Returns:
    frames: 4D array of frames (n_frames, x, y, n_colors).
    """
    # Print input params
    if canvasSize is None:
        canvasSize = (1140, 1824)
        print(f'canvasSize not provided, defaulting to {canvasSize}')
    if micronsPerPixel is None:
        micronsPerPixel = 3.37
        print(f'micronsPerPixel not provided, defaulting to {micronsPerPixel}')
    if repeating_seed is None:
        repeating_seed = 1
        print(f'repeating_seed not provided, defaulting to {repeating_seed}')
    
    # Cast to ints
    numXStixels = int(numXStixels)
    numYStixels = int(numYStixels)
    numXChecks = int(numXChecks)
    numYChecks = int(numYChecks)
    unique_frames = int(unique_frames)
    repeat_frames = int(repeat_frames)
    seed = int(seed)
    stepsPerStixel = int(stepsPerStixel)
    frameDwell = int(frameDwell)
    
    print(f'numXStixels: {numXStixels}, numYStixels: {numYStixels}, numXChecks: {numXChecks}, numYChecks: {numYChecks}')
    print(f'chromaticClass: {chromaticClass}, unique_frames: {unique_frames}, repeat_frames: {repeat_frames}')
    print(f'stepsPerStixel: {stepsPerStixel}, seed: {seed}, frameDwell: {frameDwell}')
    print(f'gaussianFilter: {gaussianFilter}, filterSdStixels: {filterSdStixels}')
    
    # Compute gridSizePix, stixelSizePix. 
    gridSizePix = lcr_video_device_um_to_pix(gridSizeUm, micronsPerPixel)
    stixelSizePix = gridSizePix * stepsPerStixel
    print(f'Grid size: {gridSizePix} pix, Stixel size: {stixelSizePix} pix')
    
    # Seed the random number generator.
    np.random.seed( int(seed) )

    # Chromatic class determines time factor
    # Eg-2 for BY first generates 2*num_frames, then splits into B and Y frames.
    if (chromaticClass == 'BY'):
        tfactor = 2
    elif (chromaticClass == 'RGB'):
        tfactor = 3
    # Black/white checks
    else: 
        tfactor = 1

    # Get the size of the time dimension; expands for chromatic stimuli.
    numFrames = unique_frames + repeat_frames
    tsize = np.ceil(numFrames*tfactor/frameDwell).astype(int)
    usize = np.ceil(unique_frames*tfactor/frameDwell).astype(int)
    rsize = np.ceil(repeat_frames*tfactor/frameDwell).astype(int)

    # Ensure even tsize for tfactor=2 (BY) stimuli.
    if (tfactor == 2 and (tsize % 2) != 0):
        tsize += 1
        rsize += 1

    # Generate the random grid of stixels.
    gridValues = np.zeros((tsize, int(numXStixels*numYStixels)), dtype=np.float32)
    # print(f'gridValues: {gridValues.shape}')
    
    # Set unique sequence.
    gridValues[:usize,:] = np.random.rand(usize, int(numXStixels*numYStixels))
    
    # Set repeating sequence.
    if repeat_frames > 0:
        # Reseed the generator.
        np.random.seed(repeating_seed)
        gridValues[usize:,:] = np.random.rand(rsize, int(numXStixels*numYStixels))
    
    # Reshape to (t, y, x)
    gridValues = np.reshape(gridValues, (tsize, int(numXStixels), int(numYStixels)))
    gridValues = np.transpose(gridValues, (0, 2, 1))

    # Binarize to 0 and 1, then convert to contrast (-1 to 1)
    gridValues = np.round(gridValues)
    gridValues = (2*gridValues-1).astype(np.float32) 
    print(f'gridValues: {gridValues.shape}')

    # Filter the stixels if indicated.
    if gaussianFilter:
        for i in range(tsize):
            frame_tmp = gaussian_filter(gridValues[i,:,:], sigma=filterSdStixels, order=0, mode='wrap')
            gridValues[i,:,:] = 0.5*frame_tmp/np.std(frame_tmp)
        gridValues[gridValues > 1.0] = 1.0
        gridValues[gridValues < -1.0] = -1.0

    # Upscale to the fullGrid of stixels. 
    # Each element represents stixelSizePix pixels on the display.
    # This is typically larger than the canvas so it can be jittered without clipping.
    full_shape = np.array(gridValues.shape)
    full_shape[1:] *= stepsPerStixel
    fullGrid = np.zeros(full_shape, dtype=np.float32)

    # Set fullGrid values by repeating stixel values. TODO: Optimize
    for k in range(fullGrid.shape[1]):
        yindex = np.floor(k/stepsPerStixel).astype(int)
        for m in range(fullGrid.shape[2]):
            xindex = np.floor(m/stepsPerStixel).astype(int)
            fullGrid[:, k, m] = gridValues[:, yindex, xindex]
    
    # For debugging: full pixel space
    # fullGrid = np.zeros((tsize,int(numYStixels*stixelSizePix),int(numXStixels*stixelSizePix)), dtype=np.float32)
    # rs_shape = (fullGrid.shape[1], fullGrid.shape[2])
    # for t in range(tsize):
    #     fullGrid[t,:,:] = cv2.resize(
    #         gridValues[t,:,:], 
    #         rs_shape[::-1], 
    #         interpolation=cv2.INTER_NEAREST_EXACT)

    print(f'fullGrid: {fullGrid.shape}')

    ## Generate the motion trajectory of the larger stixels.
    # Re-seed the number generator.
    np.random.seed( int(seed) )

    # Random steps range from 0-(stepsPerStixel-1)
    steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )

    # frameValues is downscaled version of full canvas, containing the cropped fullGrid.
    
    frameValues = np.zeros((tsize,numYChecks, numXChecks),dtype=np.float32)
    
    # For debugging: full pixel space
    # frameValues = np.zeros((tsize,canvasSize[0], canvasSize[1]),dtype=np.float32)
    print(f'frameValues: {frameValues.shape}')

    # Compute crop amounts so fullGrid is centered on frameValues canvas.
    crop_Y = (fullGrid.shape[1] - frameValues.shape[1]) / 2
    crop_X = (fullGrid.shape[2] - frameValues.shape[2]) / 2
    print(f'Precise Crop X: {crop_X:.2f}, Crop Y: {crop_Y:.2f}')
    crop_Y = np.round(crop_Y).astype(int)
    crop_X = np.round(crop_X).astype(int)
    print(f'Rounded Crop X: {crop_X}, Crop Y: {crop_Y}')

    # Apply the jittered cropping to get frameValues from fullGrid.
    for k in range(tsize):
        x_offset = steps[np.floor(k/tfactor).astype(int), 0].astype(int)
        y_offset = steps[np.floor(k/tfactor).astype(int), 1].astype(int)
        
        # For debugging: full pixel space
        # print(f'Frame {k}: x_offset: {x_offset}, y_offset: {y_offset}')
        # x_offset *= gridSizePix
        # y_offset *= gridSizePix

        # Y jitter moves up and adds to crop from the top.
        y_offset += crop_Y
    
        # X jitter moves right, so subtracts from crop from the left.
        x_offset = crop_X - x_offset
        
        # X start and ends
        grid_x_start = x_offset
        grid_x_end = x_offset + numXChecks
        frame_x_start = 0
        frame_x_end = numXChecks

        # Y start and ends
        grid_y_start = y_offset
        grid_y_end = y_offset + numYChecks
        frame_y_start = 0
        frame_y_end = numYChecks
        
        # For X, need to check if grid_x_start< 0
        # If so, then get frameValues index for leaving left edge gray.
        if grid_x_start < 0:
            frame_x_start = -grid_x_start
            grid_x_start = 0
            grid_x_end = numXChecks - frame_x_start
            
        # For both X and Y, need to check if grid_end > fullGrid
        # If so, then get frameValues index for leaving right/bottom edge gray.
        if grid_y_end > fullGrid.shape[1]:
            n_bottom_gray = grid_y_end - fullGrid.shape[1]
            frame_y_end = numYChecks - n_bottom_gray
            grid_y_end = fullGrid.shape[1]

        if grid_x_end > fullGrid.shape[2]:
            n_right_gray = grid_x_end - fullGrid.shape[2]
            frame_x_end = numXChecks - n_right_gray
            grid_x_end = fullGrid.shape[2]

        # Assign the larger grid values to the final display frame values.
        frameValues[k, frame_y_start:frame_y_end, frame_x_start:frame_x_end] = fullGrid[
            k, 
            grid_y_start:grid_y_end, 
            grid_x_start:grid_x_end]

        # For debugging: full pixel space
        # frameValues[k,:,:] = fullGrid[k, y_offset : canvasSize[0]+y_offset, x_offset : canvasSize[1]+x_offset]

    # Create your output stimulus. (t, y, x, 3 color channels)
    t_downsampled = np.ceil(tsize/tfactor).astype(int)
    stimulus = np.zeros((t_downsampled, numYChecks, numXChecks, 3), dtype=np.float32)
    
    # For debugging: full pixel space
    # stimulus = np.zeros((np.ceil(tsize/tfactor).astype(int), canvasSize[0], canvasSize[1], 3), dtype=np.float32)
    print(f'stimulus: {stimulus.shape}')
    # Get the pixel values into the proper color channels
    if (chromaticClass == 'BY'):
        stimulus[:,:,:,0] = frameValues[0::2,:,:]
        stimulus[:,:,:,1] = frameValues[0::2,:,:]
        stimulus[:,:,:,2] = frameValues[1::2,:,:]
    elif (chromaticClass == 'RGB'):
        stimulus[:,:,:,0] = frameValues[0::3,:,:]
        stimulus[:,:,:,1] = frameValues[1::3,:,:]
        stimulus[:,:,:,2] = frameValues[2::3,:,:]
    else: # Black/white checks
        stimulus[:,:,:,0] = frameValues
        stimulus[:,:,:,1] = frameValues
        stimulus[:,:,:,2] = frameValues
    
    # Deal with the frame dwell.
    if frameDwell > 1:
        stim = np.zeros((numFrames, numYChecks, numXChecks, 3), dtype=np.float32)
        for k in range(numFrames):
            idx = np.floor(k / frameDwell).astype(int)
            stim[k,:,:,:] = stimulus[idx,:,:,:]
        stimulus = stim

    d_out = {
        'stimulus': stimulus,
        'steps': steps,
    }
    
    return d_out


# Functions for PresentImages protocol
def load_and_process_img(str_img,screen_size = np.array([1140, 1824]), # rows, cols
                        magnification_factor = 8,
                        ds_factor: int=3, rescale: bool=True,
                        verbose: bool=False, bg: float=0.0):
    img = cv2.imread(str_img, cv2.IMREAD_GRAYSCALE)
    screen_size = screen_size.astype(int)

    img_size = np.array(img.shape)

    scene_size = img_size * magnification_factor
    scene_size = scene_size.astype(int)
    if verbose:
        print(f'Loaded image size: {img_size}')
        print(f'Scene size: {scene_size}')
    # Scale image to scene size with linear interpolation
    img_resized = cv2.resize(img, tuple(scene_size[::-1]), interpolation=cv2.INTER_LINEAR)

    # scene_position = screen_size / 2
    scene_position = scene_size / 2
    if verbose:
        print(f'Image resized size: {img_resized.shape}')
        print(f'Scene position: {scene_position}')
        print(f'Img dtype: {img_resized.dtype}')

    frame = np.zeros(screen_size, dtype=img_resized.dtype)
    frame += np.uint8(bg)

    x_vals = np.arange(-screen_size[1] / 2, screen_size[1] / 2)
    y_vals = np.arange(-screen_size[0] / 2, screen_size[0] / 2)
    x_idx = np.round(scene_position[1] + x_vals).astype(int)
    y_idx = np.round(scene_position[0] + y_vals).astype(int)

    x_good = (x_idx >= 0) & (x_idx < scene_size[1])  #& (x_idx < screen_size[1])
    y_good = (y_idx >= 0) & (y_idx < scene_size[0])  #& (y_idx < screen_size[0])

    assign_idx = np.ix_(np.where(y_good)[0], np.where(x_good)[0])
    frame[assign_idx] = img_resized[y_idx[y_good], :][:, x_idx[x_good]]

    # Downsample by ds factor
    if ds_factor > 1:
        ds_shape = np.array(frame.shape) // ds_factor
        if verbose:
            print(f'Downsampling by {ds_factor} to shape {ds_shape}')
        ds_shape = ds_shape.astype(int)
        frame = cv2.resize(frame, tuple(ds_shape[::-1]), interpolation=cv2.INTER_LINEAR)

    if rescale:
        # Convert from (0,255) to (-1, 1)
        frame = (frame.astype(np.float32) / 255.0) * 2 - 1

    return frame


def get_image_paths_across_epochs(df_epochs: pd.DataFrame, n_actual_imgs_per_epoch: Optional[int]=None):
    # Each epoch has multiple images with associated name and folder
    # Return a 2D (epoch, images) array of paths to images.
    all_image_names = get_df_dict_vals(df_epochs, 'imageName')
    all_image_names = np.stack([x.split(',') for x in all_image_names])

    all_image_folders = get_df_dict_vals(df_epochs, 'folder')
    all_image_folders = np.stack([x.split(',') for x in all_image_folders])
    n_epochs  = len(all_image_names)
    print(f'Found {n_epochs} epochs metadata, each with: ')
    num_imgs_across_epochs = [len(x) for x in all_image_names]
    num_folders_across_epochs = [len(x) for x in all_image_folders]
    
    # Assert all are identical
    assert np.all(np.array(num_imgs_across_epochs) == num_imgs_across_epochs[0]), "Not all epochs have same number of images!"
    assert np.all(np.array(num_folders_across_epochs) == num_folders_across_epochs[0]), "Not all epochs have same number of folders!"
    assert num_imgs_across_epochs[0] == num_folders_across_epochs[0], "Number of image names and folders do not match!"
    print(f'{num_imgs_across_epochs[0]} images per epoch')

    if n_actual_imgs_per_epoch is not None:
        print(f'Using {n_actual_imgs_per_epoch} images per epoch as specified.')
        all_image_names = all_image_names[:, :n_actual_imgs_per_epoch]
        all_image_folders = all_image_folders[:, :n_actual_imgs_per_epoch]

    # Populate full paths = 'folder/name'
    all_image_paths = np.zeros(all_image_names.shape, dtype=object)
    for i in range(n_epochs):
        for j in range(all_image_names.shape[1]):
            str_path = os.path.join(all_image_folders[i, j], all_image_names[i, j])
            all_image_paths[i, j] = str_path
    
    
    return all_image_paths

def get_old_presentimages_transitions(df_epochs: pd.DataFrame,
                                      frame_rate=59,
                                      stride=1,
                                      pre_flash_time=0.25,
                                      verbose=True):
    # Method for old versions of PresentImages protocol that used state.time for transitions.
    n_trials = len(df_epochs)
    pre_time = df_epochs.at[0, 'epoch_parameters']['preTime'] * 1e-3
    flash_time = df_epochs.at[0, 'epoch_parameters']['flashTime'] * 1e-3
    gap_time = df_epochs.at[0, 'epoch_parameters']['gapTime'] * 1e-3
    tail_time = df_epochs.at[0, 'epoch_parameters']['tailTime'] * 1e-3
    stim_time = df_epochs.at[0, 'epoch_parameters']['stimTime'] * 1e-3

    total_time = pre_time + stim_time + tail_time

    total_frames = np.round(frame_rate * total_time).astype(int)
    frame_time = 1 / frame_rate

    # Generate state.time for each frame
    state_time = np.arange(0, total_frames * frame_time, frame_time)

    n_flashes = int(np.floor(stim_time / (flash_time + gap_time)))

    pre_bins = (state_time < pre_time).sum() * stride
    stim_bins = ((state_time >= pre_time) &
                 (state_time < pre_time + stim_time)).sum() * stride
    tail_bins = (state_time >= pre_time + stim_time).sum() * stride

    total_bins = pre_bins + stim_bins + tail_bins

    state_time -= pre_time
    ls_flash_bins = np.zeros((n_trials, n_flashes), dtype=int)
    ls_gap_bins = np.zeros((n_trials, n_flashes), dtype=int)
    for i in range(n_trials):
        for j in range(n_flashes):
            flash_bins = ((state_time >= j * (flash_time + gap_time))
                          & (state_time <= (j * (flash_time + gap_time) +
                                            flash_time))).sum()
            flash_bins = flash_bins * stride
            gap_bins = ((state_time >= (j *
                                        (flash_time + gap_time) + flash_time))
                        & (state_time < (j + 1) *
                           (flash_time + gap_time))).sum()
            gap_bins = gap_bins * stride
            # ls_flash_bins.append(flash_bins)
            # ls_gap_bins.append(gap_bins)
            ls_flash_bins[i, j] = flash_bins
            ls_gap_bins[i, j] = gap_bins

    upper_lim = np.max(ls_flash_bins) + np.max(ls_gap_bins)
    # pre_flash_time must be min of pre_time and input
    pre_flash_time = min(pre_flash_time, pre_time)

    pre_flash_bins = np.round(pre_flash_time * frame_rate).astype(int) * stride

    final_bins = upper_lim + pre_flash_bins

    if verbose:
        print(
            f"Total time: {total_time}s, Total frames: {total_frames}, Frame time: {frame_time:.4f}s"
        )
        print(f"Made state time of shape {state_time.shape}")
        print(f"Number of flashes: {n_flashes}")
        print(
            f"Pre bins: {pre_bins}, Stim bins: {stim_bins}, Tail bins: {tail_bins}"
        )
        print(f"Total bins: {total_bins}")
        print(
            f"Unique flash bins: {np.unique(ls_flash_bins, return_counts=True)}"
        )
        print(f"Unique gap bins: {np.unique(ls_gap_bins, return_counts=True)}")
        print(f"Pre flash time: {pre_flash_time:.4f}s")
        print(f"Pre flash bins: {pre_flash_bins}")

    d_out = {
        'n_trials': n_trials,
        'n_flashes': n_flashes,
        'pre_bins': pre_bins,
        'stim_bins': stim_bins,
        'tail_bins': tail_bins,
        'final_bins': final_bins,
        'pre_flash_bins': pre_flash_bins,
        'flash_bins': np.unique(ls_flash_bins)[0],
        'ls_flash_bins': np.array(ls_flash_bins),
        'ls_gap_bins': np.array(ls_gap_bins)
    }
    return d_out

def compute_image_onset_offset_frames(n_epochs, imgs_per_epoch, pre_frames, epoch_flash_frames, epoch_gap_frames):
    # Utility for computing PresentImages img onset and offset frames
    img_onset_frames = np.zeros((n_epochs, imgs_per_epoch), dtype=int)
    img_offset_frames = np.zeros((n_epochs, imgs_per_epoch), dtype=int)
    
    for i in range(n_epochs):
        img_onset_frames[i, 0] = pre_frames
        img_offset_frames[i, 0] = pre_frames + epoch_flash_frames[i, 0]

        for j in range(imgs_per_epoch-1):
            n_frames_for_img = epoch_flash_frames[i, j] + epoch_gap_frames[i, j]
            img_onset_frames[i, j+1] = img_onset_frames[i, j] + n_frames_for_img
            img_offset_frames[i, j+1] = img_onset_frames[i, j+1] + epoch_flash_frames[i, j+1]
    
    return img_onset_frames, img_offset_frames

def get_present_images_transitions(df_epochs: pd.DataFrame, rb: MEAResponseBlock | SCResponseBlock):
    # Get stage_frame_rate
    stage_frame_rates = get_df_dict_vals(df_epochs, 'frameRate')
    
    # Check there's only one stage frame rate.
    if len(np.unique(stage_frame_rates)) > 1:
        raise ValueError(f'Multiple stage frame rates found: {np.unique(stage_frame_rates)}. Please ensure all epochs have the same stage_frame_rate.')
    else:
        print(f'Using stage frame rate: {stage_frame_rates[0]} Hz')
    stage_frame_rate = stage_frame_rates[0]

    # Get epoch block parameters.
    pre_time = get_df_dict_vals(df_epochs, 'preTime')[0]
    stim_time = get_df_dict_vals(df_epochs, 'stimTime')[0]
    imgs_per_epoch = get_df_dict_vals(df_epochs, 'imagesPerEpoch')[0]
    imgs_per_epoch = int(imgs_per_epoch)

    # Check if 'flashFrames' key exists. If not, use method for old protocol version.
    if 'flashFrames' not in df_epochs.at[0, 'epoch_parameters'].keys():
        print("flashFrames key not found in epoch_parameters. Using old method for transition bins.")
        d_bins = get_old_presentimages_transitions(df_epochs, frame_rate=stage_frame_rate, stride=1, verbose=True)
        epoch_flash_frames = d_bins['ls_flash_bins']
        epoch_gap_frames = d_bins['ls_gap_bins']
        
    else:
        # Frame counts for each epoch
        epoch_flash_frames = get_df_dict_vals(df_epochs, 'flashFrames')
        epoch_gap_frames = get_df_dict_vals(df_epochs, 'gapFrames')
        
        # It's constant for every flash in newer versions, so repeat it imgs_per_epoch times
        epoch_flash_frames = np.repeat(np.array(epoch_flash_frames)[:, np.newaxis], imgs_per_epoch, axis=1)
        epoch_gap_frames = np.repeat(np.array(epoch_gap_frames)[:, np.newaxis], imgs_per_epoch, axis=1)

    # Ensure ints
    epoch_flash_frames = epoch_flash_frames.astype(int)
    epoch_gap_frames = epoch_gap_frames.astype(int)

    pre_frames = np.floor(pre_time * 1e-3 * stage_frame_rate).astype(int)
    # Correct for LCR Stage error of missing first frame by subtracting 1 frame
    pre_frames -= 1
    print(f'Using {pre_frames} pre frames, {stage_frame_rate} Hz stage frame rate.')
    print(f'Protocol had {imgs_per_epoch} images per epoch, {np.unique(epoch_flash_frames)} flash frames, {np.unique(epoch_gap_frames)} gap frames.')

    n_epochs = len(df_epochs)

    img_onset_frames, img_offset_frames = compute_image_onset_offset_frames(
        n_epochs, imgs_per_epoch, pre_frames, 
        epoch_flash_frames, epoch_gap_frames
    )
    # Check if all images were shown or if last few weren't due to state.time duration.
    for i in range(n_epochs):
        n_frames = len(rb.d_timing['frameTimesMs'][i])
        n_expected_stim_frames = np.floor(stim_time/1e3 * stage_frame_rate).astype(int)
        
        # There should at least be this many stim frames, otherwise something went really wrong with delivery or frame time detection.
        if n_frames < n_expected_stim_frames:
            raise ValueError(f'Epoch {i} had only {n_frames} frames, but expected {n_expected_stim_frames} frames for {stim_time} s stimulus time at {stage_frame_rate} Hz stage frame rate.')
        
        # Check if last image offset is within the expected number of frames.
        last_img_offset = img_offset_frames[i, -1]
        if last_img_offset > n_expected_stim_frames:
            # Find last image offset that fits within the available frames.
            last_img_idx = np.where(img_offset_frames[i] <= n_expected_stim_frames)[0][-1]

            # Reset imgs_per_epoch to last img index.
            imgs_per_epoch = last_img_idx + 1
            print(f'\nDue to state.time duration and visibility controller, {n_expected_stim_frames} frames were visible.')
            print(f'But last image offset frame {last_img_offset} exceeds {n_expected_stim_frames} frames.')
            print(f'Epoch {i} thus only showed {imgs_per_epoch} images.')
            print(f'Assuming {imgs_per_epoch} is the correct number of images per epoch for all epochs.\n')

            # Recompute onset/offset frames based on new imgs_per_epoch
            img_onset_frames, img_offset_frames = compute_image_onset_offset_frames(
                n_epochs, imgs_per_epoch, pre_frames, 
                epoch_flash_frames, epoch_gap_frames
            )
        break

    # Get image paths of shape [epoch, images]
    all_image_paths = get_image_paths_across_epochs(df_epochs, n_actual_imgs_per_epoch=imgs_per_epoch)
    u_image_paths, u_repeats = np.unique(all_image_paths.flatten(), return_counts=True)
    repeats = np.unique(u_repeats)
    # if len(repeats) > 1:
        # raise NotImplementedError(f'Found images with different number of repeats: {repeats}. Please ensure all images have the same number of repeats.')
    # repeats = repeats[0]
    print(f'Found {len(u_image_paths)} unique images.')
    print(f'Found {repeats} repeats across all images.')
    print(f'These are named: {u_image_paths[0]} - {u_image_paths[-1]}')

    # Convert frame times ms to samples
    frame_times_samples = []
    avg_frame_rates = []
    for i in range(n_epochs):
        e_fts = rb.d_timing['frameTimesMs'][i] 
        e_fts = np.array(e_fts)
        e_fts *= rb.amp_sample_rate / 1000
        e_fts = np.round(e_fts).astype(int)
        frame_times_samples.append(e_fts)

        avg_frame_rate = rb.amp_sample_rate / np.mean(np.diff(e_fts))
        avg_frame_rates.append(avg_frame_rate)
        
    # Compute average frame rate, mainly for logging.
    avg_frame_rate = np.mean(avg_frame_rates)
    print(f'Average empirical frame rate: {avg_frame_rate:.4f} Hz')

    split_image_paths = []
    split_onset_samples = []
    split_offset_samples = []
    split_epoch_idx = []
    for i in range(n_epochs):
        for j in range(imgs_per_epoch):
            # Indexing explanation
            # eg-if 15 pre_frames, index 14 gives onset of last pre frame, 
            # index 15 gives onset of first flash frame
            t_onset = frame_times_samples[i][img_onset_frames[i, j]] 
            t_offset = frame_times_samples[i][img_offset_frames[i, j]]
            
            split_image_paths.append(all_image_paths[i, j])
            split_onset_samples.append(t_onset)
            split_offset_samples.append(t_offset)
            split_epoch_idx.append(i)
        
    split_image_paths = np.array(split_image_paths)
    split_onset_samples = np.array(split_onset_samples)
    split_offset_samples = np.array(split_offset_samples)
    split_epoch_idx = np.array(split_epoch_idx)

    # Convert onset and offset samples to ms.
    split_onset_ms = split_onset_samples / rb.amp_sample_rate * 1000
    split_offset_ms = split_offset_samples / rb.amp_sample_rate * 1000

    d_out = {
        'all_image_paths': all_image_paths, # 2D array of shape (n_epochs, n_images)
        'repeats': repeats, # Unique numbers of repeats across images
        # Rest are 1D arrays of shape (n_epochs * n_images,) with data for each image trial.
        'u_image_paths': u_image_paths,
        'trial_image_paths': split_image_paths,
        'trial_onset_samples': split_onset_samples,
        'trial_offset_samples': split_offset_samples,
        'trial_onset_ms': split_onset_ms,
        'trial_offset_ms': split_offset_ms,
        'trial_epoch_idx': split_epoch_idx,
    }
    return d_out


def load_all_present_images(df_epochs: pd.DataFrame, rb: MEAResponseBlock | SCResponseBlock,
                            str_parent_path: str=None, 
                            ds_mu: float=10.0, b_only_timing: bool=False)-> dict:
    """
    Load all unique flashed images and their onset/offset timing.
    
    Return dictionary with 'unique_images' array and 'image_names' list.

    Args:
        df_epochs (pd.DataFrame): Dataframe of epoch-block metadata.
        rb (MEAResponseBlock | SCResponseBlock): Responseblock with d_timing and amp_sample_rate
        str_parent_path (str, optional): Parent directory to images.
        ds_mu (float, optional): Downscale images to these um-sized pixels. Defaults to 10.0.
        b_only_timing (bool, optional): Return only transition timing. Defaults to False.

    Raises:
        ValueError: If multiple magnification factors are found across epochs.

    Returns:
        dict: Dictionary with:
            - 'all_image_paths': (n_images,) Full paths to images.
            - 'u_image_paths': (n_unique_images,) Unique image paths.
            - 'repeats': (n_unique_images,) number of repeats for each unique image.
            - 'trial_image_paths': (n_trials,) image paths for each individual image flash.
            - 'trial_onset_samples': (n_trials,)  image flash onsets in samples
            - 'trial_offset_samples': (n_trials,) image flash offsets in samples.
            - 'trial_onset_ms': (n_trials,) image flash onsets in ms
            - 'trial_offset_ms': (n_trials,) image flash offsets in ms.
            - 'trial_epoch_idx': (n_trials,) epoch index for each individual image flash.
            - 'image_data': (n_images, rows, cols) array of images.
            - 'd_display_params': Dictionary with display parameters.
    """
    # Get transition timing and associated image paths.
    d_transitions = get_present_images_transitions(df_epochs, rb)
    if b_only_timing:
        return d_transitions

    if str_parent_path is None:
        raise ValueError('str_parent_path must be provided if b_only_timing is False.')

    # Get display parameters.
    d_epoch_params = df_epochs.iloc[0]['epoch_parameters']
    # Check for magFactor consistency across epochs
    mag_factors = [row['epoch_parameters']['magnificationFactor'] for _, row in df_epochs.iterrows()]
    u_mag_factors = np.unique(mag_factors)
    if len(u_mag_factors) > 1:
        raise ValueError(f'Multiple magnification factors found: {u_mag_factors}. Please ensure all epochs have the same magnificationFactor.')
    
    # Get screen size in (rows, cols)
    screen_size = np.array(d_epoch_params['canvasSize']).astype(int)[::-1]
    print(f'Found screen size: {screen_size} (rows, cols).')
    d_display_params = {
        'screen_size': screen_size,
        'magnification_factor': d_epoch_params['magnificationFactor'],
        'mu_per_pix': d_epoch_params['micronsPerPixel'],
        'ds_mu': ds_mu
    }
    d_display_params['ds_pix'] = int(d_display_params['ds_mu'] / d_display_params['mu_per_pix'])
    d_display_params['ds_screen_size'] = (d_display_params['screen_size']/ d_display_params['ds_pix']).astype(int)

    print(f'Using {ds_mu} um pixel size, {d_display_params["ds_pix"]} pixels per um, '
          f'{d_display_params["ds_screen_size"]} resampled screen size')

    # Load all images at desired downsampling factor.
    all_images = []
    u_image_paths = d_transitions['u_image_paths']
    for str_path in tqdm.tqdm(u_image_paths, desc='Loading images'):
        str_full_path = os.path.join(str_parent_path, str_path)
        img = load_and_process_img(str_full_path, 
                                   screen_size=d_display_params['screen_size'], 
                                   magnification_factor=d_display_params['magnification_factor'], 
                                   ds_factor=d_display_params['ds_pix'])
        all_images.append(img)
    all_images = np.array(all_images)

    d_output = d_transitions
    d_output['image_data'] = all_images
    d_output['d_display_params'] = d_display_params

    return d_output

def make_doves_perturbation_alpha(df_epochs: pd.DataFrame,
    str_pkg_dir: str, exp_name: str, b_noise_only: bool=True):
    # This protocol was basically bugged before 20250805,
    # So regen only for experiments after that date.
    if int(exp_name[:8]) < 20250805:
        raise ValueError('Regen for DovesPerturbationAlpha only valid for experiments after 20250805.')

    import matlab.engine as engine #type: ignore
    if not b_noise_only:
        raise NotImplementedError('Noise + Doves not implemented yet.')
    
    print('Starting matlab engine for stim regen.')
    eng = engine.start_matlab()
    eng.addpath(str_pkg_dir)
    print('Started engine and added pkg to path.')
    
    n_epochs = len(df_epochs)
    noise_lines_epochs = []
    all_fix_indices_epochs = []
    for e_idx in range(n_epochs):
        seed = get_df_dict_vals(df_epochs, 'noiseSeed')[e_idx]
        seed = int(seed)
        # Noise std only used for opacity, not needed for noise generation
        # noise_std = get_df_dict_vals(df_epochs, 'noiseStd')[e_idx]
        # noise_std = float(noise_std)
        
        # num_checks_x needs to be float bc floor(x/2) in matlab, if int it can be wrong eg-103/2 can give 52, when we want 51.
        num_checks_x = get_df_dict_vals(df_epochs, 'numChecksX')[e_idx]
        num_checks_x = float(num_checks_x)
        
        # Also load and type cast other parameters.
        background_intensity = get_df_dict_vals(df_epochs, 'backgroundIntensity')[e_idx]
        background_intensity = float(background_intensity)
        frame_dwell = get_df_dict_vals(df_epochs, 'frameDwell')[e_idx]
        frame_dwell = int(frame_dwell)
        binary_noise = get_df_dict_vals(df_epochs, 'binaryNoise')[e_idx]
        binary_noise = int(binary_noise)
        paired_bars = get_df_dict_vals(df_epochs, 'pairedBars')[e_idx]
        paired_bars = int(paired_bars)
        n_fixations = get_df_dict_vals(df_epochs, 'num_fixations')[e_idx]
        pre_time = get_df_dict_vals(df_epochs, 'preTime')[e_idx]
        pre_time = float(pre_time)
        stim_time = get_df_dict_vals(df_epochs, 'stimTime')[e_idx]
        stim_time = float(stim_time)
        tail_time = get_df_dict_vals(df_epochs, 'tailTime')[e_idx]
        tail_time = float(tail_time)

        pre_frames = np.round(60 * pre_time/1e3).astype(int)
        stim_frames = np.round(60 * stim_time/1e3).astype(int)
        tail_frames = np.round(60 * tail_time/1e3).astype(int)
        # print(pre_frames, stim_frames, tail_frames)

        all_fix_indices = np.arange(1, n_fixations + 1)
        n_frames_per_fix = np.ceil(stim_frames / n_fixations).astype(int)
        all_fix_indices = np.repeat(all_fix_indices, n_frames_per_fix)
        all_fix_indices[all_fix_indices > n_fixations] = n_fixations
        all_fix_indices[all_fix_indices < 1] = 1
        pre_indices = np.ones(pre_frames, dtype=int)
        tail_indices = np.ones(tail_frames, dtype=int)
        all_fix_indices = np.concatenate([pre_indices, all_fix_indices, tail_indices])

        ls_input = [seed, num_checks_x, pre_time, stim_time, tail_time,
                    background_intensity, frame_dwell, binary_noise,
                    1.0, 1.0, 1, paired_bars, 0, 0]
        # print(f'Calling matlab function with inputs: {ls_input}')
        noise_lines = eng.util.getCheckerboardProjectLines(*ls_input, nargout=1)

        noise_lines = np.array(noise_lines)
        noise_lines_epochs.append(noise_lines)
        all_fix_indices_epochs.append(all_fix_indices)
    
    
    eng.quit()
    print('Matlab engine stopped.')

    # Apply jitter
    shifts_epochs = []
    for e_idx in range(n_epochs):
        mu_per_pix = get_df_dict_vals(df_epochs, 'micronsPerPixel')[e_idx]
        num_checks_x = get_df_dict_vals(df_epochs, 'numChecksX')[e_idx]
        num_checks_x = int(num_checks_x)
        stixel_size_um = get_df_dict_vals(df_epochs, 'stixelSize')[e_idx]
        stixel_size_pix = stixel_size_um / mu_per_pix
        grid_size_um = get_df_dict_vals(df_epochs, 'gridSize')[e_idx]
        grid_size_pix = grid_size_um / mu_per_pix
        steps_per_stixel = np.max([np.round(stixel_size_pix/grid_size_pix), 1]).astype(int)
        stixel_shift_pix = np.round(stixel_size_pix / steps_per_stixel).astype(int)
        print(f'Stixel size: {stixel_size_um} um, grid size: {grid_size_um} um, steps per stixel: {steps_per_stixel}, stixel shift: {stixel_shift_pix} pix')
        if steps_per_stixel <= 1:
            continue
            
        frame_dwell = get_df_dict_vals(df_epochs, 'frameDwell')[e_idx]
        frame_dwell = int(frame_dwell)

        seed = get_df_dict_vals(df_epochs, 'noiseSeed')[e_idx]
        seed = int(seed)
        np.random.seed(seed)

        # (stixel, frames)
        noise_lines = noise_lines_epochs[e_idx]
        # Upsample by steps_per_stixel
        upsample_size = (int(num_checks_x * steps_per_stixel), noise_lines.shape[1])
        # Upsample to pixel space
        # upsample_size = (int(np.round(num_checks_x * stixel_size_pix)), noise_lines.shape[1])
        print(f'Upsampling noise lines to {upsample_size}')
        noise_lines = cv2.resize(noise_lines, upsample_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        n_frames = noise_lines.shape[1]
        shifts = []
        # Start frame count at 2 bc shift for count 1 is skipped in protocol
        for f_count in range(2,n_frames+1):
            if f_count % frame_dwell == 0:
                x_shift_stix = np.round(np.random.rand() * (steps_per_stixel-1)).astype(int)
                shifts.append(x_shift_stix)
                # If in pixel space, would multiply by stixel_shift_pix
                # x_shift_stix = int(x_shift_stix * stixel_shift_pix)
                if x_shift_stix == 0:
                    continue
                noise_lines[x_shift_stix:, f_count-1] = noise_lines[:-x_shift_stix, f_count-1]
        noise_lines_epochs[e_idx] = noise_lines
        shifts_epochs.append(shifts)
    noise_lines_epochs = np.array(noise_lines_epochs)
    all_fix_indices_epochs = np.array(all_fix_indices_epochs)
    shifts_epochs = np.array(shifts_epochs)
    d_output = {
        'noise_lines': noise_lines_epochs, # (epochs, stixels*steps_per_stixel, frames)
        'all_fix_indices': all_fix_indices_epochs,
        'd_stim_timing': {
            'pre_time': pre_time,
            'stim_time': stim_time,
            'tail_time': tail_time,
            'pre_frames': pre_frames,
            'stim_frames': stim_frames,
            'tail_frames': tail_frames
        },
        'shifts': shifts_epochs # (epochs, frames-1)
    }

    
    return d_output

def make_checkerboard_noise_project(df_epochs: pd.DataFrame, exp_name:str, str_pkg_dir: str, b_noise_only: bool=True):
    exp_name = int(exp_name[:8])
    import matlab.engine #type: ignore
    print('Starting matlab engine for stim regen.')
    eng = matlab.engine.start_matlab()
    eng.addpath(str_pkg_dir)
    print('Started engine and added pkg to path.')
    preTime = matlab.double(df_epochs.loc[0,'preTime'])
    tailTime = matlab.double(df_epochs.loc[0,'tailTime'])
    stimTime = matlab.double(df_epochs.loc[0,'stimTime'])
    noiseSeeds = matlab.double([df_epochs.loc[i,'noiseSeed'] for i in df_epochs.index])
    numChecksXs = matlab.double([df_epochs['epoch_parameters'][i]['numChecksX'] for i in df_epochs.index])
    backgroundIntensity = matlab.double([df_epochs.loc[0, 'epoch_parameters']['backgroundIntensity']])
    frameDwell = matlab.double([df_epochs.loc[0, 'epoch_parameters']['frameDwell']])
    binaryNoise = matlab.double([df_epochs.loc[0, 'epoch_parameters']['binaryNoise']])
    noiseStdv = matlab.double([df_epochs.loc[0, 'epoch_parameters']['noiseStdv']])
    if b_noise_only:
        if exp_name<20250806:
            backgroundRatios = matlab.double([0 for _ in df_epochs.index])
        else:
            backgroundRatios = matlab.double([1.0 for _ in df_epochs.index])
    else:
        backgroundRatios = matlab.double([df_epochs.loc[i,'epoch_parameters']['backgroundRatio'] for i in df_epochs.index])
    backgroundFrameDwells = matlab.double([df_epochs.loc[i,'epoch_parameters']['backgroundFrameDwell'] for i in df_epochs.index])
    pairedBars = matlab.double([df_epochs.loc[0, 'epoch_parameters']['pairedBars']])
    if b_noise_only:
        noSplitField = matlab.double([1.0])
    else:
        noSplitField = matlab.double([df_epochs.loc[0, 'epoch_parameters']['noSplitField']])
    contrastJumps = matlab.double(df_epochs.loc[0, 'epoch_parameters']['contrastJumps'])
    numChecksYs = matlab.double([df_epochs['epoch_parameters'][i]['numChecksY'] for i in df_epochs.index])
    # if exp_name < 20250806:
    stimulus, line_mat, contrast_mat = eng.util.regenerateCheckerboardProject(exp_name, preTime,  tailTime, stimTime, noiseSeeds, numChecksXs, backgroundIntensity, frameDwell, binaryNoise, noiseStdv, backgroundRatios, backgroundFrameDwells, pairedBars, noSplitField, contrastJumps, numChecksYs, nargout=3);
    stimulus = np.array(stimulus);
    line_mat = np.array(line_mat);
    contrast_mat = np.array(contrast_mat);
    eng.quit()

    canvas_size_pix = df_epochs.loc[0,'epoch_parameters']['canvasSize']
    stixel_size_um = df_epochs.loc[0,'epoch_parameters']['stixelSize']
    mu_per_pix = df_epochs.loc[0,'microns_per_pixel']
    canvas_size_stix = (int(np.round(canvas_size_pix[0] * mu_per_pix / stixel_size_um)),
                            int(np.round(canvas_size_pix[1] * mu_per_pix / stixel_size_um)))
    stixel_size_pix = int(np.round(stixel_size_um / mu_per_pix))
    x_offset_pix = df_epochs.loc[0,'epoch_parameters']['xOffset']
    y_offset_pix = df_epochs.loc[0,'epoch_parameters']['yOffset']
    x_offset_stix = int(np.round(x_offset_pix / stixel_size_pix))
    y_offset_stix = int(np.round(y_offset_pix / stixel_size_pix))

    stimulus_cropped = np.ones_like(stimulus) * int((255*df_epochs.loc[0,'epoch_parameters']['backgroundIntensity']))
    lines_cropped = np.ones_like(line_mat) * df_epochs.loc[0,'epoch_parameters']['backgroundIntensity']
    if x_offset_pix > 0:
        offset = np.abs(x_offset_stix)
        stimulus_cropped[:, offset:, :, :] = stimulus[:, :-offset, :, :]
        lines_cropped[offset:, :, :] = line_mat[:-offset, :, :]
    elif x_offset_pix < 0:
        offset = np.abs(x_offset_stix)
        stimulus_cropped[:, :-offset, :, :] = stimulus[:, offset:, :, :]
        lines_cropped[:-offset, :, :] = line_mat[offset:, :, :]
    else:
        stimulus_cropped = stimulus
        lines_cropped = line_mat
    if y_offset_pix != 0:
        raise NotImplementedError('Y offset cropping not implemented yet.')
    
    stim_transitions = []
    for e_idx in df_epochs.index:
        interval = df_epochs.at[e_idx, 'backgroundFrameDwell']
        frame_transitions_ls = []
        for i in range(lines_cropped.shape[1]):
            if i % interval == 0:
                frame_transitions_ls.append(i)
        stim_transitions.append(frame_transitions_ls)

    d_output = {
        'stim_frames': stimulus_cropped,
        'line_mat': lines_cropped,
        'contrast_mat': contrast_mat,
    }
    return d_output




def make_spot_image(ht, wt, center_row, center_col, diam, background, intensity):
    Y, X = np.ogrid[:ht, :wt]
    dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
    mask = dist_from_center <= diam / 2
    img = np.zeros((ht, wt), dtype=np.float32) + background
    img[mask] = intensity
    # Scale to (-1, 1)
    img = (img - 0.5) * 2
    return img


def make_expanding_spots(df_epochs: pd.DataFrame, ds_mu: float=10.0):
    # Get display parameters
    d_epoch_params = df_epochs.iloc[0]['epoch_parameters']
    # Get screen size in (rows, cols)
    screen_size = np.array(d_epoch_params['canvasSize']).astype(int)[::-1]
    d_display_params = {
        'screen_size': screen_size,
        'mu_per_pix': d_epoch_params['micronsPerPixel'],
        'ds_mu': ds_mu
    }
    d_display_params['ds_pix'] = int(d_display_params['ds_mu'] / d_display_params['mu_per_pix'])
    d_display_params['ds_screen_size'] = (d_display_params['screen_size']/ d_display_params['ds_pix']).astype(int)
    img_size = d_display_params['ds_screen_size']

    # Generate images for all epochs
    all_images = []
    for e_idx in tqdm.tqdm(df_epochs.index, desc='Generating images'):
        spot_diam_um = get_df_dict_vals(df_epochs, 'currentSpotSize')[e_idx]
        spot_diam_ds = int(np.round(spot_diam_um / d_display_params['ds_mu']))
        # Center offset is in x, y pixels. Get in row, col format.
        center_offset_pix = get_df_dict_vals(df_epochs, 'centerOffset')[e_idx][::-1]
        center_offset_ds = (center_offset_pix / d_display_params['ds_pix']).astype(int)
        center_row = int(np.round(img_size[0] / 2 + center_offset_ds[0]))
        center_col = int(np.round(img_size[1] / 2 + center_offset_ds[1]))

        background = get_df_dict_vals(df_epochs, 'backgroundIntensity')[e_idx]
        intensity = get_df_dict_vals(df_epochs, 'spotIntensity')[e_idx]

        img = make_spot_image(img_size[0], img_size[1], center_row, center_col, spot_diam_ds, background, intensity)
        all_images.append(img)
    all_images = np.array(all_images)

    d_output = {
        'image_data': all_images,
        'd_display_params': d_display_params
    }

    return d_output


def load_doves_img(img_path, shape=(1024, 1536)):
    # Read in the image as ieee-be
    with open(img_path, 'rb') as f:
        data = f.read()
    data_f = np.frombuffer(data, dtype='uint16').byteswap()

    img = data_f.reshape(shape)
    img = img.astype(float)   
    img = img / img.max()

    # Convert to unsigned 8-bit integer
    # img = img*255
    # img = img.astype(np.uint8)

    return img


def make_single_doves_movie(
        srs_epoch: pd.Series,
        D_DOVES_DATA: dict, STR_DOVES_IMG_DIR: str,
        d_display: dict,
        b_ds_to_stix=True, stix_dims=(127, 203),
        verbose=False
    ):
    # Fetch stimulus params
    n_mag_factor = srs_epoch['epoch_parameters']['magnificationFactor']
    n_stim_idx = srs_epoch['epoch_parameters']['stimulusIndex']
    n_stim_idx = int(n_stim_idx)
    wait_time_s = srs_epoch['epoch_parameters']['waitTime'] / 1e3
    duration_s = srs_epoch['epoch_parameters']['stimTime'] / 1e3
    duration_frames = int(np.round(duration_s * d_display['stage_frame_rate']))
    wait_frames = int(np.round(wait_time_s * d_display['stage_frame_rate']))
    if verbose:
        print(f'Movie params: ')
        print(f'    magnificationFactor: {n_mag_factor}')
        print(f'    stimulusIndex: {n_stim_idx} (matlab 1-based index)')
        print(f'    waitTime: {wait_time_s:.2f} s')
        print(f'    stimTime: {duration_s:.2f} s')
        print(f'    wait_frames: {wait_frames}')
        print(f'    total_frames: {duration_frames}')

    n_stim_idx -= 1 # from matlab to python indexing
    img_size = np.array([1024, 1536])
    scene_size = img_size * n_mag_factor
    scene_size = np.round(scene_size).astype(int)
    screen_size = np.array([d_display['n_ht'], d_display['n_wt']], dtype=int)

    p0 = scene_size / 2
    x_vals = np.arange(-screen_size[1] / 2, screen_size[1] / 2)
    y_vals = np.arange(-screen_size[0] / 2, screen_size[0] / 2)

    idx_imgname = 2
    idx_eyeX = 3
    idx_eyeY = 4
    # idx_frozenX = 7
    # idx_frozenY = 8
    # micronsPerArcmin = 3.3
    xTraj = D_DOVES_DATA['FEMdata'][0, n_stim_idx][idx_eyeX][0]
    yTraj = D_DOVES_DATA['FEMdata'][0, n_stim_idx][idx_eyeY][0]

    timeTraj = np.arange(0,len(xTraj)) / 200

    # Normalize xy trajectories relative to center of image
    xTraj = -(xTraj - img_size[1] / 2)
    yTraj = yTraj - img_size[0] / 2

    # Negate xTraj bc here we are moving screen, protocol moves center of image
    xTraj = -xTraj
    # yTraj stays bc (0,0) is bottom left in display. Positive yTraj moves image up, screen down.

    # Convert from arcmin to pixels. 1 VH pixel = 1 arcmin = 3.3 um on monkey retina.
    xTraj = xTraj * 3.3 / d_display['mu_per_pixel']
    yTraj = yTraj * 3.3 / d_display['mu_per_pixel']

    imageName = D_DOVES_DATA['FEMdata'][0, n_stim_idx][idx_imgname][0]
    str_img_path = os.path.join(STR_DOVES_IMG_DIR, imageName)
    if verbose:
        print(f'Loading image {imageName} from {str_img_path}')
    img = load_doves_img(str_img_path, shape=img_size)

    img_mean = np.mean(img)
    # Upscale img to scene_size with nn
    img = cv2.resize(img, tuple(scene_size[::-1].astype(int)), interpolation=cv2.INTER_NEAREST)

    # Final movie dims
    n_traj_frames = duration_frames - wait_frames
    if not b_ds_to_stix:
        stix_dims = screen_size
    frames = np.zeros((n_traj_frames, stix_dims[0], stix_dims[1]))
    frames = frames + img_mean

    if verbose:
        print(f'Creating movie for {imageName} with shape {img.shape} and scene size {scene_size}, fitting in {screen_size}')
    if b_ds_to_stix:
        print(f'And downsampling to stix dims {stix_dims}.')
    ls_px = []
    ls_py = []
    for f_idx in tqdm.trange(n_traj_frames, desc='Generating frames'):
        t = f_idx * 1/d_display['mean_frame_rate']
        if t > timeTraj.max():
            t = timeTraj.max()
        dx = np.interp(t, timeTraj, xTraj)
        dy = np.interp(t, timeTraj, yTraj)

        # Update position
        p = p0 + np.array([dy, dx]).squeeze()
        ls_px.append(p[1])
        ls_py.append(p[0])
        x_idx = np.round(p[1] + x_vals).astype(int)
        y_idx = np.round(p[0] + y_vals).astype(int)

        x_good = (x_idx >= 0) & (x_idx < scene_size[1])
        y_good = (y_idx >= 0) & (y_idx < scene_size[0])
        
        frame = np.zeros(screen_size) + img_mean
        assign_idx = np.ix_(np.where(y_good)[0], np.where(x_good)[0])
        frame[assign_idx] = img[y_idx[y_good], :][:, x_idx[x_good]]
        
        if b_ds_to_stix:
            frame = cv2.resize(frame, stix_dims[::-1], interpolation=cv2.INTER_AREA)
        
        frames[f_idx] = frame

    # Add wait frames
    if verbose:
        print(f'Adding {wait_frames} wait frames to {frames.shape} for {wait_time_s} s.')
    full_frames = np.zeros((duration_frames, frames.shape[1], frames.shape[2]))
    full_frames[:wait_frames] = frames[0]
    full_frames[wait_frames:] = frames
    frames = full_frames
    if verbose:
        print(f'New frames shape: {frames.shape}')

    # Rescale from (0,1) to (-1, 1)
    frames = (frames*2 - 1)
    if verbose:
        print(f'Movie scaled to (-1, 1) contrast: min {frames.min():.2f}, max {frames.max():.2f}')

    d_out = {
        'frames': frames, # (frames, rows, cols)
        'image_name': imageName,
        'image': img, # (1024, 1536) original image, upscaled to scene size by mag factor
        'center_x_pos': ls_px,
        'center_y_pos': ls_py
    }
    
    return d_out

def make_all_doves_movies(
        df_epochs, str_manookin_pkg_dir, d_display,
        b_ds_to_stix=True, stix_dims=(127, 203), verbose=True
    ):
    D_DOVES_DATA = loadmat(os.path.join(str_manookin_pkg_dir, 'resources', 'doves', 'dovesFEMstims20160826.mat'))
    STR_DOVES_IMG_DIR = os.path.join(str_manookin_pkg_dir, 'resources', 'doves', 'images')
    
    d_out = {
        'movies': [],
        'image_names': [],
        'images': [],
        'center_x_pos': [],
        'center_y_pos': []
    }
    
    n_epochs = len(df_epochs)
    for i in range(n_epochs):
        print(f'Processing epoch {i+1}/{n_epochs}')
        d_movie = make_single_doves_movie(
            df_epochs.iloc[i], D_DOVES_DATA, STR_DOVES_IMG_DIR,
            d_display, b_ds_to_stix, stix_dims, verbose
        )
        d_out['movies'].append(d_movie['frames'])
        d_out['image_names'].append(d_movie['image_name'])
        d_out['images'].append(d_movie['image'])
        d_out['center_x_pos'].append(d_movie['center_x_pos'])
        d_out['center_y_pos'].append(d_movie['center_y_pos'])

    # If all movie have same frames, can stack into array
    frame_counts = [movie.shape[0] for movie in d_out['movies']]
    if len(np.unique(frame_counts)) == 1:
        print(f'All movies have same frame count of {frame_counts[0]}, stacking into array.')
        # (n_epochs, frames, rows, cols)
        d_out['movies'] = np.array(d_out['movies'])
    else:
        print(f'Movies have different frame counts: {frame_counts}, keeping as list.')
    
    return d_out