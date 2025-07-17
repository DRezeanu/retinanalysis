import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm
import pandas as pd
import os
import cv2

def make_spatial_noise(df_epochs: pd.DataFrame, center_row: int=None, center_col: int=None, n_pad: int=None):
    # Create noise movies by epochs
    ls_frames = []
    for e_idx in tqdm.tqdm(df_epochs.index):
        fts = df_epochs.at[e_idx, 'frame_times_ms']
        pre_time = df_epochs.at[e_idx, 'preTime']
        unique_time = df_epochs.at[e_idx, 'epoch_parameters']['uniqueTime']
        total_frames = len(fts)-1
        unique_frames = len(np.where(np.logical_and((fts > pre_time),(fts <= pre_time+unique_time)))[0])
        
        d_e_params = df_epochs.at[e_idx, 'epoch_parameters']
        d_meta = {
            'numXStixels': d_e_params['numXStixels'],
            'numYStixels': d_e_params['numYStixels'],
            'numXChecks': d_e_params['numXChecks'],
            'numYChecks': d_e_params['numYChecks'],
            'chromaticClass': d_e_params['chromaticClass'],
            'unique_frames': unique_frames,
            'repeat_frames': total_frames - unique_frames,
            'stepsPerStixel': d_e_params['stepsPerStixel'],
            'seed': d_e_params['seed'],
            'frameDwell': d_e_params['frameDwell'],
        }
        if 'gaussianFilter' in d_e_params:
            d_meta['gaussianFilter'] = d_e_params['gaussianFilter']
        if 'filterSdStixels' in d_e_params:
            d_meta['filterSdStixels'] = d_e_params['filterSdStixels']
        frames = get_spatial_noise_frames(**d_meta)
        
        ls_frames.append(frames)
    frames = np.array(ls_frames)
    if center_row is not None:
        # Crop frames around the cell center
        frames = frames[:, :, center_row-n_pad:center_row+n_pad+1, center_col-n_pad:center_col+n_pad+1, :]
    return frames

def get_spatial_noise_frames(numXStixels: int, 
                        numYStixels: int, 
                        numXChecks: int, 
                        numYChecks: int, 
                        chromaticClass: str, 
                        unique_frames: int, 
                        repeat_frames: int,
                        stepsPerStixel: int, 
                        seed: int, 
                        frameDwell: int=1,
                        gaussianFilter: bool=False,
                        filterSdStixels: float=1.0):
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
    # Seed the random number generator.
    np.random.seed( seed )

    # First, generate the larger grid of stixels.
    if (chromaticClass == 'BY'):
        tfactor = 2
    elif (chromaticClass == 'RGB'):
        tfactor = 3
    else: # Black/white checks
        tfactor = 1

    # Get the size of the time dimension; expands for RGB, etc.
    numFrames = unique_frames + repeat_frames
    tsize = np.ceil(numFrames*tfactor/frameDwell).astype(int)
    usize = np.ceil(unique_frames*tfactor/frameDwell).astype(int)
    rsize = np.ceil(repeat_frames*tfactor/frameDwell).astype(int)
    
    if (tfactor == 2 and (tsize % 2) != 0):
        tsize += 1
        rsize += 1
    
    # Generate the random grid of stixels.
    gridValues = np.zeros((tsize, numXStixels*numYStixels), dtype=np.float32)
    gridValues[:usize,:] = np.random.rand(usize, numXStixels*numYStixels)
    # Repeating sequence.
    if repeat_frames > 0:
        # Reseed the generator.
        np.random.seed( 1 )
        gridValues[usize:,:] = np.random.rand(rsize, numXStixels*numYStixels)
    gridValues = np.reshape(gridValues, (tsize, numXStixels, numYStixels))
    gridValues = np.transpose(gridValues, (0, 2, 1))
    gridValues = np.round(gridValues)
    gridValues = (2*gridValues-1).astype(np.float32) # Convert to contrast

    # Filter the stixels if indicated.
    if gaussianFilter:
        for i in range(tsize):
            # frame_tmp = gaussian_filter(gridValues[i,:,:], sigma=filterSdStixels, order=0, mode='wrap', radius=np.ceil(2*filterSdStixels).astype(int))
            frame_tmp = gaussian_filter(gridValues[i,:,:], sigma=filterSdStixels, order=0, mode='wrap')
            gridValues[i,:,:] = 0.5*frame_tmp/np.std(frame_tmp)
        gridValues[gridValues > 1.0] = 1.0
        gridValues[gridValues < -1.0] = -1.0

    # Translate to the full grid
    fullGrid = np.zeros((tsize,numYStixels*stepsPerStixel,numXStixels*stepsPerStixel), dtype=np.float32)
    # fullGrid = np.zeros((tsize,numYStixels*stepsPerStixel,numXStixels*stepsPerStixel), dtype=np.uint8)

    for k in range(numYStixels*stepsPerStixel):
        yindex = np.floor(k/stepsPerStixel).astype(int)
        for m in range(numXStixels*stepsPerStixel):
            xindex = np.floor(m/stepsPerStixel).astype(int)
            fullGrid[:, k, m] = gridValues[:, yindex, xindex]

    # Generate the motion trajectory of the larger stixels.
    np.random.seed( seed ) # Re-seed the number generator

    # steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps[:,0] = (stepsPerStixel-1) - steps[:,0]
    # steps = (stepsPerStixel-1) - np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    # Get the frame values for the finer grid.
    # frameValues = np.zeros((tsize,numYChecks,numXChecks),dtype=np.uint8)
    frameValues = np.zeros((tsize,numYChecks,numXChecks),dtype=np.float32)
    for k in range(tsize):
        x_offset = steps[np.floor(k/tfactor).astype(int), 0].astype(int)
        y_offset = steps[np.floor(k/tfactor).astype(int), 1].astype(int)
        frameValues[k,:,:] = fullGrid[k, y_offset : numYChecks+y_offset, x_offset : numXChecks+x_offset]

    # Create your output stimulus. (t, y, x, color)
    stimulus = np.zeros((np.ceil(tsize/tfactor).astype(int),numYChecks,numXChecks,3), dtype=np.float32)

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
    # return stimulus

    # Deal with the frame dwell.
    if frameDwell > 1:
        stim = np.zeros((numFrames,numYChecks,numXChecks,3), dtype=np.float32)
        for k in range(numFrames):
            idx = np.floor(k / frameDwell).astype(int)
            stim[k,:,:,:] = stimulus[idx,:,:,:]
        return stim
    else:
        return stimulus


# Functions for PresentImages protocol
def load_and_process_img(str_img,screen_size = np.array([1140, 1824]), # rows, cols
                        magnification_factor = 8,
                        ds_factor: int=3, rescale: bool=True,
                        verbose: bool=False):
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


def get_image_paths_across_epochs(df_epochs: pd.DataFrame):
    all_image_names = df_epochs['imageName'].values
    all_image_names = np.concatenate([x.split(',') for x in all_image_names])

    all_image_folders = df_epochs['folder'].values
    all_image_folders = np.concatenate([x.split(',') for x in all_image_folders])

    all_image_paths = []
    for folder, name in zip(all_image_folders, all_image_names):
        str_path = os.path.join(folder, name)
        all_image_paths.append(str_path)
    all_image_paths = np.array(all_image_paths)
    return all_image_paths


def load_all_present_images(df_epochs: pd.DataFrame, str_parent_path: str, 
                        ds_mu: float=10.0):
    # Given dataframe of epoch metadata, load all unique flashed images.
    # Return dataarray of unique images with coords of image name.
    # Or alternatively dictionary with 'unique_images' array and 'image_names' list.
    # Let's start with dictionary.
    # ds_mu = 10 => Downscale to 10x10 um pixels

    # Get unique image paths
    print('Loading flashed images...')
    all_image_paths = get_image_paths_across_epochs(df_epochs)
    u_image_paths, u_repeats = np.unique(all_image_paths, return_counts=True)
    repeats = np.unique(u_repeats)
    print(f'Found {len(u_image_paths)} unique images to load.')
    print(f'Found {repeats} repeats across all images.')
    print(f'These are named: {u_image_paths[0]} - {u_image_paths[-1]}')

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

    all_images = []
    for str_path in tqdm.tqdm(u_image_paths, desc='Loading images'):
        str_full_path = os.path.join(str_parent_path, str_path)
        img = load_and_process_img(str_full_path, 
                                   screen_size=d_display_params['screen_size'], 
                                   magnification_factor=d_display_params['magnification_factor'], 
                                   ds_factor=d_display_params['ds_pix'])
        all_images.append(img)
    all_images = np.array(all_images)

    d_output = {
        'image_data': all_images,
        'all_image_paths': all_image_paths,
        'u_image_paths': u_image_paths,
        'repeats': repeats,
        'd_display_params': d_display_params
    }

    return d_output