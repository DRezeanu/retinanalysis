import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm
import pandas as pd
import os
import cv2

def get_df_dict_vals(df, key, col_name='epoch_parameters'):
    vals = np.array([d[key] for d in df[col_name].values])
    return vals

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
    np.random.seed( int(seed) )

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
    gridValues = np.zeros((tsize, int(numXStixels*numYStixels)), dtype=np.float32)
    gridValues[:usize,:] = np.random.rand(usize, int(numXStixels*numYStixels))
    # Repeating sequence.
    if repeat_frames > 0:
        # Reseed the generator.
        np.random.seed( 1 )
        gridValues[usize:,:] = np.random.rand(rsize, int(numXStixels*numYStixels))
    gridValues = np.reshape(gridValues, (tsize, int(numXStixels), int(numYStixels)))
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
    fullGrid = np.zeros((tsize,int(numYStixels*stepsPerStixel),int(numXStixels*stepsPerStixel)), dtype=np.float32)
    # fullGrid = np.zeros((tsize,numYStixels*stepsPerStixel,numXStixels*stepsPerStixel), dtype=np.uint8)

    for k in range(int(numYStixels*stepsPerStixel)):
        yindex = np.floor(k/stepsPerStixel).astype(int)
        for m in range(int(numXStixels*stepsPerStixel)):
            xindex = np.floor(m/stepsPerStixel).astype(int)
            fullGrid[:, k, m] = gridValues[:, yindex, xindex]

    # Generate the motion trajectory of the larger stixels.
    np.random.seed( int(seed) ) # Re-seed the number generator

    # steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps[:,0] = (stepsPerStixel-1) - steps[:,0]
    # steps = (stepsPerStixel-1) - np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    # Get the frame values for the finer grid.
    # frameValues = np.zeros((tsize,numYChecks,numXChecks),dtype=np.uint8)
    frameValues = np.zeros((tsize,int(numYChecks),int(numXChecks)),dtype=np.float32)
    for k in range(tsize):
        x_offset = steps[np.floor(k/tfactor).astype(int), 0].astype(int)
        y_offset = steps[np.floor(k/tfactor).astype(int), 1].astype(int)
        frameValues[k,:,:] = fullGrid[k, y_offset : int(numYChecks)+y_offset, x_offset : int(numXChecks)+x_offset]

    # Create your output stimulus. (t, y, x, color)
    stimulus = np.zeros((np.ceil(tsize/tfactor).astype(int),int(numYChecks),int(numXChecks),3), dtype=np.float32)

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
        stim = np.zeros((numFrames,int(numYChecks),int(numXChecks),3), dtype=np.float32)
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
        for f_count in range(1,n_frames+1):
            if f_count % frame_dwell == 0:
                x_shift_stix = np.round(np.random.rand() * (steps_per_stixel-1)).astype(int)
                # If in pixel space, would multiply by stixel_shift_pix
                # x_shift_stix = int(x_shift_stix * stixel_shift_pix)
                if x_shift_stix == 0:
                    continue
                # I don't exactly get it, but below is correct as it matches screen cap.
                noise_lines[:-x_shift_stix, f_count-1] = noise_lines[x_shift_stix:, f_count-1]
        noise_lines_epochs[e_idx] = noise_lines

    noise_lines_epochs = np.array(noise_lines_epochs)
    all_fix_indices_epochs = np.array(all_fix_indices_epochs)
    d_output = {
        'noise_lines': noise_lines_epochs,
        'all_fix_indices': all_fix_indices_epochs,
        'd_stim_timing': {
            'pre_time': pre_time,
            'stim_time': stim_time,
            'tail_time': tail_time,
            'pre_frames': pre_frames,
            'stim_frames': stim_frames,
            'tail_frames': tail_frames
        }
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


