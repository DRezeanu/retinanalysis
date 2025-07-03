import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm
IMPLEMENTED_PROTOCOLS = [
    # 'manookinlab.protocols.FastNoise',
    'manookinlab.protocols.SpatialNoise',
    # 'manookinlab.protocols.DovesMovie'
]

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

