import numpy as np
from scipy.signal import filtfilt, butter

def detect_flips(
    trace,
    sample_rate,
    expected_frame_rate=59.94,
    hysteresis=0.4,
    min_dwell=0.5,
    f_cutoff=120.0,
):
    """Hysteresis (Schmitt) trigger for finding frame monitor transitions.

    The frame monitor stripe alternates black/white once per frame, so every
    transition marks a frame boundary and consecutive transitions are one
    frame apart.

    Args:
        trace: Raw frame monitor samples for one epoch.
        sample_rate: sample rate in Hz. Enters only as a conversion factor. Every
            tolerance below is expressed in frame periods or in fractions of the
            measured signal swing, so 1 kHz and 10 kHz recordings behave the same.
        expected_frame_rate: Expected frame rate in Hz. Used to size the
            glitch-rejection window; the actual frame period is fit by `fit_frame_times`.
        hysteresis: Width of the deadband between the two thresholds, as a fraction of
            the measured swing.
        min_dwell: Minimum spacing between accepted transitions, in frame periods.
        f_cutoff: Low-pass cutoff in Hz applied before thresholding. `filtfilt` is
            zero-phase, so this adds no timing bias. None disables filtering.
            120.0 is default that gives best results with noisy FM trace.

    Returns:
        flip_times: Transition times in ms, with sub-sample precision.
        frame_idx: Frame number of each transition, counting from 0.
            Dropped frames advance this value by more than 1.
        rising: True where the transition is dark -> light.
    """
    x = np.asarray(trace, dtype=float)

    # Low pass filter the trace to get rid of sub-frame fluctuations that cause problems
    # with detection
    if f_cutoff is not None:
        b, a = butter(4, f_cutoff / (sample_rate / 2.0), btype='low', output='ba') #type: ignore
        x = filtfilt(b, a, x)

    # Thresholds from the signal's own levels. Percentiles rather than
    # min/max so a single noise spike can't set the scale.
    lo, hi = np.percentile(x, [2, 98])
    mid = 0.5 * (lo + hi)
    thr_hi = mid + 0.5 * hysteresis * (hi - lo)
    thr_lo = mid - 0.5 * hysteresis * (hi - lo)

    # Schmitt trigger, vectorised. Outside the deadband the state is fully
    # determined regardless of history; inside it, the rule is just "hold
    # whatever you last were". So label the unambiguous samples, then
    # forward-fill their index across the deadband.
    defined = np.where(x >= thr_hi, 1, np.where(x <= thr_lo, 0, -1))
    first = np.argmax(defined >= 0)              # seed: first unambiguous sample
    src = np.where(defined >= 0, np.arange(x.size), first)
    np.maximum.accumulate(src, out=src)
    state = defined[src].astype(bool)

    confirmed = np.flatnonzero(np.diff(state))
    if confirmed.size == 0:
        return (np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=bool))
    rising = state[confirmed + 1]

    # Hysteresis confirms a transition *late*, and late in a direction-
    # dependent way, which would bias rising and falling edges oppositely.
    # So hysteresis decides *whether* an edge is real; the last mid-level
    # crossing at or before it decides *when* it happened.
    mid_cross = np.flatnonzero(np.diff(x >= mid))
    j = mid_cross[np.clip(np.searchsorted(mid_cross, confirmed, 'right') - 1, 0, None)]

    # Sub-sample timing: samples j and j+1 straddle mid, so interpolate.
    # At 1 kHz there are only ~16.7 samples per frame, so this is worth
    # ~1 ms of jitter per edge.
    frac = (mid - x[j]) / (x[j + 1] - x[j])
    flip_samples = j + frac

    # Reject glitch edges arriving too soon after the last accepted one.
    # Comparing against the last *kept* edge (not the last examined one)
    # means a whole cluster collapses to its earliest member, which removes
    # an even number of edges and so preserves rising/falling parity.
    min_gap = min_dwell * sample_rate / expected_frame_rate
    keep, last = [], -np.inf
    for i, s in enumerate(flip_samples):
        if s - last >= min_gap:
            keep.append(i)
            last = s
    flip_samples = flip_samples[keep]
    rising = rising[keep]

    # Number each transition. A dropped frame isn't an error to repair --
    # it's just an interval worth 2 (or 3...), and the running count
    # absorbs it. The median is robust because drops are a minority.
    intervals = np.diff(flip_samples)
    n_frames = np.round(intervals / np.median(intervals)).astype(int)
    frame_idx = np.concatenate([[0], np.cumsum(n_frames)])

    return flip_samples / sample_rate * 1e3, frame_idx, rising


def fit_frame_times(flip_times, frame_idx, rising=None):
    """Fit a uniform frame grid to the measured transitions.

    Regressing flip time on integer frame index uses every transition in the
    epoch and is unaffected by dropped frames, since those are already
    carried by `frame_idx`. With ~10^4 measurements constraining two
    parameters, the fitted grid is better determined than any single
    measured edge -- provided the residuals are flat (no clock drift) and
    the frame index is right. Both of these things appear to be true at least
    across three experiments I checked directly.

    Args:
        flip_times: detected flip times using the detect_flips() function
        frame_idx: list of frame indices generated with detect_flips()
        rising: Optional true-false array where all rising edges (black to
            white transitions) are True, and falling edges are False.

    Returns:
        dict with the fitted period/rate, the per-flip residuals, an
            outlier mask and the dropped-frame count
    """
    period, t0 = np.polyfit(frame_idx, flip_times, 1)
    resid = flip_times - (t0 + period * frame_idx)

    # Rising and falling edges sit on their own grids if the duty cycle
    # isn't exactly 50%, so measure each against its own band centre.
    if rising is None:
        band = np.full(resid.shape, np.median(resid))
    else:
        band = np.where(rising, np.median(resid[rising]), np.median(resid[~rising]))

    return dict(
        period=period,                       # ms per frame
        t0=t0,                               # ms, time of frame 0
        rate=1e3 / period,                   # Hz
        resid=resid,
        band=band,
        outliers=np.abs(resid - band) > 0.25 * period,
        drops=int((np.diff(frame_idx) != 1).sum()),
    )
