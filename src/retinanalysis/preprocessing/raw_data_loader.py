from dataclasses import dataclass
import numpy as np
import bin2py as b2p
from pathlib import Path
import json

@dataclass(slots=True, frozen=True)
class RawDataContainer:
    array_id: int
    n_electrodes: int
    n_samples: int
    electrode_data: np.ndarray | None
    ttl_data: np.ndarray
    epoch_starts: np.ndarray
    epoch_ends: np.ndarray


def load_raw_data(
    bin_file_dir: str,
    chunk_samples: int = 100000,
    ttl_only: bool = False,
    write_ttl: bool = False,
    verbose: bool = True
) -> RawDataContainer:

    with b2p.PyBinFileReader(bin_file_dir, chunk_samples=chunk_samples) as pbfr:
        n_points = pbfr.length
        n_electrodes = pbfr.num_electrodes
        array_id = pbfr.array_id
        if ttl_only:
            ttl_data = pbfr.get_data_for_electrode(0, 0, n_points)
            electrode_data = None
        else:
            data = pbfr.get_data(0, n_points)
            ttl_data = data[:, 0]
            electrode_data = data[:, 1:]

    epoch_starts = np.array(
        [idx for idx, i in enumerate(np.diff(ttl_data)) if i<0]
    )
    epoch_ends = np.array(
        [idx for idx, i in enumerate(np.diff(ttl_data)) if i>0]
    )

    bin_file_path = Path(bin_file_dir)
    if write_ttl:
        if bin_file_path.is_file():
            bin_folder = bin_file_path.parent
        elif bin_file_path.is_dir():
            bin_folder = bin_file_path
        else:
            print(f'Unable to write TTL data, {bin_file_dir} is neither a file nor folder')
            return RawDataContainer(
                array_id = array_id,
                n_electrodes=n_electrodes,
                n_samples=n_points,
                electrode_data=electrode_data,
                ttl_data=ttl_data.astype(int),
                epoch_starts=epoch_starts.astype(int),
                epoch_ends=epoch_ends.astype(int),
            )

        d_out = {'trace' : ttl_data.astype(int).tolist(),
                 'epochStarts' : epoch_starts.tolist(),
                 'epochEnds' : epoch_ends.tolist(),
                 'array_id': array_id,
                 'n_samples': n_points,
                 }

        filename = bin_folder / f'{bin_folder.stem}_ttl.json'
        with open(filename, 'w') as f:
            json.dump(d_out, f)

        if verbose:
            print(f'TTL data written to {filename}')

    return RawDataContainer(
        array_id = array_id,
        n_electrodes=n_electrodes,
        n_samples=n_points,
        electrode_data=electrode_data,
        ttl_data=ttl_data.astype(int),
        epoch_starts=epoch_starts.astype(int),
        epoch_ends=epoch_ends.astype(int),
    )

