import visionwriter as vw
import bin2py
import numpy as np
import os
import argparse

SPIKE_TIMES_FILENAME = 'spike_times.npy'
SPIKE_IDENTITY_FILENAME = 'spike_clusters.npy'
CLUSTER_QUALITY_FILENAME = 'cluster_KSLabel.tsv'

def get_litke_triggers(bin_path, RW_BLOCKSIZE=2000000, TTL_THRESHOLD=-1000):
    epoch_starts = []
    epoch_ends = []
    with bin2py.PyBinFileReader(bin_path, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
        array_id = pbfr.header.array_id
        n_samples = pbfr.length
        for start_idx in range(0, n_samples, RW_BLOCKSIZE):
            n_samples_to_get = min(RW_BLOCKSIZE, n_samples - start_idx)
            samples = pbfr.get_data_for_electrode(0, start_idx, n_samples_to_get)
            # Find the threshold crossings at the beginning and end of each epoch.
            below_threshold = (samples < TTL_THRESHOLD)
            above_threshold = np.logical_not(below_threshold)
            # Epoch starts.
            above_to_below_threshold = np.logical_and.reduce([
                above_threshold[:-1],
                below_threshold[1:]
            ])
            trigger_indices = np.argwhere(above_to_below_threshold) + start_idx
            epoch_starts.append(trigger_indices[:, 0])
            below_to_above_threshold = np.logical_and.reduce([
                below_threshold[:-1],
                above_threshold[1:]
            ])
            trigger_indices = np.argwhere(below_to_above_threshold) + start_idx
            epoch_ends.append(trigger_indices[:, 0])
    epoch_starts = np.concatenate(epoch_starts, axis=0)
    epoch_ends = np.concatenate(epoch_ends, axis=0)
    return epoch_starts, epoch_ends, array_id, n_samples

def build_cluster_quality_dict(filepath):
    cluster_quality_by_id = {}
    with open(filepath, 'r') as cluster_quality_file:
        cluster_quality_file.readline()

        remaining_lines = cluster_quality_file.readlines()

        for line in remaining_lines:
            data_list = line.strip('\n').split('\t')
            cluster_quality_by_id[int(data_list[0])+1] = data_list[1]

    return cluster_quality_by_id


def extract_ttl_times (raw_data_path, ttl_threshold, n_samples=None):

    ttl_times = []

    with bin2py.PyBinFileReader(raw_data_path, chunk_samples=10000) as pbfr:

        if n_samples is None:
            n_samples = pbfr.length
        ttl_samples = pbfr.get_data_for_electrode(0, 0, n_samples)
        below_threshold = (ttl_samples < -ttl_threshold)

        j = 0
        while j < ttl_samples.shape[0]:
            while j < ttl_samples.shape[0] and not below_threshold[j]:
                j += 1

            # now we've reached true, or the end
            if j < ttl_samples.shape[0] and below_threshold[j]:
                interval_start = j
                while j < ttl_samples.shape[0] and below_threshold[j]:
                    j += 1

                ttl_times.append(interval_start)

        array_id = pbfr.header.array_id

    return np.array(ttl_times).astype(np.int), n_samples, array_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Kilosort output to Vision .neurons format')
    parser.add_argument('kilosort_spike_path', type=str, help='folder containing Kilosort outputs, must already exist')
    parser.add_argument('vision_path', type=str, help='path to write Vision files, must already exist')
    parser.add_argument('vision_dset_name', type=str, help='name of dataset that we want to write, i.e. data000')
    parser.add_argument('-l', '--litke', action='store_true', help='Use if original raw data is Litke raw data')
    parser.add_argument('-k', '--hier', action='store_true', help='Use if original raw data is Hierlemann h5')
    parser.add_argument('-q', '--quality', action='store_true', help='Include only clusters that are tagged good by kilosort')
    parser.add_argument('-d', '--datapath', nargs='?', type=str, help='Path to raw data for computing triggers.')

    args = parser.parse_args()
    print("Kilosort spike path: {0}".format(args.kilosort_spike_path))
    print("Vision output path: {0}".format(args.vision_path))
    print("Vision dataset name: {0}".format(args.vision_dset_name))

    spike_times_filepath = os.path.join(args.kilosort_spike_path, SPIKE_TIMES_FILENAME)
    spike_identity_filepath = os.path.join(args.kilosort_spike_path, SPIKE_IDENTITY_FILENAME)
    cluster_quality_dict_filepath = os.path.join(args.kilosort_spike_path, CLUSTER_QUALITY_FILENAME)

    spike_times_vector = np.load(spike_times_filepath)
    spike_identity_vector = np.load(spike_identity_filepath)
    quality_by_cluster_id = build_cluster_quality_dict(cluster_quality_dict_filepath)

    n_spikes = spike_times_vector.shape[0]

    # deal with TTL and n_samples here, since we need to reach into the raw data to get it
    # also get the electrode map figured out and generate the .globals file here
    if args.datapath == None:
        if np.ndim(spike_times_vector) == 1:
            n_samples = np.max(spike_times_vector)
        else:
            n_samples = np.max(spike_times_vector[:, 0])
        array_id = 504
        ttl_times = np.array([1, ]) # temporary placeholder
    else:
        ttl_times, _, array_id, n_samples = get_litke_triggers(args.datapath, RW_BLOCKSIZE=2000000, TTL_THRESHOLD=-1000)
    print("Array ID detected: {0}".format(array_id))
    print("Number of samples detected: {0}".format(n_samples))
    neuron_time_offset = 0
    # array_id = 3501
    
    if not args.litke and not args.hier:
        assert False, "Unsupported raw data type specified in arguments"
    elif args.litke:

        # If the file exists, we need to remove it.
        if os.path.exists(args.vision_path + args.vision_dset_name + '.globals'):
            os.remove(args.vision_path + args.vision_dset_name + '.globals')
        print("Writing globals file for array id {0}".format(array_id))
        with vw.GlobalsFileWriter(args.vision_path, args.vision_dset_name) as gfw:
            gfw.write_simplified_litke_array_globals_file(array_id & 0xFFF, # FIXME get rid of this after we figure out what happened with 120um
                                                          0,
                                                          0,
                                                          'Kilosort converted',
                                                          '',
                                                          0,
                                                          n_samples)


    print("Writing neurons file")
    spikes_by_cell_id = {}
    for i in range(n_spikes):

        if np.ndim(spike_times_vector) == 1:
            spike_time = spike_times_vector[i]
            spike_id = spike_identity_vector[i] + 1
        else:
            spike_time = spike_times_vector[i,0]
            spike_id = spike_identity_vector[i,0] + 1
        # we add 1 because Vision/MATLAB requires that real cells start at index 1
        # (MATLAB does 1-based indexing)

        if args.quality:
            if quality_by_cluster_id[spike_id] == 'good':
                if spike_id not in spikes_by_cell_id:
                    spikes_by_cell_id[spike_id] = []

                spikes_by_cell_id[spike_id].append(spike_time)
        else:
            if spike_id not in spikes_by_cell_id:
                spikes_by_cell_id[spike_id] = []

            spikes_by_cell_id[spike_id].append(spike_time)

    spikes_by_cell_id_np = {}
    for cell_id, spike_list in spikes_by_cell_id.items():
        spikes_by_cell_id_np[cell_id] = np.array(spike_list) + neuron_time_offset

    print("Found {0} cells".format(len(spikes_by_cell_id_np)))

    # If the file exists, we need to remove it.
    if os.path.exists(args.vision_path + args.vision_dset_name + '.neurons'):
        os.remove(args.vision_path + args.vision_dset_name + '.neurons')
    with vw.NeuronsFileWriter(args.vision_path, args.vision_dset_name) as nfw:
        nfw.write_neuron_file(spikes_by_cell_id_np, ttl_times, n_samples)

    print("Done")
