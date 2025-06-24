"""
Parsing cell typing .txt files
Vyom Raval
Thursday, Mar 2, 2023
"""

import numpy as np


class CellTypes(object):

    def __init__(self, str_txt, ls_RGC_labels=['OffP', 'OffM', 'OnP', 'OnM']):
        self.str_txt = str_txt
        try:
            # self.arr_types = np.loadtxt(str_txt, dtype=str, sep='  ')
            self.arr_types = np.genfromtxt(str_txt, dtype=str, delimiter='  ')
        except Exception as e:
            print('Error reading file: {}'.format(e))
            self.arr_types = np.array([])
            self.d_types = {}

        # Check that arr_types has 2 columns
        if len(self.arr_types.shape) != 2:
            print(f'Error: 2 space delimiter resulted in {self.arr_types.shape} shape')
            # Try 1 space delimiter
            self.arr_types = np.genfromtxt(str_txt, dtype=str, delimiter=' ')
            if len(self.arr_types.shape) != 2:
                print(f'Error: 1 space delimiter resulted in {self.arr_types.shape} shape')
                self.arr_types = np.array([])
                self.d_types = {}
                self.d_main_IDs = {}
                return            

        # self.ls_RGC_labels = ['OffP', 'OffM', 'OnP', 'OnM']
        self.ls_RGC_labels = ls_RGC_labels

        # Map cell ID to RGC label
        self.d_types = {}
        for c_idx in range(len(self.arr_types)):
            n_ID = int(self.arr_types[c_idx, 0])
            str_type = self.arr_types[c_idx, 1]

            # Check if RGC label match in str_type
            for str_RGC in self.ls_RGC_labels:
                ls_split = str_type.split('/')
                ls_split = [x.lower() for x in ls_split]
                if str_RGC.lower() in ls_split:
                    str_type = str_RGC

            # If no match, then retain original str_type
            self.d_types[n_ID] = str_type

        # Get list of IDs for main RGC types
        self.d_main_IDs = {}
        n_matches = 0
        for str_RGC in self.ls_RGC_labels:
            arr_type_ids = self.get_ids_of_type(str_RGC)
            # Add to dictionary if there are any IDs
            if len(arr_type_ids) > 0:
                self.d_main_IDs[str_RGC] = arr_type_ids
                n_matches += len(arr_type_ids)
        
        # Check if no type matches found
        
        if n_matches == 0:
            self.no_matches = True
            print(f'No matches found in classification {str_txt}')
        else:
            self.no_matches = False

    def print_summary(self, b_only_main_types=False):
        print('Total number of cells: {}'.format(len(self.d_types)))
        arr_labels, arr_numcells = np.unique(list(self.d_types.values()),
                                             return_counts=True)

        if b_only_main_types:
            for str_RGC in self.ls_RGC_labels:
                # Check if the label is present in classification file
                if str_RGC in arr_labels:
                    # print('Number of {}: {}'.format(
                    #     str_RGC, arr_numcells[arr_labels == str_RGC][0]))
                    n_type_ids = len(self.d_main_IDs[str_RGC])
                    n_all_type_ids = arr_numcells[arr_labels == str_RGC][0]
                    print(f'{str_RGC} ({n_type_ids}/{n_all_type_ids})')
                else:
                    print('Number of {}: {}'.format(str_RGC, 0))
        else:
            for type_idx in range(len(arr_labels)):
                print('Number of {}: {}'.format(arr_labels[type_idx],
                                                arr_numcells[type_idx]))

    def get_ids_of_type(self, str_type):
        ls_cellids = []
        for n_ID in self.d_types.keys():
            if self.d_types[n_ID].lower() == str_type.lower():
                ls_cellids.append(n_ID)
        return np.array(ls_cellids)

def map_ids_to_idx(arr_source_IDs, arr_target_IDs, verbose=False):
    """
    Map cell IDs in arr_source_IDs to indices in arr_target_IDs
    """
    arr_idx = np.zeros(len(arr_source_IDs), dtype=int)
    for idx in range(len(arr_source_IDs)):
        try:
            arr_idx[idx] = np.where(arr_target_IDs == arr_source_IDs[idx])[0][0]
        except:
            if verbose:
                print('Cell ID {} not found in data'.format(arr_source_IDs[idx]))
            arr_idx[idx] = -1 # This will have to be accounted for later
    return arr_idx
