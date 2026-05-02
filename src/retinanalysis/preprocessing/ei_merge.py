import argparse
import visionloader as vl
from visionwriter import (EIWriter)
from retinanalysis import (DATA_DIR)





if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='ei_merge.py',
                                     usage='',
                                     description='Script for merging EIs from several individual datafiles\
                                     into a single ei for the chunk.')

    # Optional arguments
    parser.add_argument('-s', '--sorted_dir', help='path to directory containing datafile eis (e.g. .../Volumes/data/sorted/)',
                        type=str, default=DATA_DIR)
    parser.add_argument('-o', '--output_dir', help='path to directory containing datafile eis (e.g. .../Volumes/data/sorted/)',
                        type=str, default=DATA_DIR)




