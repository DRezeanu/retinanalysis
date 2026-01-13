if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <EXPERIMENT_DATE> <CHUNK_NAME> <SORT_ALGORITHM> <NUM_CPU>"
    echo "Example: bash $0 20250306C doves_images kilosort2.5 8"
    exit 0
fi


EXP=$1
shift
CHUNK=$1
shift
ALG=$1
shift
NUM_CPU=$1

TEMPORARY_SORT_PATH='/data/data/'
LITKE_PATH='/data/data/' #'/usr/share/pool/data/'
SORTED_SPIKE_PATH='/data/data/sorted'; #"/home/mike/ftp/files/ksort_out/"; #'/home/mike/ftp/files/20220420C/data021/kilosort2/';
KILOSORT_TTL_PATH='/data/data/sorted'; #'/home/mike/ftp/files/ksort_out/';
RAW_DATA_PATH='/data/data/raw';
VISIONPATH='/home/mike/Documents/git_repos/manookin-lab/MEA/src/Vision7_for_2015DAQ/Vision.jar';

TMP_PATH="${SORTED_SPIKE_PATH}/${EXP}/"
data_files=($(head -n 1 "${TMP_PATH}${EXP}_${CHUNK}.txt"))

EXPERIMENT_SPIKE_PATH="${SORTED_SPIKE_PATH}/${EXP}";
CHUNK_SPIKE_PATH="${EXPERIMENT_SPIKE_PATH}/${CHUNK}/${ALG}/"
KILOSORT_TTL_PATH="${KILOSORT_TTL_PATH}/${EXP}/TTLTriggers";
FILE_DATA_PATH="${EXPERIMENT_DATA_PATH}/${data_files[0]}/"

# TODO: script to parse dedup/split decisions

# Kilosort to vision globals and neurons file
python kilosort_to_vision.py $CHUNK_SPIKE_PATH $KILOSORT_TTL_PATH $CHUNK_SPIKE_PATH $ALG -l -d $FILE_DATA_PATH

# EI calculation
java -Xmx8G -cp $VISIONPATH edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" $VISION_OUT $CHUNK_SPIKE_PATH 0.01 67 133 1000000 ${NUM_CPU}