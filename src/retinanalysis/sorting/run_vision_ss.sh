#!/bin/bash

# USAGE: bash online_vision.sh 20221117C data015 1278606431 FastNoise

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <EXPERIMENT_DATE> <FILE_NAME> <SEED> <PROTCOL_ID>"
    echo "Example w/ default protocol: $0 20240926C data015 1278606431"
    echo "Example: $0 20240926C data015 1278606431 504"
    exit 0
fi


DEFAULT_PROTOCOL="FastNoise"

EXPERIMENT_NAME=$1
shift
FILE_NAME=$1
shift
START_SEED=$1
shift
PROTOCOL_ID="${1:-$DEFAULT_PROTOCOL}"

# Get the start time.
START_TIME=$(date +%s)


# Set the paths by the computer.
YASS_SPIKE_PATH='/home/vyomr/Desktop/data/sorted'
RAW_DATA_PATH='/home/vyomr/Desktop/data/raw';
VISIONPATH="/home/vyomr/Desktop/gitrepos/MEA/src/Vision7_for_2015DAQ/";
TMP_PATH="${YASS_SPIKE_PATH}/${EXPERIMENT_NAME}/"

SORT_PATH="${YASS_SPIKE_PATH}/${EXPERIMENT_NAME}/${FILE_NAME}/vision/"
DATA_PATH="${RAW_DATA_PATH}/${EXPERIMENT_NAME}/"


data_string="${DATA_PATH}$FILE_NAME"

if [[ ! -e $TMP_PATH ]]; then
   mkdir -p ${TMP_PATH}
fi

if [[ ! -e $SORT_PATH ]]; then
   mkdir -p ${SORT_PATH}
fi

# Run spike sorting in Vision.
java -Xmx8000m -Xss2m -cp ${VISIONPATH}Vision.jar edu.ucsc.neurobiology.vision.tasks.NeuronIdentification ${DATA_PATH}${FILE_NAME} ${SORT_PATH} -c ${VISIONPATH}config.xml
# -d64
# Get the EI file.
# java -Xmx8G -cp ${VISIONPATH}Vision.jar edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" ${SORT_PATH}/ ${RAW_DATA_PATH}/${EXPERIMENT_NAME}/${FILE_NAME}/ 0.01 20 40 1000000 8
java -Xmx8G -cp ${VISIONPATH}Vision.jar edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" ${SORT_PATH}/ ${RAW_DATA_PATH}/${EXPERIMENT_NAME}/${FILE_NAME}/ 0.01 67 133 1000000 6

END_TIME=$(date +%s)
RUN_TIME=$((END_TIME-START_TIME))
MINUTES_TIME=$((RUN_TIME/60))
SECONDS_TIME=$((RUN_TIME-MINUTES_TIME*60))
echo "Completed running after ${MINUTES_TIME} minutes and ${SECONDS_TIME} seconds."