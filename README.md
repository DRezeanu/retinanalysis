# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Installation
```
cd to root directory
activate your conda environment of choice (if using)
python -m pip install -e . (for dev use)
python -m pip install . (to lock the modules in their current state)
```

## Sample config.ini file (must be in same src folder as settings.py):
```
[DEFAULT]
analysis = /Volumes/data-1/analysis
data = /Volumes/data-1/data/sorted
h5 = /Volumes/data-1/data/h5
meta = /Volumes/data-1/datajoint_testbed/mea/meta
tags = /Volumes/data-1/datajoint_testbed/mea/tags
user = drezeanu

[MAC_SSD]
analysis = /Volumes/ExternalM2/mea_ssd/analysis
data = /Volumes/ExternalM2/mea_ssd/data/sorted
h5 = /Volumes/ExternalM2/mea_ssd/data/h5
meta = /Volumes/ExternalM2/mea_ssd/datajoint_testbed/mea/meta
tags = /Volumes/ExternalM2/mea_ssd/datajoint_testbed/mea/tags
user = drezeanu

[WINDOWS_NAS]
analysis = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/Z:/analysis
data = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/Z:/data/sorted
h5 = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/Z:/data/h5
meta = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/Z:/datajoint_testbed/mea/meta
tags = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/Z:/datajoint_testbed/mea/tags
user = drezeanu

[WINDOWS_SSD]
analysis = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/E:/mea_ssd/analysis
data = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/E:/mea_ssd/data/sorted
h5 = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/E:/mea_ssd/data/h5
meta = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/E:/mea_ssd/datajoint_testbed/mea/meta
tags = /Users/drezeanu/UW/Core Repositories/RetinAnalysis/E:/mea_ssd/datajoint_testbed/mea/tags
user = drezeanu
```
