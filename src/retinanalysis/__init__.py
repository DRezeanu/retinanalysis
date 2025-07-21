# Must be the first import, otherwise the database won't load properly
import retinanalysis.config.schema as schema

# Import various data and analysis directories directly.
# Settings doesn't reference any of the utils or classes so it should be
# safe to import first without circular import issues
from . import config
from .config import settings
from .config.settings import (NAS_ANALYSIS_DIR,
                              NAS_DATA_DIR,
                              H5_DIR,
                              META_DIR,
                              TAGS_DIR,
                              USER)

# Utilities imported first. They should NEVER reference the classes for anything
# other than type hints, which should be done using the TYPE_CHECKING and 
# __future__ annotations imports (see vision_utils). This avoids problems with
# circular imports
from . import utils
from .utils import database_pop
from .utils.database_pop import *

from .utils import database_utils
from .utils.database_utils import *

from .utils import datajoint_utils
from .utils.datajoint_utils import *

from .utils import ei_utils
from .utils.ei_utils import *

from .utils import regen
from .utils.regen import *

from .utils import vision_utils
from .utils.vision_utils import *

# Import classes last
from . import classes
from .classes import analysis_chunk
from .classes.analysis_chunk import AnalysisChunk

from .classes import stim
from .classes.stim import (StimBlock,
                           MEAStimBlock,
                           MEAStimGroup,
                           make_mea_stim_group,
                           D_REGEN_FXNS)

from .classes import response
from .classes.response import (ResponseBlock,
                               MEAResponseBlock,
                               SCResponseBlock)

from .classes import qc
from .classes.qc import MEAQC


# Pipeline must be imported last as it references the above pieces.
from .classes import mea_pipeline
from .classes.mea_pipeline import (MEAPipeline,
                                   create_mea_pipeline)

from .classes import dedup
from .classes.dedup import DedupBlock





