# pk_common_functions
There are many functions in pyFAT and pyHIARD that are useful. However we do not always want to dress programs up with all pyFAT machinery. This is to have a central repository for these functions to keep them up to date. !!! --- These Functions should not be used for code that is intended for distribution.

There is no pypi but downloading the repository and then installing from the directory should allow for a pip install. Documentation is limited to this README

i.e:

pip install <path_to_repository>

should install the module pk_common_functions.

function can then be accessed through:

from pk_common_functions.functions import functions

or

import pk_common_functions.functions


# Script functions
--------------------------------------

*add_bar_to_model*

add a bar to your favorite tirific model

class defaults:

    bar_angle: List = field(default_factory=lambda: [37.]) #Bar angle in degrees
    bar_length: Optional[float] = None #Bar length in arcsec, default is the estimated ILR
    bar_brightness: List = field(default_factory=lambda: [None]) 
    bar_velocities: List = field(default_factory=lambda: [None]) 
    bar_thickness: float = 0.1 #Thickness of bar in fraction of the length  
    input_disk: int = 1
    double_sides: bool = True
    disk_length: Optional[float] = None #length of the optical disk in arcsec, defaults to R_HI
    input_def: Optional[str] = None
    output_def: Optional[str] = None




