# -*- coding: future_fstrings -*-

from omegaconf import OmegaConf
import sys
import os
import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List,Optional
from omegaconf import MISSING
from astropy.io import fits
from pk_common_functions.functions import get_model_DHI,copy_disk,\
    isiterable,tirific_template,load_tirific,update_disk_angles
from scipy.interpolate import interp1d
#from multiprocessing import cpu_count

from typing import List, Optional
#import support_functions as sf
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse

@dataclass
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


def create_bar(Template, disk=1, length=None, thickness = 0.1, \
        disk_length =None, bar_angle= [37.],double_sides = True,\
        bar_velocities = [None], bar_brightness = [None] ):
    #disk_length= the size of the visible disk (WarpStart in AGC) in arcsec, 
    # set to last radius if unset
    '''Routine to create a Bar into a model'''
    Radii = np.array([x for x in Template['RADI'].split()]\
            ,dtype=float)
    
    if disk_length is None: disk_length = Radii[-1]
    if disk == 1:
        ext = ''
    else:
        ext = f'_{disk}'
    if length is None:  
        velocity = np.array([x for x in Template[f'VROT{ext}'].split()]\
            ,dtype=float)
       
        max_rad = Radii[-1]
        #The pattern speed at a given radius is vrot/radii
        V_Rot = interp1d(Radii, velocity, fill_value="extrapolate")
        # The radius of co-ration can be approximated by the extend of the visible disk ~ Warpstart (Roberts et al. 1975)
        Omega_CR = V_Rot(disk_length)/disk_length
        # From this we can estimate the inner and outer Lindblad resonances (Eq 38 Dobbs & Baba 2012)
        #The epicyclic frequency k^2=R*d/drOmega^2+4*Omega^2
        # f'(x) = (f(x+h)-f(x-h))/2h
        h = disk_length/1000.
        derive = (V_Rot(float(disk_length+h))**2/(disk_length+h)**2
                - V_Rot(float(disk_length-h))**2/(disk_length-h)**2)/(2*h)
        k_CR = (disk_length * derive+4*Omega_CR**2)**0.5
        # So the ILR =
        LLR = Omega_CR-k_CR/2.
        ULR = Omega_CR+k_CR/2.
        Radii[0] = 0.1
        om = interp1d(Radii, velocity/Radii, fill_value="extrapolate")
        Radii[0] = 0.
        r_cur = Radii[1]
        while om(r_cur) > ULR and r_cur < max_rad:
            r_cur += 0.1
        length = 0.75*r_cur #This is the inner lindblad resonance estimate
       
    bar_width = thickness*length  # the bar thickness is 10% of the n
    
    # We set the full brightness to the maximum of the disk if not given
    if bar_brightness[0] is None:
        bar_brightness = np.zeros(len(Radii))
        bar_brightness[np.where(Radii < length)[0]] = np.max([float(x) \
            for x in Template[f'SBR{ext}'].split()])
    # The width has to be 180 when R < width and 180*width/(pi*r)
    width = np.zeros(len(Radii))
    width[Radii <= bar_width] = 180.
    # the angle made up of the radius and width *2.
    width[Radii > bar_width] = 360./np.pi * \
        np.arcsin(bar_width/Radii[Radii > bar_width])
    # Get the number of disks present
    ndisk = int(Template["NDISKS"])
    ndisk += 1
    #print("We are adding disk no {:d}".format(ndisk))
    # we also need streaming motions
      

    if bar_velocities[0] is None:
        bar_velocities = np.full(len(Radii),-25)
   
    if not isiterable(bar_velocities):
        bar_velocities = [bar_velocities]
    #copy the input disk
    Template_New = copy_disk(Template, disk)
    #Template_New = copy.deepcopy(Template)


    #We offset by 37 deg.

    Template_New.insert(f"VSYS_{ndisk:d}", f"AZ1P_{ndisk:d}",
            f"{' '.join([str(e) for e in bar_angle])}")
    Template_New.insert(f"AZ1P_{ndisk:d}", f"AZ1W_{ndisk:d}"
            , f"{' '.join([str(e) for e in width])}")
    if double_sides:
        Template_New.insert(f"AZ1W_{ndisk:d}", f"AZ2P_{ndisk:d}"
            , f"{' '.join([str(x+180.) for x in bar_angle])}")
        Template_New.insert(f"AZ2P_{ndisk:d}", f"AZ2W_{ndisk:d}"
            ,f"{' '.join([str(e) for e in width])}")
    # And we add streaming motions to the bar km/s
    Template_New.insert(f"VROT_{ndisk:d}", f"VRAD_{ndisk:d}"
        , f"{' '.join([str(e) for e in bar_velocities])}")
    Template_New[f"SBR_{ndisk:d}"] = f"{' '.join(str(e) for e in bar_brightness)}"
   

    return Template_New


create_bar.__doc__ = f'''
NAME:
   create_bar

PURPOSE:
    Create a bar

CATEGORY:
   agc

INPUTS:
    velocity,Radii,disk_brightness,Template, disk=1,WarpStart=-1
    velocity = Rotation curve
    Radii= the radii
    disk_brightness = the SBR profile of the disk
    disk=1,WarpStart=-1,Bar="No_Bar"

OPTIONAL INPUTS:
    disk = 1
    Are we creating in the approaching (1) or receding side (2)

    WarpStart=-1
    radius of the start of the warp

    Bar="No_Bar"
    Boolean whether bar is included or not

OUTPUTS:
    phase = the phases of the arms
    brightness = the brightness amplitude of the arms
    width =the used width in deg

OPTIONAL OUTPUTS:

PROCEDURES CALLED:
   Unspecified

NOTE:
'''

import subprocess
def main():
        argv = sys.argv[1:]
        cfg_defaults = OmegaConf.structured(defaults)
        # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
        inputconf = OmegaConf.from_cli(argv)
        cfg = OmegaConf.merge(cfg_defaults,inputconf)
        if cfg.input_def is None:
            raise InputError(f'''There is no default for the input file please provide one by calling:
add_bar_to_model input_def=<input tirific def file>''') 

        if cfg.output_def is None:
            base = os.path.splitext(cfg.input_def)[0]
            cfg.output_def =f'{base}_bar.def'
        # Read the input_file
        print(f'Reading {cfg.input_def}')
        Template = tirific_template(f'{cfg.input_def}')
        if cfg.disk_length is None:
            cfg.disk_length = get_model_DHI(f'{cfg.input_def}')/2.

        base_files =  os.path.splitext(Template['OUTSET'])[0]
        Template['OUTSET'] = f'{base_files}_bar.fits'
        Template['TIRDEF']= f'{base_files}_bar.def'
        

       
        Template_New = create_bar(Template,disk=cfg.input_disk,\
            length = cfg.bar_length,disk_length=cfg.disk_length,\
            thickness = cfg.bar_thickness,bar_angle=cfg.bar_angle,\
            bar_velocities = cfg.bar_velocities,\
            bar_brightness=cfg.bar_brightness,\
            double_sides=cfg.double_sides) 
        
        print(f'Writing {cfg.output_def}')
        with open(f'{cfg.output_def}', 'w') as file:
            for key in Template_New:
                if key[0:5] == 'EMPTY':
                    file.write('\n')
                else:
                    file.write(f"{key}= {Template_New[key]} \n")

      


if __name__ == '__main__':
    #with VizTracer(output_file="FAT_Run_Viztracer.json",min_duration=1000) as tracer:
    main(sys.argv[1:])
