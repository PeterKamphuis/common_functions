

import numpy as np
import copy
import os
import re
import warnings
from astropy.wcs import WCS

from astropy.io import fits
from scipy.ndimage import rotate
from scipy.optimize import curve_fit, OptimizeWarning


class InputError(Exception):
    pass
class FittingError(Exception):
    pass


        # a Function to convert the RA and DEC into hour angle (invert = False) and vice versa (default)
def convertRADEC(RAin,DECin,invert=False, colon=False, verbose=False):
    if verbose:
        print(f'''CONVERTRADEC: Starting conversion from the following input.
{'':8s}RA = {RAin}
{'':8s}DEC = {DECin}
''')
    RA = copy.deepcopy(RAin)
    DEC = copy.deepcopy(DECin)
    if not invert:
        try:
            _ = (e for e in RA)
        except TypeError:
            RA= [RA]
            DEC =[DEC]
        for i in range(len(RA)):
            xpos=RA
            ypos=DEC
            xposh=int(np.floor((xpos[i]/360.)*24.))
            xposm=int(np.floor((((xpos[i]/360.)*24.)-xposh)*60.))
            xposs=(((((xpos[i]/360.)*24.)-xposh)*60.)-xposm)*60
            yposh=int(np.floor(np.absolute(ypos[i]*1.)))
            yposm=int(np.floor((((np.absolute(ypos[i]*1.))-yposh)*60.)))
            yposs=(((((np.absolute(ypos[i]*1.))-yposh)*60.)-yposm)*60)
            sign=ypos[i]/np.absolute(ypos[i])
            if colon:
                RA[i]="{}:{}:{:2.2f}".format(xposh,xposm,xposs)
                DEC[i]="{}:{}:{:2.2f}".format(yposh,yposm,yposs)
            else:
                RA[i]="{}h{}m{:2.2f}".format(xposh,xposm,xposs)
                DEC[i]="{}d{}m{:2.2f}".format(yposh,yposm,yposs)
            if sign < 0.: DEC[i]='-'+DEC[i]
        if len(RA) == 1:
            RA = str(RA[0])
            DEC = str(DEC[0])
    else:
        if isinstance(RA,str):
            RA=[RA]
            DEC=[DEC]

        xpos=RA
        ypos=DEC

        for i in range(len(RA)):
            # first we split the numbers out
            tmp = re.split(r"[a-z,:]+",xpos[i])
            RA[i]=(float(tmp[0])+((float(tmp[1])+(float(tmp[2])/60.))/60.))*15.
            tmp = re.split(r"[a-z,:'\"]+",ypos[i])
            if float(tmp[0]) != 0.:
                DEC[i]=float(np.absolute(float(tmp[0]))+((float(tmp[1])+(float(tmp[2])/60.))/60.))*float(tmp[0])/np.absolute(float(tmp[0]))
            else:
                DEC[i] = float(np.absolute(float(tmp[0])) + ((float(tmp[1]) + (float(tmp[2]) / 60.)) / 60.))
                if tmp[0][0] == '-':
                    DEC[i] = float(DEC[i])*-1.
        if len(RA) == 1:
            RA= float(RA[0])
            DEC = float(DEC[0])
        else:
            RA =np.array(RA,dtype=float)
            DEC = np.array(DEC,dtype=float)
    return RA,DEC

convertRADEC.__doc__ =f'''
 NAME:
    convertRADEC

 PURPOSE:
    convert the RA and DEC in degre to a string with the hour angle

 CATEGORY:
    support_functions

 INPUTS:
    Configuration = Standard FAT configuration
    RAin = RA to be converted
    DECin = DEC to be converted

 OPTIONAL INPUTS:


    invert=False
    if true input is hour angle string to be converted to degree

    colon=False
    hour angle separotor is : instead of hms

 OUTPUTS:
    converted RA, DEC as string list (hour angles) or numpy float array (degree)

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

# function for converting kpc to arcsec and vice versa
def convertskyangle(angle, distance=-1., unit='arcsec', \
        distance_unit='Mpc', physical=False, verbose=False):
    if distance == -1.:
        raise InputError(f'convertskyangle needs a distance')
    if verbose:
        print(f'''CONVERTSKYANGLE: Starting conversion from the following input.
    {'':8s}Angle = {angle}
    {'':8s}Distance = {distance}
''')

    try:
        _ = (e for e in angle)
    except TypeError:
        angle = [angle]

        # if physical is true default unit is kpc
    angle = np.array(angle,dtype=float)
    if physical and unit == 'arcsec':
        unit = 'kpc'
    if distance_unit.lower() == 'mpc':
        distance = distance * 10 ** 3
    elif distance_unit.lower() == 'kpc':
        distance = distance
    elif distance_unit.lower() == 'pc':
        distance = distance / (10 ** 3)
    else:
        print(f'''CONVERTSKYANGLE: {distance_unit} is an unknown unit to convertskyangle.
{'':8s}CONVERTSKYANGLE: please use Mpc, kpc or pc.
''')
        raise InputError(f'CONVERTSKYANGLE: {distance_unit} is an unknown unit to convertskyangle.')
    if not physical:
        if unit.lower() == 'arcsec':
            radians = (angle / 3600.) * ((2. * np.pi) / 360.)
        elif unit.lower() == 'arcmin':
            radians = (angle / 60.) * ((2. * np.pi) / 360.)
        elif unit.lower() == 'degree':
            radians = angle * ((2. * np.pi) / 360.)
        else:
            print(f'''CONVERTSKYANGLE: {unit} is an unknown unit to convertskyangle.
{'':8s}CONVERTSKYANGLE: arcsec, arcmin or degree.
''')
            raise InputError(f'CONVERTSKYANGLE: {unit} is an unknown unit to convertskyangle.')


        kpc = 2. * (distance * np.tan(radians / 2.))
    else:
        if unit.lower() == 'kpc':
            kpc = angle
        elif unit.lower() == 'mpc':
            kpc = angle * (10 ** 3)
        elif unit.lower() == 'pc':
            kpc = angle / (10 ** 3)
        else:
            print(f'''CONVERTSKYANGLE: {unit} is an unknown unit to convertskyangle.
{'':8s}CONVERTSKYANGLE: please use Mpc, kpc or pc.
''')
            raise InputError(f'CONVERTSKYANGLE: {unit} is an unknown unit to convertskyangle.')

        radians = 2. * np.arctan(kpc / (2. * distance))
        kpc = (radians * (360. / (2. * np.pi))) * 3600.
    if len(kpc) == 1:
        kpc = float(kpc[0])
    return kpc

convertskyangle.__doc__ =f'''
 NAME:
    convertskyangle

 PURPOSE:
    convert an angle on the sky to a distance in kpc or vice versa

 CATEGORY:
    common_functions

 INPUTS:
    Configuration = Standard FAT configuration
    angle = the angles or lengths to be converted

 OPTIONAL INPUTS:


    distance=1.
    Distance to the galaxy for the conversion

    unit='arcsec'
    Unit of the angle or length options are arcsec (default),arcmin, degree, pc, kpc(default) and Mpc

    distance_unit='Mpc'gauss_parameters = np.array(['NaN','NaN','NaN'],dtype=float)
        gauss_covariance = np.array(['NaN','NaN','NaN'],dtype=float)

    Unit of the distance options are pc, kpc and Mpc

    physical=False
    if true the input is a length converted to an angle

 OUTPUTS:
    converted value or values

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def cutout_cube(filename,sub_cube, outname=None):

    if outname == None:
        outname = f'{os.path.splitext(filename)[0]}_cut.fits'

    Cube = fits.open(filename,uint = False, do_not_scale_image_data=True,ignore_blank = True, output_verify= 'ignore')
    hdr = Cube[0].header

    if hdr['NAXIS'] == 3:
        print(sub_cube[0,0],sub_cube[0,1],sub_cube[1,0],sub_cube[1,1],sub_cube[2,0],sub_cube[2,1])
        data = Cube[0].data[sub_cube[0,0]:sub_cube[0,1],sub_cube[1,0]:sub_cube[1,1],sub_cube[2,0]:sub_cube[2,1]]
        hdr['NAXIS1'] = sub_cube[2,1]-sub_cube[2,0]
        hdr['NAXIS2'] = sub_cube[1,1]-sub_cube[1,0]
        hdr['NAXIS3'] = sub_cube[0,1]-sub_cube[0,0]
        hdr['CRPIX1'] = hdr['CRPIX1'] -sub_cube[2,0]
        hdr['CRPIX2'] = hdr['CRPIX2'] -sub_cube[1,0]
        hdr['CRPIX3'] = hdr['CRPIX3'] -sub_cube[0,0]
        #Only update when cutting the cube

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coordinate_frame = WCS(hdr)
            xlow,ylow,zlow = coordinate_frame.wcs_pix2world(1,1,1., 1.)
            xhigh,yhigh,zhigh = coordinate_frame.wcs_pix2world(hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3'], 1.)
            xlim = np.sort([xlow,xhigh])
            ylim = np.sort([ylow,yhigh])
            zlim =np.sort([zlow,zhigh])/1000.

    elif hdr['NAXIS'] == 2:
        data = Cube[0].data[sub_cube[1,0]:sub_cube[1,1],sub_cube[2,0]:sub_cube[2,1]]
        hdr['NAXIS1'] = sub_cube[2,1]-sub_cube[2,0]
        hdr['NAXIS2'] = sub_cube[1,1]-sub_cube[1,0]
        hdr['CRPIX1'] = hdr['CRPIX1'] -sub_cube[2,0]
        hdr['CRPIX2'] = hdr['CRPIX2'] -sub_cube[1,0]

    Cube.close()
    fits.writeto(outname,data,hdr,overwrite = True)
    return outname

cutout_cube.__doc__ =f'''
 NAME:
    cutout_cube

 PURPOSE:
    Cut filename back to the size of subcube, update the header and write back to disk.

 CATEGORY:
    fits_functions

 INPUTS:
    filename = name of the cube to be cut
    outname = name of the output file
    sub_cube = array that contains the new size as
                [[z_min,z_max],[y_min,y_max], [x_min,x_max]]
                adhering to fits' idiotic way of reading fits files.

 OPTIONAL INPUTS:

 OUTPUTS:
    the cut cube is written to disk.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def fit_gaussian(x,y, covariance = False,errors = None, \
    verbose= False):
    if verbose:
        print(f'''FIT_GAUSSIAN: Starting to fit a Gaussian.
{'':8s}x = {x}
{'':8s}y = {y}
''')
    # Make sure we have numpy arrays
    x= np.array(x,dtype=float)
    y= np.array(y,dtype=float)
    # First get some initial estimates
    est_peak = np.nanmax(y)

    if errors == None:
        errors = np.full(len(y),1.)
        absolute_sigma = False
    else:
        absolute_sigma = True
    peak_location = np.where(y == est_peak)[0]
    if peak_location.size > 1:
        peak_location = peak_location[0]
    est_center = float(x[peak_location])
    est_sigma = np.nansum(y*(x-est_center)**2)/np.nansum(y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        succes = False
        maxfev= int(100*(len(x)))
        increase =False
        while not succes:
            if verbose:
                print(f'''FIT_GAUSSIAN: Starting the curve fit with {maxfev}
''')
            try:
                gauss_parameters, gauss_covariance = curve_fit(gaussian_function, \
                        x, y,p0=[est_peak,est_center,est_sigma],sigma= errors,\
                        absolute_sigma= absolute_sigma,maxfev=maxfev)

                succes = True
            except OptimizeWarning:
                maxfev =  2000*(len(x))
                print(f'maxfev = {maxfev} which should break the loop {1000*(len(x))}')
            except RuntimeError as e:
                split_error = str(e)
                if 'Optimal parameters not found: Number of calls to function has reached maxfev' in \
                    split_error:
                    maxfev += 100*int(len(x))
                    if verbose:
                        print(f'''FIT_GAUSSIAN: We failed to find an optimal fit due to the maximum number of evaluations. increasing maxfev to {maxfev}
''')
                else:
                    print(f'''FIT_GAUSSIAN: failed due to the following error {split_error}
''')
                    raise RuntimeError(split_error)
            if maxfev >  1000*(len(x)):
                gauss_parameters = np.array(['NaN','NaN','NaN'],dtype=float)
                gauss_covariance = np.array(['NaN','NaN','NaN'],dtype=float)
                succes = True
                #print(f'Failed to find a proper fit to the gaussian')
                #raise FittingError("FIT_GAUSSIAN: failed to find decent gaussian parameters")

    #gauss_parameters, gauss_covariance = curve_fit(gaussian_function, x, y,p0=[est_peak,est_center,est_sigma],sigma= errors,absolute_sigma= absolute_sigma)
    if covariance:
        return gauss_parameters, gauss_covariance
    else:
        return gauss_parameters

fit_gaussian.__doc__ =f'''
 NAME:
    fit_gaussian
 PURPOSE:
    Fit a gaussian to a profile, with initial estimates
 CATEGORY:
    supprt_functions

 INPUTS:
    x = x-axis of profile
    y = y-axis of profile
    Configuration = Standard FAT configuration

 OPTIONAL INPUTS:
    covariance = false
    return to covariance matrix of the fit or not



 OUTPUTS:
    gauss_parameters
    the parameters describing the fitted Gaussian

 OPTIONAL OUTPUTS:
    gauss_covariance
    The co-variance matrix of the fit

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def gaussian_function(axis,peak,center,sigma):
    return peak*np.exp(-(axis-center)**2/(2*sigma**2))



def load_tirific(def_input,Variables = None,array = False,\
        ensure_rings = False ,dict=False):
    #Cause python is the dumbest and mutable objects in the FAT_defaults
    # such as lists transfer
    if Variables is None:
        Variables = ['BMIN','BMAJ','BPA','RMS','DISTANCE','NUR','RADI',\
                     'VROT','Z0', 'SBR', 'INCL','PA','XPOS','YPOS','VSYS',\
                     'SDIS','VROT_2',  'Z0_2','SBR_2','INCL_2','PA_2','XPOS_2',\
                     'YPOS_2','VSYS_2','SDIS_2','CONDISP','CFLUX','CFLUX_2']


    # if the input is a string we first load the template
    if isinstance(def_input,str):
        def_input = tirific_template(filename = def_input )

    out = []
    for key in Variables:
        try:
            out.append([float(x) for x  in def_input[key].split()])
        except KeyError:
            out.append([])
        except ValueError:
            out.append([x for x  in def_input[key].split()])

    #Because lists are stupid i.e. sbr[0][0] = SBR[0], sbr[1][0] = SBR_2[0] but  sbr[:][0] = SBR[:] not SBR[0],SBR_2[0] as logic would demand

    if array:
        tmp = out
        #We can ensure that the output has the same number of values as there are rings
        if ensure_rings:
            length=int(def_input['NUR'])
        else:
            #or just take the longest input as the size
            length = max(map(len,out))
        #let's just order this in variable, values such that it unpacks properly into a list of variables
        out = np.zeros((len(Variables),length),dtype=float)
        for i,variable in enumerate(tmp):
            if len(variable) > 0.:
                out[i,0:len(variable)] = variable[0:len(variable)]

    if dict:
        tmp = {}
        for i,var in enumerate(Variables):
            tmp[var] = out[i]
        out = tmp
    elif len(Variables) == 1:
        out= out[0]
    #print(f'''LOAD_TIRIFIC: We extracted the following profiles from the Template.
#{'':8s}Requested Variables = {Variables}
#{'':8s}Extracted = {out}
#''')
    #Beware that lists are stupid i.e. sbr[0][0] = SBR[0], sbr[1][0] = SBR_2[0] but  sbr[:][0] = SBR[:] not SBR[0],SBR_2[0] as logic would demand
    # However if you make a np. array from it make sure that you specify float  or have lists of the same length else you get an array of lists which behave just as dumb

    return out
load_tirific.__doc__ =f'''
 NAME:
    load_tirific

 PURPOSE:
    Load values from variables set in the tirific files

 CATEGORY:
    common_functions

 INPUTS:
    def_input = Path to the tirific def file or a FAT tirific template dictionary

 OPTIONAL INPUTS:
    Variables = ['BMIN','BMAJ','BPA','RMS','DISTANCE','NUR','RADI','VROT',
                 'Z0', 'SBR', 'INCL','PA','XPOS','YPOS','VSYS','SDIS','VROT_2',  'Z0_2','SBR_2',
                 'INCL_2','PA_2','XPOS_2','YPOS_2','VSYS_2','SDIS_2','CONDISP','CFLUX','CFLUX_2']


    array = False
        Specify that the output should be an numpy array with all varables having the same length

    ensure_rings =false
        Specify that the output array should have the length of the NUR parameter in the def file

    dict = False
        Return the output as a dictionary with the variable names as handles
 OUTPUTS:
    outputarray list/array/dictionary with all the values of the parameters requested

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
    This function has the added option of a dictionary compared to pyFAT
'''
def reduce_header_axes(hdr,axes= 3):
    ax = ['CDELT','CTYPE','CUNIT','CRPIX','CRVAL','NAXIS']
    while hdr['NAXIS'] > axes:
        for par in ax:
            try:
                hdr.remove(f"{par}{hdr['NAXIS']}")
            except KeyError:
                pass

        hdr['NAXIS'] -= 1
    return hdr

def reduce_data_axes(data,axes= 3):
    while len(data.shape) > axes:
        data = data[0,:]
    return data

def rotateCube(Cube, angle, pivot,order=1):
    padX= [int(Cube.shape[2] - pivot[0]), int(pivot[0])]
    padY= [int(Cube.shape[1] - pivot[1]), int(pivot[1])]
    imgP= np.pad(Cube, [[0, 0], padY, padX], 'constant')
    #Use nearest neighbour as it is exact enough and doesn't mess up the 0. and is a lot faster
    imgR = rotate(imgP, angle, axes =(2, 1), reshape=False,order=order)
    return imgR[:, padY[0]: -padY[1], padX[0]: -padX[1]]
rotateCube.__doc__=f'''
 NAME:
    rotateCube(Cube, angle, pivot)

 PURPOSE:
    rotate a cube in the image plane

 CATEGORY:
    common_functions

 INPUTS:
    Cube = the cube data array
    angle = the angle to rotate under
    pivot = the point around which to rotate

 OPTIONAL INPUTS:

 OUTPUTS:
    the rotated cube

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def regrid_array(oldarray, Out_Shape):
    oldshape = np.array(oldarray.shape)
    newshape = np.array(Out_Shape, dtype=float)
    ratios = oldshape/newshape
        # calculate new dims
    nslices = [ slice(0,j) for j in list(newshape) ]
    #make a list with new coord
    new_coordinates = np.mgrid[nslices]
    #scale the new coordinates
    for i in range(len(ratios)):
        new_coordinates[i] *= ratios[i]
    #create our regridded array
    newarray = map_coordinates(oldarray, new_coordinates,order=1)
    if any([x != y for x,y in zip(newarray.shape,newshape)]):
        print("Something went wrong when regridding.")
    return newarray
regrid_array.__doc__ =f'''
 NAME:
    regridder
 PURPOSE:
    Regrid an array into a new shape through the ndimage module
 CATEGORY:
    fits_functions

 INPUTS:
    oldarray = the larger array
    newshape = the new shape that is requested

 OPTIONAL INPUTS:

 OUTPUTS:
    newarray = regridded array

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    scipy.ndimage.map_coordinates, np.array, np.mgrid

 NOTE:
'''
def update_disk_angles(Tirific_Template, verbose = False):
    extension = ['','_2']
    for ext in extension:
        PA = np.array(load_tirific(Tirific_Template,Variables = [f'PA{ext}']),dtype=float)
        inc = np.array(load_tirific(Tirific_Template,Variables = [f'INCL{ext}']),dtype=float)
        if verbose:
            print(f'''UPDATE_DISK_ANGLES: obtained  this from the template
{'':8s} inc{ext} = {inc}
{'':8s} PA{ext} = {PA}
''')
        angle_adjust=np.array(np.tan((PA[0]-PA)*np.cos(inc*np.pi/180.)*np.pi/180.)*180./np.pi,dtype = float)
        if ext == '_2':
            angle_adjust[:] +=180.
        if verbose:
            print(f'''UPDATE_DISK_ANGLES: adusting AZ1P{ext} with these angles
{'':8s}{angle_adjust}
''')
        Tirific_Template.insert(f'AZ1W{ext}',f'AZ1P{ext}',f"{' '.join([f'{x:.2f}' for x in angle_adjust])}")
update_disk_angles.__doc__ =f'''
 NAME:
    update_disk_angles

 PURPOSE:
    Update the AZ1W and AZ1P parameters to match the warp

 CATEGORY:
    modify_template

 INPUTS:
    Tirific_Template = Standard FAT Tirific Template, so a proper dictionary

 OPTIONAL INPUTS:


 OUTPUTS:
    Updated template

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def tirific_template(filename = ''):
    if filename == '':
        raise InputError(f'Tirific_Template does not know a default')
    else:
        with open(filename, 'r') as tmp:
            template = tmp.readlines()
    result = {}
    counter = 0
    # Separate the keyword names
    for line in template:
        key = str(line.split('=')[0].strip().upper())
        if key == '':
            result[f'EMPTY{counter}'] = line
            counter += 1
        else:
            result[key] = str(line.split('=')[1].strip())
    return result
tirific_template.__doc__ ='''
 NAME:
    tirific_template

 PURPOSE:
    Read a tirific def file into a dictionary to use as a template.
    The parameter ill be the dictionary key with the values stored in that key

 CATEGORY:
    read_functions

 INPUTS:
    filename = Name of the def file

 OPTIONAL INPUTS:
    filename = ''
    Name of the def file, if unset the def file in Templates is used



 OUTPUTS:
    result = dictionary with the read file

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
      split, strip, open

 NOTE:
'''
