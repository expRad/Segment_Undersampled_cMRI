import pydicom as dicom
import numpy as np
import nibabel as nib
from glob import glob
from pathlib import Path
from tqdm import tqdm
import re
import os

# inspired by https://stackoverflow.com/a/42320319
def get_largest_mode(elements):
    counts = {k:elements.count(k) for k in set(elements)}
    modes = sorted(dict(filter(lambda x: x[1] == max(counts.values()), counts.items())).keys())
    return modes[-1]

studies = glob("kaggle_download/*/*/*/study")

def get_slices_for_study(study):
    images = glob(study+"/sax_*/*.dcm")
    total_images = len(images)
    slices = [x for x in images if re.search(r'IM-\d+-0001.*\.dcm$', x)]
    sliceinfo = dict()
    dims = list()
    phases = list()
    for s in slices:
        dcm = dicom.dcmread(s)
        seriesNumber = int(dcm.SeriesNumber)
        sliceLocation = float(dcm.SliceLocation)
        dim = (dcm.Rows, dcm.Columns)
        cardNumIm = int(dcm.CardiacNumberOfImages)
        sliceinfo[s] = dict(dim = dim, location = sliceLocation, series=seriesNumber, phases=cardNumIm)
        dims.append(dim)
        phases.append(cardNumIm)
    dim_mode = get_largest_mode(dims)
    phases_mode = get_largest_mode(phases)
    for s in slices:
        if(sliceinfo[s]["dim"] != dim_mode):
            print("Slice removed\t{}\tdim\tmode:{} this:{}".format(Path(s),dim_mode,sliceinfo[s]["dim"]))
            del sliceinfo[s]
        elif(sliceinfo[s]["phases"] != phases_mode):
            print("Slice removed\t{}\tphases\tmode:{} this:{}".format(Path(s),phases_mode,sliceinfo[s]["phases"]))
            del sliceinfo[s]
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    prevSlice = sorted_slices[0]
    prevSliceLoc = sliceinfo[prevSlice]['location']
    for s in sorted_slices[1:]:
        sliceLoc = sliceinfo[s]['location']
        if sliceLoc-prevSliceLoc < 1:
            toRemove = s
            if sliceinfo[prevSlice]['series'] == sliceinfo[s]['series']:
                if prevSlice < s:
                    toRemove = prevSlice
            else:
                if sliceinfo[prevSlice]['series'] < sliceinfo[s]['series']:
                    toRemove = prevSlice
            del sliceinfo[toRemove]
            if s == toRemove:
                print("Slice removed\t{}\tcollission\t{} ({},{})".format(Path(toRemove),Path(prevSlice),sliceLoc,prevSliceLoc))
                continue
            else:
                print("Slice removed\t{}\tcollission\t{} ({},{})".format(Path(toRemove),Path(s),prevSliceLoc,sliceLoc))
        prevSlice = s
        prevSliceLoc = sliceLoc
    sorted_slices = sorted(list(sliceinfo.keys()), key=lambda x: sliceinfo[x]['location'])
    d = dicom.dcmread(sorted_slices[0])
    if hasattr(d, 'SpacingBetweenSlices'):
        expectedGap = float(d.SpacingBetweenSlices)
    else:
        expectedGap = float(dicom.dcmread(sorted_slices[1]).SliceLocation)-float(d.SliceLocation)
    prevSliceLoc = sliceinfo[sorted_slices[0]]['location']
    for s in sorted_slices[1:]:
        sliceLoc = sliceinfo[s]['location']
        actualGap = sliceLoc-prevSliceLoc
        if(abs(actualGap-expectedGap)>.1):
            print("Warning\t{}\tinconsistent spacing\texpected:{} actual:{}".format(Path(s),expectedGap,actualGap))
        prevSliceLoc = sliceLoc
    return sorted_slices


# adjusted from https://github.com/baiwenjia/ukbb_cardiac/blob/master/data/biobank_utils.py by Wenjia Bai (under Apache-2 license)
# changes:
#  - use previously determined sorted slices rather than subdirs
def convert_dicom_stack_to_nii(sorted_slices, used_files_log=None, path="."):
    """ Read dicom images and store them in a 3D-t volume. """
    # Number of slices
    Z = len(sorted_slices)
    #for name, dir in sorted(self.subdir.items()):
    
    # Read a dicom file at the first slice to get the temporal information
    d = dicom.read_file(sorted_slices[0])
    X = d.Columns
    Y = d.Rows
    T = d.CardiacNumberOfImages
    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])
    patientID = d.PatientID

    # DICOM coordinate (LPS)
    #  x: left
    #  y: posterior
    #  z: superior
    # Nifti coordinate (RAS)
    #  x: right
    #  y: anterior
    #  z: superior
    # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
    # Refer to
    # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

    # The coordinate of the upper-left voxel of the first and second slices
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]

    if Z >= 2:
        # Read a dicom file at the second slice
        d2 = dicom.read_file(sorted_slices[1])
        pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
        pos_ul2[:2] = -pos_ul2[:2]
        axis_z = pos_ul2 - pos_ul
        axis_z = axis_z / np.linalg.norm(axis_z)
    else:
        axis_z = np.cross(axis_x, axis_y)

    # Determine the z spacing
    if hasattr(d, 'SpacingBetweenSlices'):
        dz = float(d.SpacingBetweenSlices)
    elif Z >= 2:
        print('Warning: can not find attribute SpacingBetweenSlices. '
              'Calculate from two successive slices.')
        dz = float(np.linalg.norm(pos_ul2 - pos_ul))
    else:
        print('Warning: can not find attribute SpacingBetweenSlices. '
              'Use attribute SliceThickness instead.')
        dz = float(d.SliceThickness)

    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3, 0] = axis_x * dx
    affine[:3, 1] = axis_y * dy
    affine[:3, 2] = axis_z * dz
    affine[:3, 3] = pos_ul

    # The 4D volume
    volume = np.zeros((X, Y, Z, T), dtype='float32')

    # Go through each slice
    for z in range(0, Z):
        # In a few cases, there are two or three time sequences or series within each folder.
        # We need to find which seires to convert.
        files = glob(sorted_slices[z].replace("0001","*",1))

        # Now for this series, sort the files according to the trigger time.
        files_time = []
        for f in files:
            d = dicom.read_file(f)
            t = d.TriggerTime
            files_time += [[f, t]]
        files_time = sorted(files_time, key=lambda x: x[1])

        # Read the images
        for t in range(0, T):
            # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
            # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
            # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
            # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
            # with nibabel's dimension.
            try:
                f = files_time[t][0]
                d = dicom.read_file(f)
                volume[:, :, z, t] = d.pixel_array.transpose()
            except IndexError:
                print('Warning: dicom file missing for {0}: time point {1}. '
                      'Image will be copied from the previous time point.'.format(f, t))
                volume[:, :, z, t] = volume[:, :, z, t - 1]
            except (ValueError, TypeError):
                print('Warning: failed to read pixel_array from file {0}. '
                      'Image will be copied from the previous time point.'.format(f))
                volume[:, :, z, t] = volume[:, :, z, t - 1]
            except NotImplementedError:
                print('Warning: failed to read pixel_array from file {0}. '
                      'pydicom cannot handle compressed dicom files. '
                      'Switch to SimpleITK instead.'.format(f))
                reader = sitk.ImageFileReader()
                reader.SetFileName(f)
                img = sitk.GetArrayFromImage(reader.Execute())
                volume[:, :, z, t] = np.transpose(img[0], (1, 0))
            if used_files_log is not None:
                used_files_log.write(str(Path(f))+"\n")

    # Temporal spacing
    dt = (files_time[1][1] - files_time[0][1]) * 1e-3

    # Store the image
    os.makedirs(path+"/"+patientID)
    nim = nib.Nifti1Image(volume, affine)
    nim.header['pixdim'][4] = dt
    nim.header['sform_code'] = 1
    nib.save(nim, path+"/"+patientID+"/sa.nii.gz")

os.makedirs('python_conversion', exist_ok=True)

f = open("python_conversion/used_dicoms.log", "w")

for study in tqdm(studies):
    tmpslices = get_slices_for_study(study)
    convert_dicom_stack_to_nii(tmpslices, used_files_log=f, path="python_conversion")

f.close()
