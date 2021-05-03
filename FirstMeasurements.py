import matplotlib.pyplot as plt
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
import radiomics
import logging
import json
import warnings

from pathlib import Path
from matplotlib.widgets import Slider
from radiomics import featureextractor
from pydicom.data import get_testdata_file

"""
def Visualize_image(imageName, maskName, slice_index=None):
    '''
    Creates a Figure with the image and superimoposes the mask from the segmentation. The plot only shows the slice chosen with slice_number

        Parameters
        imageName (str) :   Path of the image to be visualized
        maskName (str) : Path of the mask to superimpose onto the image
        slice_number (int): Number of the slice that needs to be seen. if None is given chooses the midpoint of the stack

        Raises:
        ValueError: if slice_number is not contained within the given image

        Returns
        None

    '''
    test_image = nib.load(imageName).get_data()
    test_mask = nib.load(maskName).get_data()

    if slice_index != None:
        if (slice_index > test_image.shape[2] or slice_index > test_mask.shape[2]):
            raise ValueError("The chosen slice cannot be visualized because it's out of range, try with a smaller number")
    else:
        slice_index = test_image.shape[2]//2

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(np.rot90(test_image[:,:,slice_index]), cmap='gray')
    ax1.set_title('Image')
    ax2.imshow(np.rot90(test_image[:,:,slice_index]), cmap='gray')
    ax2.imshow(np.rot90(test_mask[:,:,slice_index]), cmap='Reds', alpha = 0.33)
    ax2.set_title('Image + Mask')
    plt.show()

def update(val):
    idx = slidx.val
    l.set_data(A[idx])
    fig.canvas.draw_idle()

"""

def Quantifier(image_path, mask_path, Log_filename='TestLog.txt', print_all_features = False):
    '''
    Computes the chosen features of the segmentated regions(in maskName) as referred to the original image(imageName). The features must be provided in
    a list of string format, if empty list is given all possible features are retured. Also returns a logger file in Log_filename which contains
    info on the computations of the features

    This function uses PyRadiomics as dependece so for in depth information go to https://pyradiomics.readthedocs.io/en/latest/features.html

        Parameters
        image_path (str) :   Path of the image to be visualized
        mask_path (str) : Path of the mask to superimpose onto the image
        features (list of str) : names of the features that need to be computed

        Returns
        featureDict (dict): Dictionary of the computed features. Data can be obtained with features[featureName] where featureName is a string
                            containing the name of the wanted feature

    '''

    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(Log_filename, mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline

    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Disable all classes except firstorder
    # extractor.disableAllFeatures()

    # Enable all features in firstorder
    # extractor.enableFeatureClassByName('firstorder')

    # Only enable mean and skewness in firstorder
    #extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    #extractor.enableFeaturesByName(shape=['Compactness1'])

    print("Calculating features")
    featureDict = extractor.execute(image_path, mask_path)

    if print_all_features:
        for featureName in featureDict.keys():
            print("Computed %s: %s" % (featureName, featureDict[featureName]))

    return featureDict

def save_to_json(filename, feature_dict):
    '''
    Takes the data contained in the feature_dict and saves it into filename.json

            Parameters
            filename (str) :   Name of the file on which to save, extension excluded
            feature_dict (dict) : Dictionary of the features to save

            Raises
            WARNING: if the filename parameter contains an extension beacuse this will be overwritten and turned into '.json'

            Returns


    '''
    #To write into json it's necessary that the dictionary does not contain any numpy arrays, for this reason convert all instances to lists/floats
    final_dict = {}
    for key in feature_dict.keys():
        if isinstance(feature_dict[key],np.ndarray):
            final_dict[key] = feature_dict[key].tolist()
        else:
            final_dict[key] = feature_dict[key]
    if filename.find('.')!=-1:

        warnings.warn("The given filename has an extension, check the final result")

        filename = filename[0:filename.find('.')] + '.json'
    else:
        filename = filename+'.json'

    path = Path(filename)
    with path.open("a") as file:
        json.dump(final_dict, file)
        file.write('\n')
    #maybe consider returning a value if successful



'''
Masks are from 0255 to 0304 in format study_0255_mask.nii.gz
Actual scans are in format study_0255.nii.gz
'''

Dataset_path = "./MosMed_Labeled_Dataset/COVID19_1110"
masks = Dataset_path + "/masks"
studies = Dataset_path + "/studies"

imageName = studies+"/CT-1/study_0255.nii.gz"
maskName = masks+"/study_0255_mask.nii.gz"

Relevant_featureNames = ["glcm_ClusterShade", "glcm_ClusterProminance", "glcm_IDM", "glcm_JointEnergy", "glcm_JointEntropy", ]

#Visualize_image(imageName = imageName , maskName=maskName)
#feat_dict = Quantifier(image_path=imageName, mask_path=maskName, Log_filename='TestLog.txt', print_all_features = False)

#save_to_json('FileDIProva.txt', feat_dict)
