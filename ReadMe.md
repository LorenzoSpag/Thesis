# Thesis project

I'll keep in this file the todos and references just to be sure not to lose anything. Then when I'll have more information this will become also the description of the whole project.

## References
  1. MosMed dataset, used to get comfortable with datatype and first measurements. Downloadable from [Lung segmentation challenge](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#https://wiki.cancerimagingarchive.net/display/DOI/Thoracic+Volume+and+Pleural+Effusion+Segmentations+in+Diseased+Lungs+for+Benchmarking+Chest+CT+Processing+Pipelines#7c5a8c0c0cef44e488b824bd7de60428) following the [MosMed dataset link](https://mosmed.ai/en/datasets/covid19_1110/). This is related to the [article](https://doi.org/10.1101/2020.05.20.20100362)
  2. NSCLC dataset, contains all DICOM images and was used as the MosMed dataset. This however does not contain the segmentations related to the images. The files are Downloadable from [NSCLC Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) using the [NBIA Data retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
  3. Haralick features and related [article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212110)
  4. Signed distance function [Definition on wikipedia](https://en.wikipedia.org/wiki/Signed_distance_function)
  5. [Radiomics features "how to" guide and critical reflection](https://insightsimaging.springeropen.com/articles/10.1186/s13244-020-00887-2)
  6. Libraries used: [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/index.html) , [Pydicom](https://pydicom.github.io/pydicom/stable/tutorials/installation.html), [ITK](https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html#),


### Description of the scripts
  * FirstMeasurements.py contains a first, very basic approach to visualization of images and segmented pieces as well as a basic approach to measuring quantities from the segmented parts. It also contains a function to save the dictionaries resulting from the segmentation analysis into a json file. The functions contained are:
    1. Quantifier(image_path, mask_path, Log_filename , print_all_features): Takes image and segmentation(mask) and computes all features
    2. save_to_json(filename, feature_dict) : Converts all np.arrays in the dictionary of features into lists and saves the whole structure to a json file(filename.json)
  * Some measurements are:
    1. Looking at the definition of energy in the Haralick features papers it can be seen that the corresponding measure in Pyradiomics is JointEnergy computed from glcm. JointEnergy gives an idea of how often how often valuePairs are neighbouring eachother, the higher the energy the more homogeneous is the image
    2. Ditto for entropy definition, JointEntropy in PyRadiomics seems equivalent to the Haralick definition of entropy via the GrayLevelCooccurrenceMatrix. JointEntropy is a measure of randomness and variability in the intensity values within the neighobouring
    3. Inertia seems to be non readymade in the PyRadiomics module, further research is needed

    4. ClusterShade is a measure of skewness and asymmetry of the GLCM, the higher the value the more asymmetric it is
    5. ClusterProminance is a measure of skewness and uniformity of GLCM, the higher the value the more asymmetric the GLCM the lower the values the more likely it is to find a peak about the mean of the GLCM with lower variability
    6. Inverse Difference Momentum(IDM) quantifies local homogeneity of the image
    7. Numerosity(By which we mean the number of segmented lesions)
    8. Volume(i.e. volume of the segmented region)
    9. Right/left balance log of ratio between the damaged percentage of left lung over right lung, definition is still a work in progress
    10. Volume percentage

  * TacSegmentViz.py is a visualization tool to superimpose the segmentation of a clinical image onto the image itself. It creates a window that shows the axial, coronal and sagittal views of the images given in input and allows control over what slice is visualized via the use of three sliders situated below the images. There is also the possibility to change the transparency of the image in the [0,1] range where zero makes image completely transparent(i.e. see only the mask) and 1 makes the image completely opaque(i.e. see only the image without the mask), default value is 0.33. There also is the possibility to gamma-correct the images by using a slider in the [0.1-2] range(the range can be easily changed). It can catch mouse clicks on screen and when it does draws circles in all views on the same data point and synchronizes the view of the slices to see said point. Finally there's the option to rotate the images to aid in visualization. It's run via command window and it takes as arguments.
    1. --file the path to the image to be visualized
    2. --nii/--dcm to specify which extension the file in input has
    3. --norm to normalize the image into uint8
    4. --mask the path to the mask to be superimposed
    5. --m_nii / --m_dcm format of the mask
    6. --m_norm to normalize the mask into uint8
    7. --do_thresh Flags the thresholder as active and perform a threshold on 0 if nothing else is specified
    8. --threshold is to choose the value onto which we are thresholding
    9. --test_click was used in debugging to check if the test catcher event was behaving as desired
