# DICOM visualizer

## Description of the scripts
  * TacSegmentViz.py is a visualization tool to superimpose the segmentation of a clinical image onto the image itself. It creates a window that shows the axial, coronal and sagittal views of the images given in input and allows control over what slice is visualized via the use of three sliders situated below the images. There is also the possibility to change the transparency of the image in the [0,1] range where zero makes image completely transparent(i.e. see only the mask) and 1 makes the image completely opaque(i.e. see only the image without the mask), default value is 0.33. There also is the possibility to gamma-correct the images by using a slider in the [0.1-2] range(the range can be easily changed by editing the script). It can catch mouse clicks on screen and when it does draws circles in all views on the same data point and synchronizes the view of the slices to see said point. Finally there's the option to rotate the images to aid in visualization. It takes as arguments:

    1. --file the path to the image to be visualized
    2. --nii/--dcm to specify which extension the file in input has
    3. --norm to normalize the image into uint8
    4. --mask the path to the mask to be superimposed
    5. --m_nii / --m_dcm format of the mask
    6. --m_norm to normalize the mask into uint8
    7. --do_thresh Flags the thresholder as active and perform a threshold on 0 if nothing else is specified
    8. --threshold is to choose the value onto which we are thresholding
    9. --test_click was used in debugging to check if the test catcher event was behaving as desired

  It t can be called via I-python shell, supposing that the script is in a folder which also contains the data from [MosMed dataset link](https://mosmed.ai/en/datasets/covid19_1110/) in a folder called "MosMed_Labeled_Dataset", using:
  ```
  %run TacSegmentViz.py --file=./MosMed_Labeled_Dataset/COVID19_1110/studies/CT-1/study_0264.nii.gz --nii --mask=./MosMed_Labeled_Dataset/COVID19_1110/masks/study_0264_mask.nii.gz --m_nii --do_thresh --threshold=500
  ```
  * test_TacViz.py is the script used to test the functions used in TacSegmentViz.py to find the coordinates needed to draw the points as they get clicked on. The main properties it tests are the following:
    1. Test if the function called 'inverse_rotation' is truly the inverse of the 'rotate_indices' function by checking if the element in the matrix is truly the same
    2. Test that any multiple of 4 and 0 rotations lead to the same output coordinates equal to input coordinates(because we are supposing 90°rotations)
    3. test that the function 'rotate_indices' is the inverse of the 'inverse_rotation' function
    4. Test that any reasonable shape and element coordinates leads to all positive value_selected
    5. Test that any reasonable shape and element produce coordinates which are still valid indices within the rotated matrix


## References
  1. MosMed dataset, used to get comfortable with datatype and first measurements. Downloadable from [Lung segmentation challenge](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark#https://wiki.cancerimagingarchive.net/display/DOI/Thoracic+Volume+and+Pleural+Effusion+Segmentations+in+Diseased+Lungs+for+Benchmarking+Chest+CT+Processing+Pipelines#7c5a8c0c0cef44e488b824bd7de60428) following the [MosMed dataset link](https://mosmed.ai/en/datasets/covid19_1110/). This is related to the [article](https://doi.org/10.1101/2020.05.20.20100362)
  2. NSCLC dataset, contains all DICOM images and was used as the MosMed dataset. This however does not contain the segmentations related to the images. The files are Downloadable from [NSCLC Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) using the [NBIA Data retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
  3. Haralick features and related [article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212110)
  4. Signed distance function [Definition on wikipedia](https://en.wikipedia.org/wiki/Signed_distance_function)
  5. [Radiomics features "how to" guide and critical reflection](https://insightsimaging.springeropen.com/articles/10.1186/s13244-020-00887-2)
  6. Libraries used: [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/index.html) , [Pydicom](https://pydicom.github.io/pydicom/stable/tutorials/installation.html), [ITK](https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html#),
