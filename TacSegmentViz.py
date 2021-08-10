#import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, MultiCursor, RadioButtons
from matplotlib.backend_bases import MouseButton
#from functools import partial

__author__  = ['Lorenzo Spagnoli']
__email__ = ['lorenzo.rspagnoli@gmail.com']

'''
   Loading the data in memory has been adapted from Nico Curti Medical_viewer.py visible at https://gist.github.com/Nico-Curti/86678eb42f72a32bad53475ff665c8ff on which 
   i made a contribution to update the point marker identification.
'''

def load_dcm (dirname):
  '''
  Load DicomDir files using SimpleITK OR Pydicom
  functions.

  Parameters
  ----------
    dirname : str
      Path to the directory

  Returns
  -------
    img : array-like
      3D tensor of the image slices
  '''
  try:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(dirname)
    reader.SetFileNames(dicom_names)

    img = reader.Execute()
    img = sitk.GetArrayFromImage(img)

  except ImportError :

    import pydicom
    from glob import glob

    files = glob('{0}/*[0-9]'.format(dirname))

    z = [float(pydicom.dcmread(f)[('0020', '0032')][-1]) for f in files]
    z_order = np.argsort(z)
    files = np.asarray(files)[z_order]
    img = np.asarray([pydicom.dcmread(f).pixel_array for f in files])

  return img

def load_nii (filename):
  '''
  Load Nifti file using SimpleITK OR Nibabel
  functions.

  Parameters
  ----------
    filename : str
      Path to the Nifti filename

  Returns
  -------
    img : array-like
      3D tensor of the image slices
  '''
  try:
    import SimpleITK as sitk

    img = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(img)

  except ImportError:

    import nibabel as nib

    img = nib.load(filename)
    img = img.get_fdata()

  return img


def parse_args ():

  description = '3D DICOM viewer'

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--file',
                      dest='filename',
                      required=True,
                      type=str,
                      action='store',
                      help='Path to the directory/file'
                      )
  parser.add_argument('--dcm',
                      dest='dcm',
                      required=False,
                      action='store_true',
                      help='Specify the usage of the DICOM loader',
                      default=None
                      )
  parser.add_argument('--nii',
                      dest='nii',
                      required=False,
                      action='store_true',
                      help='Specify the usage of the Nifti loader',
                      default=None
                      )
  parser.add_argument('--norm',
                      dest='norm',
                      required=False,
                      action='store_true',
                      help='Perform the normalization of the image in [0, 255] usin NORM_MINMAX',
                      default=True
                      )
  parser.add_argument('--do_thresh',
                      dest='threshold_flag',
                      required=False,
                      action='store_true',
                      help='Choose if the threshold needs to be done, you can choose the value with --threshold',
                      default=False
                      )
  parser.add_argument('--threshold',
                      dest='threshold',
                      type=float,
                      required=False,
                      action='store',
                      help='Choose thresholding value for the image, needs --do_thresh to be passsed as argument',
                      default=0.0
                      )
  parser.add_argument('--mask',
                      dest='mask_filename',
                      required=True,
                      type=str,
                      action='store',
                      help='Path to the directory/file'
                      )
  parser.add_argument('--m_dcm',
                      dest='mask_dcm',
                      required=False,
                      action='store_true',
                      help='Specify the usage of the DICOM loader',
                      default=None
                      )
  parser.add_argument('--m_nii',
                      dest='mask_nii',
                      required=False,
                      action='store_true',
                      help='Specify the usage of the Nifti loader',
                      default=None
                      )
  parser.add_argument('--m_norm',
                      dest='mask_norm',
                      required=False,
                      action='store_true',
                      help='Perform the normalization of the image in [0, 255] usin NORM_MINMAX',
                      default=True
                      )
  parser.add_argument('--test_click',
                      dest='test_click',
                      required=False,
                      action='store_true',
                      help='Perform the normalization of the image in [0, 255] usin NORM_MINMAX',
                      default=False
                      )

  args = parser.parse_args()

  if not args.dcm and not args.nii:
    print('Please specify one of the possible data types. '
          'Currently supported dtypes are Dicom [--dcm], Nifti [--nii]',
          flush=True, end='\n', file=sys.stderr)
    parser.print_help(sys.stderr)
    exit(0)

  return args

def gamma_trasf(image, gamma):
    """
    Takes the image matrix and gamma-corrects it by elevating the pixel values to the specified power gamma

    Args:
        image (float ndarray): The matrix of the image to gamma-correct
        gamma (float): Power of the gamma transformation

    Returns:
        gamma_modified: The matrix of the image gamma-corrected

    Raises:
        Exception:

        """
    if np.min(image)<0:
        raise ValueError("Passed a matrix with negative values into a gamma correction.The behaviour would be undefined so check the input matrix and renormalize it if necessary")
    gamma_modified = np.power(image,gamma)
    return gamma_modified

def threshold_func(image, threshold_value):
    """
    Takes the image and performs a threshold on threshold_value: any pixel value bigger than threshold_value is saturated to 255, all lower values are set to zero

    Args:
        image (float ndarray): The matrix of the image
        threshold_value (float): The value around which the thresholding operation is performed

    Returns:
        thresh (float ndarray): The thresholded matrix

    Raises:
        Exception: description

    """
    thresh = np.zeros_like(image)
    thresh[image<threshold_value]=0
    thresh[image>threshold_value]=255
    return thresh

def cut_negative(image):
    """
    Takes the image and performs a threshold on zero: any pixel values bigger than zero are kept the same, all lower values are set to zero

    Args:
        image (float ndarray): The matrix of the image

    Returns:
        thresh (float ndarray): The thresholded matrix

    Raises:
        Exception: description

    """
    thresh = np.zeros_like(image)
    thresh[image<0]=0
    thresh[image>=0]=image[image>=0]
    return thresh

def find_first_nonzero_element(image):
    """
    Given the image returns the first nonzero element in the matrix. The utility in this script is to find the starting slices in the visualization tool.
    It's used on the segmentation masks for the disease which have boolean logic so this amounts to seeing the first occurence of the segmentation in the image
    stack

    Args:
        image (float ndarray): The matrix of the image

    Returns:
        first_nonzero (tuple of int): The indices og the first non-null value in the image

    Raises:
        Exception: description

    """
    nonzero = np.nonzero(image) #gives a tuple of len equal to num dimension of the image
    first_nonzero = [nonzero[i][0] for i in range(len(nonzero))]
    return tuple(first_nonzero)

def rotate_indices(image, n_rot, coordinates):
    """
    This function assumes that rotations are of 90° in the counterclockwise direction. Given the shape of the image(matrix), the numer of times it was rotated
    and the indices of a certain element in the original matrix it returns the indices of that same element but in the rotated matrix.

    This is used to coordinate the drawing of the circles in the visualization part of the code. Works recursively and is also used to define it's inverse called
    inverse_rotation. Since it's recursive the number of times it's called can be reduced by noticing that 90° rotations repeat themselves every four instances
    so n_rot can be considered in modulus 4

    WARNING: this assumes (row, col) indices shape, effectively this means that when giving coordinates on a plane they should be in (y,x) shape and not (x,y)

    Would it be better to just pass the shape instead of the whole image????? maybe a bit more agile
    image_shape (tuple of ints): Shape of the matrix before any rotation, i.e. shape of the original data matrix

    Args:
        image (numpy.ndarray): Matrix before any rotation
        n_rot (int): Number of rotations the matrix underwent
        coordinates (tuple of ints) : (row, col) coordinates of the element considered
                                            ATTENTION: if thinking in (x,y), coordinates should be passed as (y,x)

    Returns:
        new_coord (tuple of ints): The (row, col) indices in the rotated matrix of the element considered in the original matrix

    Raises:
        ValueError: Since this function works only with two indices at a time and assumes rotation around the centre of a flat, 2D matrix this error is
                    thrown any time a higher dimensional array is passed
    """
    if len(image.shape)>2:
        raise ValueError("The passed matrix has more than 2D. This function works only on flat matrices i.e. images")
    image_shape = image.shape
    n_rot = n_rot%4
    num_rows,num_cols = image_shape
    row,col = coordinates
    if n_rot >1:
        n_rot -=1
        new_coord = rotate_indices(image, 1, coordinates)
        return  rotate_indices(np.rot90(image), n_rot, new_coord)
    elif n_rot==1:
        return (num_cols-col-1,row)
    elif n_rot==0:
        return coordinates

def inverse_rotation(image, n_rot, coordinates):
    """
    This function assumes that rotations are of 90° in the counterclockwise direction. Given the rotated image(matrix) shape, the numer of times it was rotated
    and the indices of a certain element inside it this function returns the indices of that same element but in the original matrix.

    This is used to coordinate the drawing of the circles in the visualization part of the code. Works recursively and is defined as "inverse" of rotate_indices
    since this operation amounts to finding the indices after a rotation in the clockwise direction. Instead of defining this rotation simply note that 3(=m)
    counterclockwise rotations of 90° correspond to a single(3-m) clockwise rotation

    WARNING: this assumes (row, col) indices shape, effectively this means that when giving coordinates on a plane they should be in (y,x) shape and not (x,y)
    image_shape (tuple of ints): Shape of the matrix rotated image, i.e. shape of the rotated data matrix
    Args:
        image (np.ndarray) : Rotated image or matrix
        n_rot (int): Number of rotations the matrix needs to undergo
        coordinates (tuple of ints) : (row, col) coordinates of the element considered
                                            ATTENTION: if thinking in (x,y), coordinates should be passed as (y,x)

    Returns:
        new_coord (tuple of ints): The (row, col) indices in the original matrix of the element considered in the rotated matrix

    Raises:
        ValueError: Since the function of which this is the inverse works only with two indices at a time and assumes rotation around the centre of a flat, 2D matrix
                    this error is thrown any time a higher dimensional array is passed
    """
    n_rot = n_rot%4
    return rotate_indices(image, 4-n_rot, coordinates)


def main():
    try :
        import seaborn as sns
        sns.set()
    except ImportError:
        print('Problems importing seaborn, check if installed. Will use matplotlib')
    '''parse arguments, and reorder data to have it homogeneous'''
    args = parse_args()

    if args.dcm is not None:
      image = load_dcm(args.filename)
    elif args.nii is not None:
      image = load_nii(args.filename)
    else:
      raise ValueError('Could not find the correct data loader for the image')
    if args.mask_dcm is not None:
      mask = load_dcm(args.mask_filename)
    elif args.mask_nii is not None:
      mask = load_nii(args.mask_filename)
    else:
      raise ValueError('Could not find the correct data loader for the mask')

    if args.norm:
      image = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16S)
      image = image.astype('uint8')
    if args.mask_norm:
      mask = cv2.normalize(mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16S)
      mask = mask.astype('uint8')

    dims = np.argsort(image.shape)
    dims_m = np.argsort(mask.shape)

    image = image.transpose(dims)
    mask = mask.transpose(dims_m)

    if args.threshold_flag:
        image = threshold_func(image, args.threshold)

    '''create figure and axes to plot the images and visualize midpoint of the stack'''
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    axial = plt.subplot(gs[0,:])
    sagittal = plt.subplot(gs[1,0])
    coronal = plt.subplot(gs[1,1])
    axial.set_title('Axial')
    axial.axis('off')
    sagittal.set_title('Sagittal')
    sagittal.axis('off')
    coronal.set_title('Coronal')
    coronal.axis('off')

    init_axial = find_first_nonzero_element(mask)[0] if not find_first_nonzero_element(mask)[0]<5 else image.shape[0]//2
    init_sagittal = find_first_nonzero_element(mask)[2] if not find_first_nonzero_element(mask)[0]<5 else image.shape[2]//2
    init_coronal = find_first_nonzero_element(mask)[1] if not find_first_nonzero_element(mask)[0]<5 else image.shape[1]//2

    init_alpha = 0.33
    gamma_init = 1

    '''plot image and mask superimposed with variable opacity. Assign these artists to attributes of main so they are accessible everywhere'''
    main.axial_mask = axial.imshow(mask[init_axial,:,:], cmap='magma')
    main.axial_im = axial.imshow(image[init_axial,...], cmap='gray', alpha = init_alpha)

    main.sagittal_mask = sagittal.imshow(mask[...,init_sagittal], cmap='magma', aspect='auto')
    main.sagittal_im = sagittal.imshow(image[...,init_sagittal], cmap='gray', alpha = init_alpha, aspect='auto' )

    main.coronal_mask = coronal.imshow(mask[:,init_coronal,...], cmap='magma', aspect='auto')
    main.coronal_im = coronal.imshow(image[:,init_coronal,...], cmap='gray', alpha = init_alpha, aspect='auto')

    '''create and locate button on the right of the slider'''
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    axcolor = 'lightgoldenrodyellow'
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        '''
        When the reset button is pressed the figure returns to the original condition. All sliders are reset, all figures and artists are erased and only the
        main images are redrawn
        '''
        saxial.reset()
        ssagittal.reset()
        scoronal.reset()
        salpha.reset()
        sgamma.reset()
        radio.set_active(0)
        axial.cla()
        sagittal.cla()
        coronal.cla()

        main.axial_mask = axial.imshow(mask[init_axial,:,:], cmap='magma')
        main.axial_im = axial.imshow(image[init_axial,:,:], cmap='gray', alpha = init_alpha)

        main.sagittal_mask = sagittal.imshow(mask[:,:,init_sagittal], cmap='magma', aspect='auto')
        main.sagittal_im = sagittal.imshow(image[:,:,init_sagittal], cmap='gray', alpha = init_alpha, aspect='auto')

        main.coronal_mask = coronal.imshow(mask[:,init_coronal,:], cmap='magma', aspect='auto')
        main.coronal_im = coronal.imshow(image[:,init_coronal,:], cmap='gray', alpha = init_alpha, aspect='auto')

        axial.axis('off')
        coronal.axis('off')
        sagittal.axis('off')
        axial.set_title('Axial')
        sagittal.set_title('Sagittal')
        coronal.set_title('Coronal')

    button.on_clicked(reset)

    '''create locations for the sliders below figures or on the side of the figure. Instanciate sliders as s-(controlled-quantity) for slider-quantity'''
    axial_ax = plt.axes([0.25, 0.01, 0.45, 0.02], facecolor=axcolor)
    sagittal_ax = plt.axes([0.25, 0.04, 0.45, 0.02], facecolor=axcolor)
    coronal_ax = plt.axes([0.25, 0.07, 0.45, 0.02], facecolor=axcolor)
    alpha_ax = plt.axes([0.01, 0.1, 0.02, 0.8], facecolor=axcolor)
    gamma_ax = plt.axes([0.03, 0.1, 0.02, 0.8], facecolor=axcolor)

    saxial = Slider(axial_ax, 'Axial', 0, image.shape[0]-1, valinit=init_axial, valstep=1)
    ssagittal = Slider(sagittal_ax, 'Sagittal', 0, image.shape[1]-1, valinit=init_sagittal, valstep=1)
    scoronal = Slider(coronal_ax, 'Coronal', 0, image.shape[2]-1, valinit=init_coronal, valstep=1)
    sgamma = Slider(gamma_ax, r'$\gamma$', 0.5, 2, valinit=gamma_init, orientation='vertical')
    salpha = Slider(alpha_ax, r'$\alpha$', 0, 1, valinit=init_alpha, orientation='vertical')

    '''locate radio'''
    rax = plt.axes([0.85, 0.80, 0.10, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('default', r'$90^{\circ}$', r'$180^{\circ}$', r'$270^{\circ}$'))

    def rotate_func(label):
        '''
        The RadioButtonsare used to control rotation of the image, given the box shape we reasonably only care about discrete 90° rotation but it can be easily
        generalized to a more continuous slider case
        '''
        z_index = saxial.val
        x_index = ssagittal.val
        y_index = scoronal.val
        sgamma.reset()
        if label==r'$90^{\circ}$':
            n=1
        elif label==r'$180^{\circ}$':
            n=2
        elif label==r'$270^{\circ}$':
            n=3
        else:
            n=0

        axial.cla()
        coronal.cla()
        sagittal.cla()

        '''plot image and mask superimposed with opacity selected via slider'''
        main.axial_mask = axial.imshow(np.rot90(mask[z_index,:,:],n), cmap='magma')
        main.axial_im = axial.imshow(np.rot90(image[z_index,...],n), cmap='gray', alpha = salpha.val)

        main.sagittal_mask = sagittal.imshow(np.rot90(mask[...,x_index],n), cmap='magma' )
        main.sagittal_im = sagittal.imshow(np.rot90(image[...,x_index],n), cmap='gray', alpha = salpha.val)

        main.coronal_mask = coronal.imshow(np.rot90(mask[:,y_index,:],n), cmap='magma')
        main.coronal_im = coronal.imshow(np.rot90(image[:,y_index,:],n), cmap='gray', alpha = salpha.val)

        axial.set_title('Axial')
        axial.axis('off')
        sagittal.set_title('Sagittal')
        sagittal.axis('off')
        coronal.set_title('Coronal')
        coronal.axis('off')
        sagittal.set_aspect('auto')
        coronal.set_aspect('auto')
        fig.canvas.draw_idle()

    radio.on_clicked(rotate_func)

    def update(val):
        '''
        Call this function any time the sliders are changed and update the whole figure. Gamma slider controls the gamma transformation, alpha controls opacity and
        the other three (i.e. axial, coronal, sagittal) control the visualized slice
        '''
        alpha = salpha.val
        z_index = saxial.val
        x_index = ssagittal.val
        y_index = scoronal.val
        gamma = sgamma.val
        rotation = radio.value_selected

        if rotation=='default':n=0
        elif rotation==r'$90^{\circ}$':n=1
        elif rotation==r'$180^{\circ}$':n=2
        else:n=3

        image1 = gamma_trasf(image, gamma)

        main.axial_mask.set_data(np.rot90(mask[z_index,:,:],n))           #xyplane
        main.axial_im.set_data(np.rot90(image1[z_index,:,:],n))

        main.coronal_mask.set_data(np.rot90(mask[:,y_index,:],n))           #yzplane
        main.coronal_im.set_data(np.rot90(image1[:,y_index,:],n))

        main.sagittal_mask.set_data(np.rot90(mask[:,:,x_index],n))           #xzplane
        main.sagittal_im.set_data(np.rot90(image1[:,:,x_index],n))

        main.coronal_im.set_alpha(alpha)
        main.axial_im.set_alpha(alpha)
        main.sagittal_im.set_alpha(alpha)

        fig.canvas.draw_idle()

    '''When sliders are changed call the update function'''
    saxial.on_changed(update)
    ssagittal.on_changed(update)
    scoronal.on_changed(update)
    salpha.on_changed(update)
    sgamma.on_changed(update)

    def on_click(event):
        '''
        Catches the mouse event and does something, the spliting of the axes events is done so that the datapoint can be individuated more easily.
        Left mouse button draws a circle on all the axes by getting the data coordinates and synchronizes the three views to see coordinated slices.
        '''
        rotation = radio.value_selected
        if rotation=='default':n=0
        elif rotation==r'$90^{\circ}$':n=1
        elif rotation==r'$180^{\circ}$':n=2
        else:n=3
        if event.button is MouseButton.LEFT:
            if event.inaxes in set([axial,coronal,sagittal]):
                if event.inaxes == axial:
                    z_index = saxial.val            #xyplane
                    col = event.xdata               #switching to (row,col) logic and back to be compatible with the inverse_rotation function
                    row = event.ydata
                    (y_index, x_index) = inverse_rotation(image[z_index,...], n, (round(row),round(col)))
                    ssagittal.set_val(x_index)       # Note that the slider controls the axis outside the plane we are seeing which so we use the remaining
                    scoronal.set_val(y_index)        # letter from xyz excluding those of the plane

                elif event.inaxes == coronal:
                    col = event.xdata               #switching to (row,col) logic and back to be compatible with the inverse_rotation function
                    row = event.ydata
                    y_index = scoronal.val          #xzplane
                    (z_index, x_index) = inverse_rotation(image[:,y_index, :], n, (round(row),round(col)))
                    ssagittal.set_val(round(event.xdata))       # Note that the slider controls the axis outside the plane we are seeing which so we use the remaining
                    saxial.set_val(round(event.ydata))        # letter from xyz excluding those of the plane

                elif event.inaxes == sagittal:
                    x_index = ssagittal.val         #yzplane
                    col = event.xdata               #switching to (row,col) logic and back to be compatible with the inverse_rotation function
                    row = event.ydata
                    (z_index, y_index) = inverse_rotation(image[...,x_index], n, (round(row),round(col)))
                    scoronal.set_val(round(y_index))       # Note that the slider controls the axis outside the plane we are seeing which so we use the remaining
                    saxial.set_val(round(z_index))        # letter from xyz excluding those of the plane

                #Draw coordinated circles on the point onto which the user clicked, to determine where to put the circle transform use the inverse of the inverse_rotation
                #i.e. rotate_indices
                z_draw, y_draw = rotate_indices(image[...,x_index], n, (z_index,y_index))
                sag_cir = plt.Circle((y_draw, z_draw),7,color='green', alpha = 0.5)
                sagittal.add_artist(sag_cir)

                z_draw, x_draw = rotate_indices(image[:,y_index,:], n, (z_index,x_index))
                cor_cir = plt.Circle((x_draw, z_draw),7,color='green', alpha = 0.5)
                coronal.add_artist(cor_cir)

                y_draw, x_draw = rotate_indices(image[z_index,...], n, (y_index,x_index))
                ax_cir = plt.Circle((x_draw, y_draw),7,color='green', alpha = 0.5)
                axial.add_artist(ax_cir)

                #data_x, data_y = event.inaxes.transData.inverted().transform((event.x, event.y)) is equivalent to event.xdata event.ydata
                #print the coordinates on screen
                print("Data coord = (%d,%d,%d) of value %d" %(round(x_index), round(y_index), round(z_index), image[round(z_index), round(y_index), round(x_index)]))
                if args.test_click:
                    mask1 = np.random.rand(image.shape[0],image.shape[1],image.shape[2])
                    mask1[round(z_index-2):round(z_index+2), round(y_index-100):round(y_index+100), round(x_index-100):round(x_index+100)]=255
                    print("mask, mask1 values = (%d,%d)" %(mask[round(z_index),round(y_index),round(x_index)],mask1[round(z_index), round(y_index), round(x_index)]))
                    sagittal_mask.set_data(mask1[:,:,round(x_index)])
                    coronal_mask.set_data(mask1[:,round(y_index),:])
                    axial_mask.set_data(mask1[round(z_index),:,:])
            else:
                pass

            fig.canvas.draw_idle()

    plt.connect('button_press_event', on_click)

    plt.show()
    return image

if __name__ == '__main__':
    image = main ()
    '''
    # create the histogram
    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    histogram, bin_edges = np.histogram(image, bins='auto')

    # configure and draw the histogram figure
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    '''
