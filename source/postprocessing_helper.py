import os
import numpy
import matplotlib.image as mpimg
import re
from PIL import Image
import constants
from helpers import load_image
from preprocessing_helper import create_windows, img_crop


def extract_data(filename, num_images, window_size):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/constants.IMG_PATCH_SIZE)*(IMG_HEIGHT/constants.IMG_PATCH_SIZE)

    img_patches = [create_windows(imgs[i], window_size) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


       
#Assign a label to a patch v
def value_to_class(v):
    df = numpy.sum(v)
    if df > constants.FOREGROUND_THRESHOLD:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5: # FIXME make something cleaner?
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * constants.PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*constants.PIXEL_DEPTH
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Get prediction for given input image 
def get_prediction(img, model, window_size):
    data = numpy.asarray(create_windows(img, window_size))
    output_prediction = model.predict(data)
    img_prediction = label_to_img(img.shape[0], img.shape[1], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE, output_prediction)
    return img_prediction


# Get a concatenation of the prediction and image for given input file
def get_prediction_with_mask(img, model, window_size):
    img_prediction = get_prediction(img, model, window_size)
    cimg = concatenate_images(img, img_prediction)
    return cimg


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(img, model, window_size):
    img_prediction = get_prediction(img,model, window_size)
    oimg = make_img_overlay(img, img_prediction)
    return oimg


# assign a label to a patch
def patch_to_label(patch):
    df = numpy.mean(patch)
    if df > constants.FOREGROUND_THRESHOLD:
        return 0
    else:
        return 1

    
def mask_to_submission_strings(img_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", img_filename).group(0))
    im = mpimg.imread(img_filename)
    patch_size = 16
    for j in range(0, im.shape[1], constants.IMG_PATCH_SIZE):
        for i in range(0, im.shape[0], constants.IMG_PATCH_SIZE):
            patch = im[i:i + constants.IMG_PATCH_SIZE, j:j + constants.IMG_PATCH_SIZE]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))      
            

#load test image test them and return submission     
def createSubmission(model, window_size):
    submission_filename = 'submission.csv'
    image_filenames = []
    prediction_test_dir = "predictions_test/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)
    pred_filenames = []
    for i in range(1, constants.TEST_SIZE+1):
        image_filename = '../dataset/test_set_images/test_' + str(i) +"/test_"+ str(i) +".png"
        image_filenames.append(image_filename)
    test_imgs = [load_image(image_filenames[i]) for i in range(constants.TEST_SIZE)]
    for i in range(constants.TEST_SIZE):
        pimg = get_prediction(test_imgs[i],model,window_size)
        #save prediction next to the image
        cimg = concatenate_images(test_imgs[i], pimg)
        Image.fromarray(cimg).save(prediction_test_dir + "prediction_mask_" + str(i) + ".png")
        w = pimg.shape[0]
        h = pimg.shape[1]
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(pimg)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        pred_filename = prediction_test_dir + "prediction_" + str(i+1) + ".png"
        Image.fromarray(gt_img_3c).save(pred_filename)
        pred_filenames.append(pred_filename)
    masks_to_submission(submission_filename, *pred_filenames)
    
    

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*255
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

#Method to take care of the values that are 253 or 254 on the groundtruth images 
def round(x):
    if(x < 0.5):
        return 0.
    else:
        return 1.

    
#create image with the errors of the prediction highlighted
def checkImageTrainSet(model,imgs,gt_imgs,window_size):
    num_images = len(imgs)
    dir_error = 'error_training_set/'
    if not os.path.isdir(dir_error):
        os.mkdir(dir_error)
    for k in range(1, constants.TRAINING_SIZE+1):
        pimg = get_prediction(imgs[k-1],model,window_size)
        w=pimg.shape[0]
        h=pimg.shape[1]
        gt_img = numpy.vectorize(round)(gt_imgs[k-1])
        
        gt_patches = img_crop(gt_imgs[k-1], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE)
        gt_labels = numpy.asarray([value_to_class(numpy.mean(gt_patches[i])) for i in range(len(gt_patches))])
        
        patches = img_crop(pimg, constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE)
        labels = numpy.asarray([value_to_class(numpy.mean(patches[i])) for i in range(len(patches))])
       
        error_label = numpy.zeros((labels.shape[0],2))
        for j in range (labels.shape[0]):
            if(labels[j][0] == gt_labels[j][0]):
                error_label[j] = [1, 0]
            else:
                error_label[j] = [0, 1]
        
        error = label_to_img(w,h,constants.IMG_PATCH_SIZE,constants.IMG_PATCH_SIZE,error_label)
        color_mask = numpy.zeros((w,h,3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(error)  
        color_mask[:,:,0] = gt_img8
        
        img8 = img_float_to_uint8(imgs[k-1])
        background = Image.fromarray(img8, 'RGB').convert("RGBA")
        overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
        new_img = Image.blend(background, overlay, 0.5)
        new_img.save(dir_error + "error_%.3d" % k + ".png") 