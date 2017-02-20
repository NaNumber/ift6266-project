'''
Created on Feb 11, 2017

@author: francis
'''

import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image

def getTrainingData(batch_idx, batch_size,
                  # ## PATH need to be fixed
                  mscoco="../data/inpainting/", split="train2014", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    batch_input = []
    batch_target = []

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # ## Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]
            
        Image.fromarray(input).show()

        batch_input.append(input)
        batch_target.append(target)

    return np.array(batch_input,dtype='float32') / 256, np.array(batch_target,dtype='float32') / 256

def splitChannel(img):
    return np.array([[y[0] for y in x] for x in img]), np.array([[y[1] for y in x] for x in img]), np.array([[y[2] for y in x] for x in img])

if __name__ == '__main__':
    # resize_mscoco()
    input, target = getTrainingData(5, 10)
    
    red, green, blue = splitChannel(input[0])
    
    i = 1
