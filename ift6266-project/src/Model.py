'''
Created on Feb 12, 2017

@author: francis
'''

import numpy as np
import theano
import theano.tensor as T
import numpy.random as rng
import DataAccess
import pylab
import PIL.Image as Image

input = T.ftensor4()

w = theano.shared(rng.normal(size=(3, 3, 3, 3)).astype('float32'))

conv = T.nnet.conv2d(input, w)
relu = T.nnet.relu(conv)
sigmoid = T.nnet.sigmoid(conv)

f = theano.function([input], sigmoid*256)

inputs,_ = DataAccess.getTrainingData(0, 2)

inputs = np.transpose(inputs, (0,3,1,2)).reshape(2, 3, 64, 64)

filtered_img = f(inputs)

print filtered_img.max()

filtered_img=filtered_img.transpose(0,2,3,1).reshape(2,62,62,3)

filtered_img = filtered_img.astype(np.uint8)

Image.fromarray(filtered_img[1]).show()

'''
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[1, 0, :, :])
pylab.show()
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[1, 1, :, :])
pylab.show()
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[1, 2, :, :])
pylab.show()'''
