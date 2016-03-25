import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import convolve2d
import cv2
import sklearn
def ldp(image):
	x,y = image[:,:,0].shape
	#image = image[:,:,2]	
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
	ldp_image = np.zeros((x,y,8),dtype='int')
	mask1 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]],dtype=int);
	mask2 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]],dtype=int);
	for i in xrange(4):
		m1 = mask1;
		m2 = mask2;
		for j in xrange(i):
			m1 = np.rot90(m1);
			m2 = np.rot90(m2);
		ldp_image[:,:,2*i] = convolve2d(image,m1,mode='same');
		ldp_image[:,:,2*i+1] = convolve2d(image,m2,mode='same');
	ldp_hist = np.zeros(image.shape,dtype='uint8');
	max_image = np.zeros(image.shape,dtype='uint8')
	for i in xrange(x):	
		for j in xrange(y):
			a = ldp_image[i,j,:];
			a = abs(a);
			b = np.sort(a);
			thresh = b[4];
			a = a -thresh;
			a[a>0] = 1;
			a[a<=0] = 0;
			c = np.packbits(a,axis=None);
			ldp_hist[i,j] = int(c);
			max_image[i,j] = b[0];
	return(ldp_image,ldp_hist,max_image);
	
