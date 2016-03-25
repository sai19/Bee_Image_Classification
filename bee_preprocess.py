from __future__ import division
import cv2
import numpy as np
from  scipy.ndimage import morphology
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from sklearn import svm
import ldp
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Returns the subimage from the overall image
# similar to crop but tilted cropping is also allowed
def subimage(image, centre, theta, width, height):
   	#output_image = np.zeros((height,width,3), np.uint8)
   	#mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
         #              [np.sin(theta), np.cos(theta), centre[1]]])
	mapping = cv2.getRotationMatrix2D(centre,theta,1.0)
   #map_matrix_cv = cv2.fromarray(mapping)
   #cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
	image = cv2.warpAffine(image,mapping,image[:,:,0].shape,flags=cv2.INTER_LINEAR)
	output_image = cv2.getRectSubPix(image,(height,width),centre)   
	return output_image

boundaries = [
	([219,182, 182], [255,255,255]),
	([0,154, 48], [64,255,140]),
]

# returns the probable region where the bee might be present
def detect_bee(image):
	mask = np.ones(image[:,:,0].shape,dtype='uint8');
	mask = 255*mask;
	if (len(image.shape)!=3):
		print "Please input the proper image, make sure the image is coded in RGB space,not BGR and HSV";
		output = None;
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);			
		gray = cv2.Laplacian(gray,cv2.CV_8U)
		ret1, gray = cv2.threshold (gray,np.mean(gray), 255, cv2.THRESH_BINARY)
		#gray = cv2.medianBlur(gray,5);
		#gray = morphology.binary_dilation(gray,iterations=9);
		#gray = gray.astype('uint8');
		mask = cv2.bitwise_and(mask,gray);
		output = cv2.bitwise_and(image, image, mask=mask)
	return (output,mask);

# returns the histogram array in order R,G,B
# if the returned values are zero then make sure that 
# you have imported division
def color_hist(image_array):
	# split the matrix into three sub parts
	red = image_array[:,0];
	green = image_array[:,1];
	blue =  image_array[:,2];
	(hist_red,_) = np.histogram(red,bins=256);
	hist_red = hist_red.astype("float")
	hist_red /= hist_red.max()
	(hist_green,_) = np.histogram(green,bins=256);
	hist_green = hist_green.astype("float")
	hist_green /= hist_green.max()	
	(hist_blue,_) = np.histogram(blue,bins=256);
	hist_blue = hist_blue.astype("float")
	hist_blue /= hist_blue.max()
	hist = np.concatenate((hist_red,hist_green,hist_blue),axis=1);	
	return (hist);
#  color histogram
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist


#  visualization of color histogram
def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar


def delete_other_colors(image):
	final = np.zeros(image.shape,dtype = "uint8")
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
 	
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		final = cv2.bitwise_or(output,final);
	final = cv2.bitwise_not(final);
	final = cv2.bitwise_and(image,final);
	return (final);

def color_gradient(image):
	(x,y,z) = image.shape;
	image_grad = np.zeros((x,y),dtype = 'float64');
	laplacian = cv2.Laplacian(image,cv2.CV_64F);
	laplacian = abs(laplacian);
	image_grad = laplacian[:,:,0]+laplacian[:,:,1]+laplacian[:,:,2];
	image_grad = (image_grad-np.min(image_grad))*(255/(np.max(image_grad)-np.min(image_grad)));
	grad = image_grad.astype('uint8');
	ret1, grad = cv2.threshold (grad,np.mean(grad), 255, cv2.THRESH_BINARY)	
	grad = morphology.binary_fill_holes(grad);
	grad = grad.astype('uint8');
	#grad = cv2.medianBlur(grad,9);
	#_,contours, hierarchy = cv2.findContours(grad,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
	#area = 0;	
	#for cnt in contours:
	#	if(cv2.contourArea(cnt)>=area):
	#		area = cv2.contourArea(cnt);
	#		large = cnt;
	#cv2.drawContours(mask,[large],0,255,-1);
	return(grad); 

def blobs(image):
	detector = cv2.SimpleBlobDetector_create()
 
	# Detect blobs.
	keypoints = detector.detect(image)
	print keypoints;
 
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 	return(im_with_keypoints);

def largest_std(image,original):
		im = cv2.bitwise_and(original,original,mask=image);
		mask_not = cv2.bitwise_not(image);
		_,contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
		ret1, mask_not = cv2.threshold (mask_not,np.mean(mask_not), 255, cv2.THRESH_BINARY);
		im = cv2.bitwise_and(original,original,mask=mask_not);		
		im = im.reshape((im.shape[0]*im.shape[1]),im.shape[2]);
		im = im[np.any(im!=0,axis=1)];
		hist_back = color_hist(im);
		match = 0;			
		if(len(contours)>1):		
			for cnt in contours:
					mask = np.zeros(original[:,:,0].shape,dtype='uint8');	
					cv2.drawContours(mask, [cnt], 0,255, 3);
					im = cv2.bitwise_and(original,original,mask=mask);
					im = im.reshape((im.shape[0]*im.shape[1]),im.shape[2]);
					im = im[np.any(im!=0,axis=1)];
					hist = color_hist(im);
					dist = np.linalg.norm(hist_back-hist);
					dist = 1/dist;
					if(match<dist):
						match=dist;
						large = cnt;
						#print match;
						#print "\n";
		else:
			large = contours[0];
		rect = cv2.minAreaRect(large);		
		center,(w,h),t = rect;
		if(t<-45):
			t = t + 90;
		t = np.pi*t/180.0
		mask = np.zeros(original[:,:,0].shape,dtype='uint8');			
		mask1 = np.zeros(original[:,:,0].shape,dtype='uint8');			
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		patch = subimage(original,(int(center[0]),int(center[1])),t,int(h),int(w));
		patch = cv2.transpose(patch);		
		cv2.drawContours(mask,[box],0,255,-1);
		cv2.drawContours(mask1,[large],0,255,-1);
		output = cv2.bitwise_and(original,original,mask=mask);
		output1 = cv2.bitwise_and(original,original,mask=mask1);
		x,y,w,h = cv2.boundingRect(large);
		output1 = output1[y:y+h,x:x+w];
		return(output1,output,patch);

def return_crop(image):
	laplacian = cv2.Laplacian(image,cv2.CV_64F);
	upper =  0.8*np.max(abs(laplacian));
	lower =  np.average(abs(laplacian));
	edges = cv2.Canny(image,lower,upper);
	edges = morphology.binary_closing(edges,iterations=10);
	edges = morphology.binary_fill_holes(edges);
	edges = edges.astype('uint8');
	mask = copy.copy(edges);	
	#mask = color_gradient(image);
	#mask = morphology.binary_opening(mask);
	#mask = morphology.binary_closing(mask,iterations=10);
	#mask = morphology.binary_fill_holes(mask);
	#mask = mask.astype('uint8');	
	#edges = remove_small(edges);
	#im,mask = remove_abundant(image,mask);
	#mask = cv2.bitwise_and(mask,edges);	
	#im,mask = remove_abundant(image,mask);
	#mask = remove_unwanted(mask,image);
	im = cv2.bitwise_and(image,image,mask=mask);	
	#im,mask = remove_abundant(image,mask);		
	#im,_,_= largest_std(mask,image);		
	return(im,mask,edges);
def remove_small(image):
	_,contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);	
	mask = np.zeros(image.shape,np.uint8)		
	for cnt in contours:
		if(cv2.contourArea(cnt)>=200):
			area = cv2.contourArea(cnt);
			cv2.drawContours(mask,[cnt],0,255,-1);
	return(mask); 


def remove_abundant(image,mask):
	edges = cv2.bitwise_and(image,image,mask=mask);
	red_mean = np.average(edges[:,:,0], weights=edges[:,:,0].astype(bool));
	green_mean = np.average(edges[:,:,1], weights=edges[:,:,1].astype(bool));
	blue_mean = np.average(edges[:,:,2], weights=edges[:,:,2].astype(bool));
	im_temp = copy.copy(image);
	im = np.zeros(im_temp.shape,dtype='int');	
	im[:,:,0] = abs(im_temp[:,:,0] - red_mean);
	im[:,:,1] = abs(im_temp[:,:,1] - green_mean);
	im[:,:,2] = abs(im_temp[:,:,2] - blue_mean);
	im = im.astype('uint8');
	im = cv2.bitwise_and(im,im,mask=mask)	
	red_mean = np.average(im[:,:,0], weights=im[:,:,0].astype(bool));
	green_mean = np.average(im[:,:,1], weights=im[:,:,1].astype(bool));
	blue_mean = np.average(im[:,:,2], weights=im[:,:,2].astype(bool));
	ret1, grad0 = cv2.threshold (im[:,:,0],red_mean, 255, cv2.THRESH_BINARY);
	ret1, grad1 = cv2.threshold (im[:,:,1],green_mean, 255, cv2.THRESH_BINARY);
	ret1, grad2 = cv2.threshold (im[:,:,2],blue_mean, 255, cv2.THRESH_BINARY);
	grad = cv2.bitwise_and(grad0,grad1);
	grad = cv2.bitwise_and(grad,grad2);
	grad = morphology.binary_closing(grad,iterations=5);
	grad = morphology.binary_fill_holes(grad);
	grad = grad.astype('uint8');
	im = cv2.bitwise_and(image,image,mask=grad);	
	return (im,grad);

def remove_unwanted(mask,image):
	_,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);	
	mask1 = np.zeros(image[:,:,0].shape,np.uint8);		
	for cnt in contours:
			mask = np.zeros(image[:,:,0].shape,np.uint8);		
			cv2.drawContours(mask,[cnt],0,255,-1);
			area = cv2.contourArea(cnt);
			im = cv2.bitwise_and(image,image,mask=mask);
			_,grad = remove_abundant(image,mask);
			if(cv2.countNonZero(grad)/area>0.5):
				print cv2.countNonZero(grad)/area;
				cv2.drawContours(mask1,[cnt],0,255,-1);						
	return(mask1); 
	
def remove_outliers(image,mask):
#taking the mask part to image to check the presence of bee
	im = cv2.bitwise_and(image,image,mask=mask);
	ldp_image,_,_ = ldp.ldp(im);
	test_Y = ldp_image.reshape((ldp_image.shape[0] * ldp_image.shape[1], ldp_image.shape[2]));
	test_rgb = im.reshape((im.shape[0] * im.shape[1], im.shape[2]));
	test = np.concatenate((test_Y,test_rgb),axis=1);
	mask_not = cv2.bitwise_not(mask);
	ret1, mask_not = cv2.threshold (mask_not,np.mean(mask_not), 255, cv2.THRESH_BINARY);		
	im = cv2.bitwise_and(image,image,mask=mask_not);
	ldp_image,_,_ = ldp.ldp(im);	
	data_ldp = ldp_image.reshape((ldp_image.shape[0] * ldp_image.shape[1], ldp_image.shape[2]));
	data_rgb = im.reshape((im.shape[0] * im.shape[1], im.shape[2]));
	data = np.concatenate((data_rgb,data_ldp),axis=1);
	data = data[np.any(data!=0,axis=1)];	
	print data.shape;		
	data = data.astype('float64');		
	data = preprocessing.normalize(data,axis=0);
	ss = StandardScaler();	
	data = ss.fit_transform(data);
	clf = svm.OneClassSVM(nu=0.8, kernel="rbf", gamma=0.1)
	clf.fit(data);
	test = test.astype('float64');		
	test = preprocessing.normalize(test,axis=0);	
	print test.shape;	
	test = ss.fit_transform(test);
	test = clf.predict(test);
	test = test.reshape((image.shape[0] , image.shape[1]));
	test[test==-1] = 0;
	test[test==1] = 255;
	test = test.astype('uint8');
	im = cv2.bitwise_and(image,image,mask=test);	
	im = cv2.bitwise_and(im,im,mask=mask);	
	#print test[:,0],test[:,1];	
	return(im,test);  
	


	
	
