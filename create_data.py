# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import bee_preprocess as bee
from scipy.ndimage import morphology
import copy
import ldp
from skimage.feature import daisy
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
import kmeans as kmeans



# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
path = "/media/sai/New Volume1/Practice/beeImages/bee_images/train/"
out = "/media/sai/New Volume1/Practice/beeImages/bee_images/processed/"
labels = pd.read_csv("/media/sai/New Volume1/Practice/beeImages/train_label.csv")
data_0 = [];
data_1 = [];
count = 0;
for index,row in tqdm(labels.iterrows()):
	count = count+1;
	#if(count>200):
	#	break;
	#print row['id'];	
	fname =  path + str(int(row['id'])) + ".jpg";
	#if int(row['genus']) == 0 :	
	#	fout = out + "out0/" + str(int(row['id'])) + ".jpg"; 
	#else:	
	fout = out + str(int(row['id'])) + ".jpg";
	image = cv2.imread(fname,1);
	image,_,_ = bee.return_crop(image);
	#image = kmeans.kmeans(image)
#remove the probable background
	x,y,z = image.shape; 
	image = image.reshape((x * y, z));
	image[np.all(image==0,axis=1)] = tuple([0,0,50]);
	image = image.reshape((x,y,z));
	cv2.imwrite(fout,image)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
# directional gradient information
	#im,ldp_hist = ldp.ldp(image);
	#hist = ldp_hist.flatten();
	#hist = hist[hist!=0];
	#(hist,_) = np.histogram(hist,bins=256);
# color information
	'''	
	image = image.reshape((image.shape[0] * image.shape[1], 3));
	image = image[np.any(image!=0,axis=1)];
	hist_red,hist_green,hist_blue = bee.color_hist(image)
	data = np.concatenate((hist_red,hist_green,hist_blue),axis=1);
	if(np.isnan(data).any()):
		print "error at " + str(int(row['id'])) + ".jpg"
		break;
	if int(row['genus']) == 0 :
		data_0.append(data);
	else :
		data_1.append(data);

data_0 = np.vstack(data_0);
data_1 = np.vstack(data_1);
print data_0.shape
print data_1.shape
np.savetxt('/media/sai/New Volume1/Practice/beeImages/data_0.txt', data_0);
np.save('/media/sai/New Volume1/Practice/beeImages/data_0.npy', data_0);
np.savetxt('/media/sai/New Volume1/Practice/beeImages/data_1.txt', data_1);
np.save('/media/sai/New Volume1/Practice/beeImages/data_1.npy', data_1);
'''




