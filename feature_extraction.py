# imports
import math
import numpy as np                     # numeric python lib
import pandas as pd
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
import scipy.sparse
from skimage import measure,io         # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality

from pylab import rcParams
                    
import os, os.path 

# cartesian to polar coordinates, just as the image shows above
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  
 
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# Columns
columns = [ ]
columns.append('Mean')
columns.append('Variance')
columns.append('total_maxima')
columns.append('total_minima')
columns.append('axis-y/axis-x')
columns.append('area/rounded_length')

# Import train set
train = pd.read_csv('train.csv')

# Import train set
test = pd.read_csv('test.csv')

train_collection = pd.DataFrame(columns=columns)
test_collection = pd.DataFrame(columns=columns)

# Image path and valid extensions
imageDir = 'images_resize' #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]    

# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

for file_path in image_path_list:

    # reading an image file using matplotlib into a numpy array
    img = mpimg.imread(file_path)
    img = rgb2gray(img)   
    
    base = os.path.basename(file_path)  # Ex. images_resize//1.jpg
    name = os.path.splitext(base)[0]    # name = 1
    
    # using image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)

#    plt.imshow(img, cmap='Set3')  # show me the leaf
#    plt.scatter(cx, cy)           # show me its center
#    plt.show()

    # scikit-learn imaging contour finding, returns a list of found edges
    contours = measure.find_contours(img, .8)

    # from which we choose the longest one
    contour = max(contours, key=len)
    
    distances = []
    
    for y, x in contour:
        dist = calculateDistance(cx,x,cy,y)
        distances.append(dist)
    
    # axis-y length & axis-x length
    axis_y_len = pd.DataFrame(contour).loc[:,0].max()-pd.DataFrame(contour).loc[:,0].min()
    axis_x_len = pd.DataFrame(contour).loc[:,1].max()-pd.DataFrame(contour).loc[:,1].min()
    
    # Find leaf area
    area = PolyArea(contour[::,1],contour[::,0])
    
    # Find rounded length
    rounded_length = len(contour[::,1])
    
#    print(pd.DataFrame(contour).loc[:,1].max())
#    print(pd.DataFrame(contour).loc[:,1].min())
#    
#    print(pd.DataFrame(contour).loc[:,0].max())
#    print(pd.DataFrame(contour).loc[:,0].min())
#    print(ratio)
    
#    let us see the contour that we hopefully found
#    plt.plot(contour[::,1], contour[::,0], linewidth=0.5)  # (I will explain this [::,x] later)
#    plt.imshow(img, cmap='Set3')
#    plt.scatter(cx, cy)
#    plt.show()

    # move the center to (0,0)
    contour[::,1] -= cx  # demean X
    contour[::,0] -= cy  # demean Y

    # just calling the transformation on all pairs in the set
    polar_contour = np.array([cart2pol(x, y) for x, y in contour])

    from scipy.signal import argrelextrema
    
    # for local maxima
    c_max_index = argrelextrema(polar_contour[::,0], np.greater, order=100)
    c_min_index = argrelextrema(polar_contour[::,0], np.less, order=100)
    
    # Number of maxima & minima
    maxima = len(polar_contour[::,1][c_max_index])
    minima = len(polar_contour[::,1][c_min_index])
    
#    print(len(polar_contour[::,1][c_max_index]))
#    print(len(polar_contour[::,1][c_min_index]))
#    plt.subplot(121)
#    plt.scatter(polar_contour[::,1], polar_contour[::,0], 
#                linewidth=0, s=2, c='k')
#    plt.scatter(polar_contour[::,1][c_max_index], 
#                polar_contour[::,0][c_max_index], 
#                linewidth=0, s=30, c='b')
#    plt.scatter(polar_contour[::,1][c_min_index], 
#                polar_contour[::,0][c_min_index], 
#                linewidth=0, s=30, c='r')
#    
#    plt.subplot(122)
#    plt.scatter(contour[::,1], contour[::,0], 
#                linewidth=0, s=2, c='k')
#    plt.scatter(contour[::,1][c_max_index], 
#                contour[::,0][c_max_index], 
#                linewidth=0, s=30, c='b')
#    plt.scatter(contour[::,1][c_min_index], 
#                contour[::,0][c_min_index], 
#                linewidth=0, s=30, c='r')
#    
#    plt.show()
    
    #Attribute collector
    attributes = [0 for i in range(6)]
    
    attributes[0] = np.mean(distances)         # Mean of distance between center point and surround point
    attributes[1] = np.var(distances)          # Variance of distance between center point and surround point
    attributes[2] = maxima                     # Number of maxima
    attributes[3] = minima                     # Number of minima
    attributes[4] = axis_y_len/axis_x_len      # Ratio between axis-y length and axis-x length
    attributes[5] = area/rounded_length        # Ratio bewteen area and rounded length
    
    if int(name) in list(train.loc[:,'id']):
        train_collection = train_collection.append(pd.DataFrame([attributes],index=[int(name)] , columns=columns))
    else:
        test_collection = test_collection.append(pd.DataFrame([attributes],index=[int(name)] , columns=columns))

# Join species column to train_collection
train_collection = train_collection.sort_index()
species = pd.DataFrame(list(train.loc[:,'species']), index=list(train_collection.index), columns=['species'])
train_collection = train_collection.join(species)

# Reorder columns
cols = list(train_collection)
cols.insert(0, cols.pop(cols.index('species')))
train_collection = train_collection.ix[:,cols]

test_collection = test_collection.sort_index()
    
train_collection.to_csv("output_train.csv",sep=',')
test_collection.to_csv("output_test.csv",sep=',')
