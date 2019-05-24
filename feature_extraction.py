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
from shapely.geometry import Polygon

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

# Import train set
train = pd.read_csv('train.csv')

# Import train set
test = pd.read_csv('test.csv')

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

# Columns
columns = list()
columns.append('Mean')
columns.append('Variance')
columns.append('total_maxima')
columns.append('total_minima')
columns.append('axis-y/axis-x')
columns.append('area/rounded_length')

# Build DataFrame for collecting extracted features
train_collection = pd.DataFrame(columns=columns)
test_collection = pd.DataFrame(columns=columns)

for file_path in image_path_list:

    # reading an image file using matplotlib into a numpy array
    img = mpimg.imread(file_path)
    img = rgb2gray(img)   
    
    base = os.path.basename(file_path)  # Ex. images_resize//1.jpg
    name = os.path.splitext(base)[0]    # name = 1
    
    # using image processing module of scipy to find the center of the leaf
    cy, cx = ndi.center_of_mass(img)
    # Visualize center point
    '''
    plt.imshow(img, cmap='Set3')  # show me the leaf
    plt.scatter(cx, cy)           # show me its center
    plt.show()
    continue
    exit()
    '''

    # scikit-learn imaging contour finding, returns a list of found edges
    contours = measure.find_contours(img, 200)
    
    # from which we choose the longest one
    contour = max(contours, key=len)
    shape = Polygon(contour)
    x, y = shape.exterior.xy
    # Visualize leaf perimeter
    '''
    plt.plot(x, y, color='red', alpha=0.7, linewidth=1, solid_capstyle='round')
    plt.show()
    exit()
    '''

    distances = []
    
    for y, x in contour:
        dist = calculateDistance(cx,x,cy,y)
        distances.append(dist)
    '''
    plt.plot(distances)
    plt.show()
    continue
    exit()
    '''

    # axis-y length & axis-x length
    bounds = shape.bounds # (minx, miny, maxx, maxy)
    axis_y_len = bounds[3] - bounds[1]
    axis_x_len = bounds[2] - bounds[0]
    
    # Find leaf area
    area = shape.area

    # Find rounded length
    rounded_length = shape.length

    # Visualize perimeter along with center point
    '''
    plt.plot(contour[::,1], contour[::,0], linewidth=0.5)  # (I will explain this [::,x] later)
    plt.imshow(img, cmap='Set3')
    plt.scatter(cx, cy)
    plt.show()
    exit()
    '''
    
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
    
    # Visualize maxima and minima
    '''
    print(maxima)
    print(minima)
    plt.subplot(121)
    plt.scatter(polar_contour[::,1], polar_contour[::,0], 
                linewidth=0, s=2, c='k')
    plt.scatter(polar_contour[::,1][c_max_index], 
                polar_contour[::,0][c_max_index], 
                linewidth=0, s=30, c='b')
    plt.scatter(polar_contour[::,1][c_min_index], 
                polar_contour[::,0][c_min_index], 
                linewidth=0, s=30, c='r')
    
    plt.subplot(122)
    plt.scatter(contour[::,1], contour[::,0], 
                linewidth=0, s=2, c='k')
    plt.scatter(contour[::,1][c_max_index], 
                contour[::,0][c_max_index], 
                linewidth=0, s=30, c='b')
    plt.scatter(contour[::,1][c_min_index], 
                contour[::,0][c_min_index], 
                linewidth=0, s=30, c='r')
    
    plt.show()
    continue
    '''
    #Attribute collector
    attributes = [0 for i in range(len(columns))]
    
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

# Merge extracted features with species
# Extract genus
sci_names = train.loc[:,['id', 'species']]
sci_names['genus'] = sci_names['species'].str.split('_').str.get(0)
# Merge
train_collection = train_collection.sort_index()
train_collection.reset_index(inplace=True)
train_collection.rename({'index': 'id'}, axis='columns', inplace=True)

train_collection = pd.merge(sci_names, train_collection, on='id', how='inner')
train_collection.set_index('id', inplace=True)

test_collection = test_collection.sort_index()
    
train_collection.to_csv("output_train.csv",sep=',')
test_collection.to_csv("output_test.csv",sep=',')
