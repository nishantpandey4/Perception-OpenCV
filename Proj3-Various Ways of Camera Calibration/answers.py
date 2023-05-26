#FIRST QUESTION
import numpy as np
from scipy import linalg
# Define the 3D world points 
world_points = np.array([    [0, 0, 0],
    [0, 3, 0],
    [0, 7, 0],
    [0, 11, 0],
    [7, 1, 0],
    [0, 11, 7],
    [7, 9, 0],
    [0, 1, 7]
], dtype=np.float32)

# Normalize the world points
world_points_norm = (world_points - np.mean(world_points, axis=0)) / np.std(world_points)

# Define the corresponding 2D image points
image_points = np.array([    [757, 213],
    [758, 415],
    [758, 686],
    [759, 966],
    [1190, 172],
    [329, 1041],
    [1204, 850],
    [340, 159]
], dtype=np.float32)

# Normalize the image points 

image_points_norm = (image_points - np.mean(image_points, axis=0)) / np.std(image_points)
#homogenize for the image points 
image_points_homogenous= np.concatenate((image_points_norm, np.ones((image_points_norm.shape[0], 1))), axis=1)

# Compute Linear Transform (DLT) algorithm to get the P matrix 
ones_col = np.ones((world_points_norm.shape[0], 1))

Homogenous_World = np.hstack((world_points_norm, ones_col)).T # 4*8 matrix for direct linear transformation homogenous coordinates
X = image_points_homogenous[:, 0]#x points
Y = image_points_homogenous[:, 1]#y points
Z = np.ones((world_points_norm.shape[0],))
A = np.zeros((2 * Homogenous_World.shape[1], 12)) 
for i in range(world_points_norm.shape[0]):
    A[2 * i] = np.concatenate((Homogenous_World[:, i], np.zeros(4), -X[i]*Homogenous_World[:, i]))
    A[2 * i + 1] = np.concatenate((np.zeros(4), Homogenous_World[:, i], -Y[i]*Homogenous_World[:, i]))

U, S, Vt = np.linalg.svd(A)
P = Vt[:,-1].reshape((3, 4))
# P = P / P[2, 3]
print("The projection matrix:\n",P)
#Part four of question one
# Define the Gram-Schmidt function 
# gram-schmidt - http://homepages.math.uic.edu/~jan/mcs507f13/gramschmidt.py
def gram_schmidt_method(matrix):
    Q = np.zeros_like(matrix)
    R = np.zeros((matrix.shape[1], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        v = matrix[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], matrix[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v) 
        Q[:, j] = v / R[j, j]
    return Q, R

#part five of question 1
# Apply the Gram-Schmidt process to decompose P into its intrinsic and extrinsic components
Q, R = gram_schmidt_method(P[:, :3])
M=P[:,:3]
K = R
K=K/K[-1,-1]
U,S,Vt=np.linalg.svd(P)
c=Vt[:,-1]
c = Vt[:,-1]
t = c[:-1] / c[-1]
# Compute the projected 2D image points using the camera matrix M and the 3D world points
proj_points = np.dot(P, np.vstack((world_points_norm.T, np.ones((1, world_points_norm.shape[0])))))
proj_points = proj_points[:2, :] / proj_points[2, :]
proj_points = proj_points.T
proj_points= np.concatenate((proj_points, np.ones((proj_points.shape[0], 1))), axis=1)

# Compute the root mean square (RMS) error between the projected and original 2D image points
diff =  np.absolute(image_points_homogenous-proj_points)
diff=diff[:,0]+diff[:,1]/2
# err = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
# Print the intrinsic and extrinsic matrices

print("Intrinsic matrix K for question 1 :\n", K)
print("Rotation matrix R:\n", Q)
print("Translation vector t:\n", t)
# Print the camera matrix M and the reprojection error
# print("Reprojection error for question 1:", err)
print("Reprojection error for each point",diff)

#SECOND QUESTION

import cv2
import os
import numpy as np
#loading the images
folder_path = 'Calibration_Imgs/Calibration_Imgs/'
image_files = os.listdir(folder_path)
images = []
for image_file in image_files:
    image = cv2.imread(os.path.join(folder_path, image_file))
    images.append(image)
# Define the number of squares on the x and y axes reducing one line of squares from both sides
nx = 9
ny = 6

# Define the size of each square in millimeters
square_size = 21.5

td_points = np.zeros((nx*ny, 3), np.float32)
td_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
td_points *= square_size
corner_points_list = []

# Loop through each image and detect the corners
for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    # Find the corners using the manual corner locations
    ret, corners = cv2.findChessboardCorners(img_small, (nx, ny), None)
    cv2.drawChessboardCorners(img_small, (nx, ny), corners,ret)
    #displaying the images
    cv2.imshow('image', img_small)
    # cv2.imwrite('image.jpg',img_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # If corners are found, add them to the list of corner points
    if ret == True:
        corner_points_list.append(corners)

# Calibrate the camera using the detected corner points
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
img_shape = gray.shape[::-1]
ret, cammatrix, dist, rvecs, tvecs = cv2.calibrateCamera([td_points]*len(corner_points_list), 
                                                   corner_points_list, 
                                                   img_shape, 
                                                   None, 
                                                   None)

# Compute the reprojection error for each image
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
reprojection_errors = []
for i in range(len(corner_points_list)):
    imgpoints, _ = cv2.projectPoints(td_points, rvecs[i], tvecs[i], cammatrix, dist)
    error_each_image = cv2.norm(corner_points_list[i], imgpoints, cv2.NORM_L2) / len(imgpoints)
    reprojection_errors.append(error_each_image)
    print("Image {}: Reprojection error: {}".format(i, error_each_image))
k=cammatrix[:3,:3]
print("Intrinsic Camera matrix for question 2: \n",k)

