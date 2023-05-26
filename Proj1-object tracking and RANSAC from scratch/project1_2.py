import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Step 1: Import the necessary libraries

# Step 2: Read the .csv data file and store it in a  dataframe
df = np.genfromtxt('pc1.csv', delimiter=',')#pc1 data frame
df = np.nan_to_num(df, nan=0)
df1 = np.genfromtxt('pc2.csv', delimiter=',')#pc2 dataframe
df1 = np.nan_to_num(df, nan=0)
# Step 3: Extract the x, y and z data from the dataframe and convert them into numpy arrays
x = df[:, 0]
y = df[:, 1]
z = df[:, 2]
x1 = df1[:, 0]
y1 = df1[:, 1]
z1 = df1[:, 2]

############################### Calculating the Covariance for pc1 #################################
#Calculating mean of the dataframe
mean = np.mean(df, axis=0)

# Calculate the covariance matrix

covariance = 1/(df.shape[0]-1) * (df - mean).T.dot(df - mean)
# compute the eigenvectors and eigenvalues of the covariance matrix
eigvals, eigvecs = np.linalg.eig(covariance)

# find the index of the smallest eigenvalue
min_eigval_idx = np.argmin(eigvals)
# the corresponding eigenvector is the surface normal
normal = eigvecs[:, min_eigval_idx]
magnitude = np.sqrt(eigvals[min_eigval_idx])
print("covariance is:")
print(covariance)
print("Direction vector is:")
print(normal)
print("Magnitude is:")
print(magnitude)
############################# Standard Least square and Total least square ##########################
def svd(A):
    """
    Compute Singular Value Decomposition (SVD) of matrix A using NumPy.
    Returns matrices U, S, V such that A = U @ S @ V.T.
    """
    # Compute eigenvalues and eigenvectors of A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    
    # Sort the eigenvalues in decreasing order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Compute diagonal matrix of singular values from eigenvalues
    singular_values = np.sqrt(eigenvalues)
    S = np.diag(singular_values)
    
    # Compute matrix U
    U = A @ eigenvectors
    U = U / singular_values
    
    # Compute matrix V
    V = eigenvectors
    
    return U, S, V.T

# Step 4: Define the standard least square and total least square functions to fit a surface to the data
def standard_least_square(x, y, z):
    X = np.column_stack([x, y, np.ones_like(x)])
    A = np.linalg.pinv(X.T @ X) @ X.T @ z
    return A

def total_least_square(x, y, z):
    X = np.column_stack([x, y, z])
    V = X - X.mean(axis=0)
    U, S, VT = svd(V)
    A = -VT[-1, :-1] / VT[-1, -1]
    B = np.append(A, 1)
    return B

# Step 5: Use the standard least square and total least square functions to fit a surface to the data and obtain the coefficients
A_standard = standard_least_square(x, y, z)
B_total = total_least_square(x, y, z)
A_standard1 = standard_least_square(x1, y1, z1)
B_total1 = total_least_square(x1, y1, z1)

# Step 6: Plot the surface using the obtained coefficients for both methods
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
z_standard = A_standard[0]*x_grid + A_standard[1]*y_grid + A_standard[2]
z_total = B_total[0]*x_grid + B_total[1]*y_grid + B_total[2]

x_grid1, y_grid1= np.meshgrid(np.linspace(x1.min(), x1.max(), 100), np.linspace(y1.min(), y1.max(), 100))
z_standard1 = A_standard1[0]*x_grid1 + A_standard1[1]*y_grid1 + A_standard1[2]
z_total1 = B_total1[0]*x_grid1 + B_total1[1]*y_grid1 + B_total1[2]

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(211, projection='3d')
ax1.scatter(x, y, z)
ax1.plot_surface(x_grid, y_grid, z_standard, alpha=0.5, cmap='viridis')
ax1.set_title('Standard Least Square for pc1')

ax2 = fig.add_subplot(212, projection='3d')
ax2.scatter(x, y, z)
ax2.plot_surface(x_grid, y_grid, z_total, alpha=0.5, cmap='viridis')
ax2.set_title('Total Least Square for pc1')

fig = plt.figure(figsize=(10,5))
ax3 = fig.add_subplot(211, projection='3d')
ax3.scatter(x1, y1, z1)
ax3.plot_surface(x_grid1, y_grid1, z_standard1, alpha=0.5, cmap='viridis')
ax3.set_title('Standard Least Square for pc2')

ax4 = fig.add_subplot(212, projection='3d')
ax4.scatter(x1, y1, z1)
ax4.plot_surface(x_grid1, y_grid1, z_total1, alpha=0.5, cmap='viridis')
ax4.set_title('Total Least Square for pc2')
plt.show()
############################################ RANSAC using Total least square method #############################################

# Function to calculate the distance 

def math_model(x_p,y_p,z_p,coef,threshold=0.1):
    distance=np.empty(1,)
    x_m=np.mean(x_p)
    y_m=np.mean(y_p)
    z_m=np.mean(z_p)
    for i in range(x_p.shape[0]):
        #distance between the  hypothesis plane and the points in the given dataset
        dist=(np.abs((coef[0]*(x_p[i]-x_m))+(coef[1]*(y_p[i]-y_m))+(coef[2]*(z_p[i]-z_m))))/np.sqrt(((coef[0]**2)+(coef[1]**2)+coef[2]**2))
        #dataset of all distances of Points 
        distance=np.append(distance,dist)

    suc=np.where(distance<=threshold)[0].shape[0]
    return suc

#RANSAC using Toltal least square method
def rtls(x_p,y_p,z_p):
    hypo=[]
    p_list=np.empty(0)
    inliers=np.empty(0)
    t_result=[]
    s=1100 #Number of iterations 
    for i in range(0,s):
        points=np.random.randint(0,x_p.shape[0],(3,))
        p_list=np.append(p_list,points)
        x=x_p[points]
        y=y_p[points]
        z=z_p[points]
        coef=total_least_square(x,y,z)
        suc=math_model(x_p,y_p,z_p,coef)
        t_result.append(suc)
        inliers=np.append(inliers,suc)
        hypo.append(coef)
    
    best=np.argmax(inliers)
    return hypo[best],inliers[best]


#Calling the functions for both the point clouds 
p1=total_least_square(x,y,z)

p2=total_least_square(x1,y1,z1)

rancof,inliers=rtls(x,y,z)

rancof_2,inliers_2=rtls(x1,y1,z1)

#Displaying the figures

fig3=plt.figure(3)
ax5=fig3.add_subplot(211,projection='3d')
xx, yy = np.meshgrid(x,y)
zz = np.mean(z)-(((rancof[0]*(xx-np.mean(x)))+(rancof[1]*(yy-np.mean(y)))) / (rancof[2]))
ax5.plot_surface(xx, yy, zz, alpha=0.5)
ax5.scatter(x,y,z)
ax5.set_title("Ransac for PC1 using tls")

ax5=fig3.add_subplot(212,projection='3d')
xx1, yy1 = np.meshgrid(x1,y1)
zz1 = np.mean(z1)-(((rancof_2[0]*(xx1-np.mean(x1)))+(rancof_2[1]*(yy1-np.mean(y1)))) / (rancof_2[2]))
ax5.plot_surface(xx1, yy1, zz1, alpha=0.5)
ax5.scatter(x1,y1,z1)
ax5.set_title("Ransac for PC2 using tls")
plt.show()