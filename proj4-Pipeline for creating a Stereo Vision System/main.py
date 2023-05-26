import cv2
import numpy as np

def calculate_disparity(img_left, img_right,ndisp, vmin ,vmax, window_size=5):
    if len(img_left.shape) == 2:
        img_left = np.expand_dims(img_left, axis=-1)
        img_right = np.expand_dims(img_right, axis=-1)
    height, width, channels = img_left.shape
    half_window = window_size // 2
    disparity_range = np.arange(ndisp) + vmin 
    cost_volume = np.zeros((height, width, ndisp), dtype=np.float32)
    for d in range(ndisp):
        if d == 0:
            cost_volume[:, :, d] = np.sum((img_left - img_right)**2, axis=-1)
        else:
            cost_volume[:, d:, d] = np.sum((img_left[:, d:] - img_right[:, :-d])**2, axis=-1)
    disparity = disparity_range[np.argmin(cost_volume, axis=2)]
    disparity = np.uint8((disparity - vmin) / (vmax - vmin) * 255)
    return disparity
def calculate_depth(disparity, baseline, focal_length):
    depth = np.zeros_like(disparity, dtype=np.float32)
    mask = disparity > 0
    depth[mask] = (baseline * focal_length) / disparity[mask]
    return depth
def normalize_disparity(disparity):
    disparity_min = np.min(disparity)
    disparity_max = np.max(disparity)
    return ((disparity - disparity_min) / (disparity_max - disparity_min) * 255).astype(np.uint8)
#refernce-https://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/homography.pdf
def rectify_images(F, K1, K2, pts1, pts2, img1, img2,inliners0,inliners1):
    # Compute the Essential matrix
    K1=np.array(K1)
    K2=np.array(K2)
    E = estimate_Essential_Matrix(F, K1, K2)
    # define the image size
    img_size = (img1.shape[0], img2.shape[1])
    # Compute the rotation and translation matrices
    R, t = Rt(E)
    R, T = triangulation(R, t,pts1, pts2,K1,K2) 
    _,H1,H2 = cv2.stereoRectifyUncalibrated(inliners0,inliners1, F, img_size)
    img1_rect = cv2.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
    img2_rect = cv2.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))
    img1_new=np.copy(img1_rect)
    img2_new=np.copy(img2_rect)
    # https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/epipolarGeometry.py
    lines1 = cv2.computeCorrespondEpilines(inliners1.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_lines = img1_rect
    #Draw epilines in the first image
    for r, pt in zip(lines1,inliners0):
        pt1 = [int(pt[0]),int(pt[1])]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img1_rect.shape[1], -(r[2]+r[0]*img1_rect.shape[1])/r[1]])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        img1_lines = cv2.circle(img1_lines,tuple(pt1),2,color,-1)
    # Draw epilines in the second image
    lines2 = cv2.computeCorrespondEpilines(inliners0.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_lines = img2_rect
    for r, pt in zip(lines2,inliners1):
        pt2 = [int(pt[0]),int(pt[1])]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img2_rect.shape[1], -(r[2]+r[0]*img2_rect.shape[1])/r[1]])
        img2_lines = cv2.line(img2_lines, (x0, y0), (x1, y1), color, 1)
        img2_lines = cv2.circle(img2_lines,tuple(pt2),2,color,-1)
    
    return H1,H2,img1_rect, img2_rect,img1_new,img2_new

def normalize_points(pts):
    """
    Centroid Normalization
    """
    # Compute the mean of the points
    centroid = np.mean(pts, axis=0)
    
    # Translate the points to the origin
    pts_trans = pts - centroid
    
    # Compute the scale factor to make the RMS distance from the origin to be sqrt(2)
    dist_rms = np.sqrt(np.mean(np.sum(pts_trans**2, axis=1)))
    scale = np.sqrt(2) / dist_rms
    
    # Construct the normalization matrix
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])
    
    # Normalize the points
    pts_norm = np.dot(T, np.concatenate((pts, np.ones((len(pts), 1))), axis=1).T).T[:, :2]
    return pts_norm, T
#Reference-http://ai.stanford.edu/~birch/projective/node20.html#:~:text=Thus%20both%20the%20Essential%20and,latter%20deals%20with%20uncalibrated%20cameras.
def estimate_Essential_Matrix(F, K1, K2):
    E = np.dot(K2.T, np.dot(F, K1))
    U, S, Vt = np.linalg.svd(E)
    S = np.array([[1,0,0], [0,1,0], [0,0,0]])
    E = np.dot(U, np.dot(S, Vt))

    return E
#finding rt according to the slides provided and the link given 
def Rt(E):

    # perform singular value decomposition of E
    U, S, Vt = np.linalg.svd(E)
    # define a skew-symmetric matrix W
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # compute two possible rotation matrices R1 and R2
    R_1 = np.dot(U, W)
    R_1 = np.dot(R_1, Vt)
    c1 = U[:, 2]
    if (np.linalg.det(R_1) < 0):
        R_1 = -R_1
        c1 = -c1 
    R_2 = R_1
    c2 = -U[:, 2]
    if (np.linalg.det(R_2) < 0):
        R_2 = -R_2
        c2 = -c2
    # compute two more possible rotation matrices R3 and R4
    R_3 = np.dot(U, W.T)
    R_3 = np.dot(R_1, Vt)
    c3 = U[:, 2]
    if (np.linalg.det(R_3) < 0):
        R_3 = -R_3
        c3 = -c3
    R_4 = R_3
    c4 = -U[:, 2]
    if (np.linalg.det(R_4) < 0):
        R_4 = -R_4
        c4 = -c4
    # reshape the translation vectors to (3,1) matrices
    c1 = c1.reshape((3,1))
    c2 = c2.reshape((3,1))
    c3 = c3.reshape((3,1))
    c4 = c4.reshape((3,1))
    # store the rotation matrices and translation vectors in lists
    r_f = [R_1,R_2,R_3,R_4]
    c_f = [c1,c2,c3,c4]
    # return the list of possible rotation matrices and their corresponding translation vectors
    return r_f,c_f
#estimating point correspondences 
def tD_pts(r2,c2,pts1,pts2,cam0,cam1):
    c1 = np.array([[0],[0],[0]])
    
    # Set the identity matrix as the first camera rotation matrix and second camera rotation matrix as r2
    r1 = np.identity(3)

    # Calculate the vectors from camera centers to rotation centers as r1_c1 and r2_c2 respectively
    r1_c1dot = -np.dot(r1,c1)
    r2_c2dot = -np.dot(r2,c2)
    # Concatenate the rotation matrices and the vectors from camera centers to rotation centers
    j1_con = np.concatenate((r1, r1_c1dot), axis = 1)
    j2_con = np.concatenate((r2, r2_c2dot), axis = 1)
    # Calculate projection matrices for the two cameras
    P1_dot = np.dot(cam0,j1_con)
    P2_dot = np.dot(cam1,j2_con)
    # Initialize empty list to store the triangulated 3D points
    t_mat = []

    # Loop through all the image points and triangulate them
    for i in range(len(pts1)):
        
        # Get the image points from both the cameras
        x_1 = np.array(pts1[i])
        x_2 = np.array(pts2[i])
        
        # Reshape the image points to match the dimensions of the projection matrix
        x_1 = np.reshape(x_1,(2,1))
        q = np.array([1]).reshape((1,1))
        
        # Concatenate a 1 to the image point to make it homogeneous
        x_1 = np.concatenate((x_1,q), axis = 0)
        x_2 = np.reshape(x_2,(2,1))
        x_2 = np.concatenate((x_2,q), axis = 0)

        # Calculate the skew-symmetric matrix for both image points
        x_1_skewmatrix = np.array([[0,-x_1[2][0],x_1[1][0]],[x_1[2][0], 0, -x_1[0][0]],[-x_1[1][0], x_1[0][0], 0]])
        x_2_skewmatrix = np.array([[0,-x_2[2][0],x_2[1][0]],[x_2[2][0], 0, -x_2[0][0]],[-x_2[1][0], x_2[0][0], 0]])
        
        # Combine the skew-symmetric matrix with the projection matrix for both cameras
        A1_dot = np.dot(x_1_skewmatrix, P1_dot)
        A2_dot = np.dot(x_2_skewmatrix, P2_dot)
        
        # Combine the two matrices A1 and A2 to create a 6x4 matrix A
        A = np.zeros((6,4))
        for i in range(6):
            if i<=2:
                A[i,:] = A1_dot[i,:]
            else:
                A[i,:] = A2_dot[i-3,:]
        
        # Use Singular Value Decomposition (SVD) to solve for the homogeneous coordinate of the triangulated point
        U, sigma, VT = np.linalg.svd(A)
        VT = VT[3]
        VT = VT/VT[-1]
        t_mat.append(VT)
    t_mat = np.array(t_mat) 
    
    return t_mat    
#checks the cheirality condition 
def triangulation(R_lis,C_lis,p1,p2,cam0,cam1):
    
    clis = list()
    for i in range(4):
        # calling tD_pts function to obtain 3D coordinates of corresponding points in two cameras
        x = tD_pts(R_lis[i],C_lis[i],p1,p2,cam0,cam1)
        n = 0
        for j in range(x.shape[0]):
            cord = x[j,:].reshape(-1,1)
            # check if the triangulated point satisfies the cheirality condition
            if np.dot(R_lis[i][2], (cord[0:3] - C_lis[i])) > 0 and cord[2]>0:
                n += 1
        # append number of points satisfying the cheirality condition
        clis.append(n)
        # get the index of the camera with the most number of points satisfying the cheirality condition
        ind = clis.index(max(clis))
        # check if the camera's translation vector is in front of the camera
        if C_lis[ind][2]>0:
            C_lis[ind] = -C_lis[ind]
    # return the rotation and translation matrices of the camera with the most number of points satisfying the cheirality condition
    return R_lis[ind], C_lis[ind]
#calculating the fundamental matrix by shifting the origin to mean, then applying svd and unnormalising the points to get F
def Funda(f1,f2):
    # Initialize arrays
    f1_x = [] ; f1_y = [] ; f2_x = [] ; f2_y = []
    # Convert input lists to numpy arrays
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    # Compute the mean values of the points
    f1_xmean = np.mean(f1[:,0])    
    f1_ymean = np.mean(f1[:,1])    
    f2_xmean = np.mean(f2[:,0])        
    f2_ymean = np.mean(f2[:,1])
    # Subtract mean values from the points
    for i in range(len(f1)): f1[i][0] = f1[i][0] - f1_xmean
    for i in range(len(f1)): f1[i][1] = f1[i][1] - f1_ymean
    for i in range(len(f2)): f2[i][0] = f2[i][0] - f2_xmean
    for i in range(len(f2)): f2[i][1] = f2[i][1] - f2_ymean
    # Extract x and y coordinates of points into separate arrays
    f1_x = np.array(f1[:,0])
    f1_y = np.array(f1[:,1])
    f2_x = np.array(f2[:,0])
    f2_y = np.array(f2[:,1])
    # Compute the sum of squares of the points
    sum_f1 = np.sum((f1)**2, axis = 1)
    sum_f2 = np.sum((f2)**2, axis = 1)
    #scaling factors 
    k_1 = np.sqrt(2.)/np.mean(sum_f1**(1/2))
    k_2 = np.sqrt(2.)/np.mean(sum_f2**(1/2))
    # Define scaling matrices                        
    s_f1_1 = np.array([[k_1,0,0],[0,k_1,0],[0,0,1]])
    s_f1_2 = np.array([[1,0,-f1_xmean],[0,1,-f1_ymean],[0,0,1]])
    
    s_f2_1 = np.array([[k_2,0,0],[0,k_2,0],[0,0,1]])
    s_f2_2 = np.array([[1,0,-f2_xmean],[0,1,-f2_ymean],[0,0,1]])
    # Compute the total scaling matrix
    t_1 = np.dot(s_f1_1,s_f1_2)
    t_2 = np.dot(s_f2_1,s_f2_2)
    # Scale the points
    x1 = ( (f1_x).reshape((-1,1)) ) * k_1
    y1 = ( (f1_y).reshape((-1,1)) ) * k_1
    x2 = ( (f2_x).reshape((-1,1)) ) * k_2
    y2 = ( (f2_y).reshape((-1,1)) ) * k_2
    # A (8X9) matrix
    Alist = []
    for i in range(x1.shape[0]):
        X1, Y1 = x1[i][0],y1[i][0]
        X2, Y2 = x2[i][0],y2[i][0]
        Alist.append([X2*X1 , X2*Y1 , X2 , Y2 * X1 , Y2 * Y1 ,  Y2 ,  X1 ,  Y1, 1])
    A = np.array(Alist)
    #Find SVD 
    U, S, VT = np.linalg.svd(A)
    
    v = VT.T
    
    f_val = v[:,-1]
    f_mat = f_val.reshape((3,3))
    
    Uf, S_f, Vf = np.linalg.svd(f_mat)
    #forcing the rank 2 constraint
    S_f[-1] = 0
    
    S_final = np.zeros(shape=(3,3)) 
    S_final[0][0] = S_f[0] 
    S_final[1][1] = S_f[1] 
    S_final[2][2] = S_f[2] 
    #un-normalizing 
    f_main = np.dot(Uf , S_final)
    f_main = np.dot(f_main , Vf)
    
    f_un = np.dot(t_2.T , f_main)
    f_un = np.dot(f_un , t_1)
    # if f_un/f_un[-1,-1]==0: continue
    f_un = f_un/f_un[-1,-1]
    
    return f_un

#RANSAC implementation
def RANSAC(pts1,pts2):
   
    #parameters
    N_iter = 1000 #Number of iterations
    s = 0#number of samples
    threshold = 0.05
    inliers_amt = 0#number of inliers
    P = 0.99#success rate
    best_f = []#list for best f

    while s < N_iter:
        random_p1 = [] ; random_p2 = []
        
        #getting a set of random 8 points
        index = np.random.randint( len(pts1) , size = 8)
        
        for i in index:
            random_p1.append(pts1[i])
            random_p2.append(pts2[i])
        #calling the matrix
        Fundamental_matrix = Funda(random_p1, random_p2)
        
        #Hartley's 8 points algorithm
        ones = np.ones((len(pts1),1))
        x_1 = np.concatenate((pts1,ones),axis=1)
        x_2 = np.concatenate((pts2,ones),axis=1)
        
        l_1 = np.dot(x_1, Fundamental_matrix.T)
        
        l_2 = np.dot(x_2,Fundamental_matrix)
    
        error1 = np.sum(l_2* x_1,axis=1,keepdims=True)**2
        
        error2 = np.sum(np.hstack((l_1[:, :-1],l_2[:,:-1]))**2,axis=1,keepdims=True)
        
        error =  error1 / error2 
        
        inliers = error <= threshold
         
        inlier_count = np.sum(inliers)
        
        #estimating best Fundamental M
        if inliers_amt <  inlier_count:
            
            inliers_amt = inlier_count
            
            g_ones = np.where(inliers == True)
            
            i_1pts = np.array(pts1)
            i_2pts = np.array(pts2)
            
            inliers1 = i_1pts[g_ones[0][:]]
            inliers2 = i_2pts[g_ones[0][:]]

            best_f = Fundamental_matrix
            
        #iterating for N number of times
        inlier_ratio = inlier_count/len(pts2)
        
        deno = np.log(1-(inlier_ratio**8))
        
        num = np.log(1-P)
        
        if deno == 0: continue
        N_iter =  num / deno
        s += 1
        
    return best_f, inliers1, inliers2
def OutputFunction(img0,img1,cam0,cam1):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img0, None)
    kp2, des2 = orb.detectAndCompute(img1, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Filter matches
    pts0 = []
    pts1 = []
    for match in matches:
        pts0.append(kp1[match.queryIdx].pt)
        pts1.append(kp2[match.trainIdx].pt)
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    img_matches = cv2.drawMatches(img0, kp1, img1, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    F,inliners0,inliners1=RANSAC(pts0,pts1)
    cam0=np.array(cam0)
    cam1=np.array(cam1)
    K=np.array(cam0)
    E=estimate_Essential_Matrix(F, cam0, cam1)
    R,t=Rt(E)
    R, T = triangulation(R, t, pts0, pts1,cam0,cam1)
    return F,E,R,T,pts0,pts1,img_matches,inliners0,inliners1
###start###
img0_a = cv2.imread('artroom/artroom/im0.png', 0)
img1_a = cv2.imread('artroom/artroom/im1.png', 0)
img0_c = cv2.imread('chess/chess/im0.png', 0)
img1_c = cv2.imread('chess/chess/im1.png', 0)
img0_l = cv2.imread('ladder/ladder/im0.png', 0)
img1_l = cv2.imread('ladder/ladder/im1.png', 0)
#for artroom
cam0_a=[[1733.74, 0, 792.27],
        [0, 1733.74, 541.89],
        [0, 0, 1]]
cam1_a=[[1733.74 ,0, 792.27],
        [ 0, 1733.74, 541.89],
        [ 0, 0, 1]]
baseline=536.62
width=1920
height=1080
ndisp=170
vmin=55
vmax=142
focal_length=1733.74
F,E,R,t,pts0,pts1,img_matches,inliners0,inliners1=OutputFunction(img0_a,img1_a,cam0_a,cam1_a)
H1,H2,img1_rect,img2_rect,img1_new,img2_new=rectify_images(F, cam0_a, cam1_a, pts0, pts1, img0_a, img1_a,inliners0,inliners1)
disparity = calculate_disparity(img1_new,img2_new,ndisp,vmin,vmax)
disparity_normalized = normalize_disparity(disparity)
scale_percent =25 # percent of original size
width = int(img1_rect.shape[1] * scale_percent / 100)
height = int(img1_rect.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img1_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img1_rect_artroom.png',resized)
width = int(img2_rect.shape[1] * scale_percent / 100)
height = int(img2_rect.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img2_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img2_rect_artroom.png',resized)
cv2.imwrite('disparity_grayscale_artroom.png', disparity_normalized)
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_HOT)
cv2.imwrite('disparity_color_artoom.png', disparity_color)
depth = calculate_depth(disparity_normalized, baseline, focal_length)
cv2.imwrite('depth_grayscale_artroom.png', depth)
depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/np.max(depth)), cv2.COLORMAP_JET)
cv2.imwrite('depth_colormap_artroom.png', depth_color)
print("VALUES FOR ART ROOM")
print("Fundamental Matrix for art room\n",F)
print("Essential Matrix for art room \n",E)
print("First Rotation for art room \n",R,"\nTranslation for art room \n",t.T)
print("\nLeft image Homogrphy matrix \n",H1,"\nRight image homogrpahy matrix\n",H2)
#for chess
cam0_c=[[1758.23, 0, 829.15],
        [ 0, 1758.23, 552.78],
        [ 0 ,0 ,1]]
cam1_c=[[1758.23, 0, 829.15 ],
        [0, 1758.23, 552.78],
        [0, 0, 1]]
baseline_c=97.99
print("VALUES FOR CHESS")
F,E,R,T,pts0,pts1,img_matches,inliners0,inliners1=OutputFunction(img0_c,img1_c,cam0_c,cam1_c)
H1,H2,img1_rect,img2_rect,img1_new,img2_new=rectify_images(F, cam0_c, cam1_c, pts0, pts1, img0_c, img1_c,inliners0,inliners1)
print("Fundamental Matrix for chess\n",F)
print("Essential Matrix for chess \n",E)
print("Rotation for chess \n",R,"\nTranslation for chess \n",T.T)
print("\nLeft image Homogrphy matrix \n",H1,"\nRight image homogrpahy matrix\n",H2)
scale_percent =25 # percent of original size
width = int(img1_rect.shape[1] * scale_percent / 100)
height = int(img1_rect.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img1_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img1_rect_chess.png',resized)
width = int(img2_rect.shape[1] * scale_percent / 100)
height = int(img2_rect.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img2_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img2_rect_chess.png',resized)
ndisp=220
vmin=65
vmax=197
baseline=97.99
focal_length=1758.23
disparity = calculate_disparity(img1_new,img2_new,ndisp,vmin,vmax)
disparity_normalized = normalize_disparity(disparity)
cv2.imwrite('disparity_grayscale_chess.png', disparity_normalized)
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_HOT)
cv2.imwrite('disparity_color_chess.png', disparity_color)
depth = calculate_depth(disparity_normalized, baseline, focal_length)
cv2.imwrite('depth_grayscale_chess.png', depth)
depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/np.max(depth)), cv2.COLORMAP_JET)
cv2.imwrite('depth_colormap_chess.png', depth_color)
#for ladder
cam0_l=[[1734.16, 0, 333.49],
        [ 0, 1734.16, 958.05],
        [ 0, 0, 1]]
cam1_l=[[1734.16, 0, 333.49],
        [0, 1734.16, 958.05],
        [ 0 ,0 ,1]]
baseline_l=228.38
print("VALUES FOR LADDER")
F,E,R,T,pts0,pts1,img_matches,inliners0,inliners1=OutputFunction(img0_l,img1_l,cam0_l,cam1_l)
H1,H2,img1_rect,img2_rect,img1_new,img2_new=rectify_images(F, cam0_l, cam1_l, pts0, pts1, img0_l, img1_l,inliners0,inliners1)
width = int(img1_rect.shape[1] * scale_percent / 100)
height = int(img1_rect.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img1_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img1_rect_ladder.png',resized)
width = int(img2_rect.shape[1] * scale_percent / 100)
height = int(img2_rect.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img2_rect, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('img2_rect_ladder.png',resized)
ndisp=110
vmin=27
vmax=85
baseline=97.99
focal_length=1734.16
disparity = calculate_disparity(img1_new,img2_new,ndisp,vmin,vmax)
disparity_normalized = normalize_disparity(disparity)
cv2.imwrite('disparity_grayscale_ladder.png', disparity_normalized)
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_HOT)
cv2.imwrite('disparity_color_ladder.png', disparity_color)
depth = calculate_depth(disparity_normalized, baseline, focal_length)
cv2.imwrite('depth_grayscale_ladder.png', depth)
depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/np.max(depth)), cv2.COLORMAP_HOT)
cv2.imwrite('depth_colormap_ladder.png', depth_color)
print("Fundamental Matrix for ladder\n",F)
print("Essential Matrix for ladder \n",E)
print("Rotation for ladder \n",R,"\nTranslation for ladder \n",T.T)
print("\nLeft image Homogrphy matrix \n",H1,"\nRight image homogrpahy matrix\n",H2)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()