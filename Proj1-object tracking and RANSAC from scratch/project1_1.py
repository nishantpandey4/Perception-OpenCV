import cv2
import numpy as np
# from scipy import optimize
import matplotlib.pyplot as plt
# Load the video
cap = cv2.VideoCapture('ball.mov')

x_co_ordinates = []

y_co_ordinates = []

# Define the lower and upper boundaries of the red color in HSV
lower_red = np.array([0, 110, 110])
upper_red = np.array([1, 255, 255])
# Loop through the frames in the video
while cap.isOpened():
    # Read the frame and convert it to HSV color space
    ret, frame = cap.read()
    if ret:
        height,width,_ = frame.shape
        #print(height)
        # roi = frame[30:height, 50:width-55]
        # Apply Gaussian blur to remove noise
        # blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(cv2.rotate(frame, cv2.ROTATE_180), cv2.COLOR_BGR2HSV)
        # Threshold the image to extract the red color
        mask = cv2.inRange(hsv, lower_red, upper_red)

        avg_intensity = np.mean(mask)
    
        # find the x,y coordinates of the maximum value in the grayscale image
        max_location = np.where(mask == np.max(mask))
        x, y = max_location[1][0], max_location[0][0]
        # display the x,y coordinates and the average pixel intensity
        #print(f"x={x}, y={y}, avg_intensity={avg_intensity}")
        if x and y == 0:
            break
        else:
            x_co_ordinates.append(x)
            y_co_ordinates.append(y)
        
        # Display the frame
        cv2.imshow('frame', mask)
        
        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
# Release the video capture and close the window
plt.plot(x_co_ordinates, y_co_ordinates,'ro')
plt.xlim(1200.0,2.0)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of X and Y')
plt.show()
x=np.array(x_co_ordinates)
y=np.array(y_co_ordinates)
x = x[np.nonzero(x)]
y = y[np.nonzero(y)]
################################### Second Part ############################################################
A = np.vstack([x**2, x, np.ones(len(x))]).T
# create the observation matrix b
b =y
# compute the least squares solution
x_ls = np.linalg.inv(A.T @ A) @ A.T @ b
print("Equation of the Curve: y = {:.6f}x^2 + {:.6f}x {:.6f}".format(x_ls[0], x_ls[1],x_ls[2]))
# plot the data and the fitted curve
plt.scatter(x, y)
plt.xlim(1200.0,1.0)
plt.plot(x, A @ x_ls, color='red')
plt.show()
################################## Third Part ############################################################
x_ls[2]=x_ls[2]-y[0]-300+421
discriminant = x_ls[1]**2 - 4*x_ls[0]*x_ls[2]

# check if roots are real or imaginary
if discriminant < 0:
    print("Roots are imaginary")
else:
    # calculate the roots
    root1 = (-x_ls[1] + np.sqrt(discriminant)) / (2*x_ls[0]) #discarded as the value is negative
    root2 = (-x_ls[1] - np.sqrt(discriminant)) / (2*x_ls[0])
    print("Obtained value of x-coordinate of landing point of the ball is:" )
    print(root2)