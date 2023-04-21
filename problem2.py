import numpy as np
import cv2 as cv
import os

#  trmination criteria to specify the maximum number of iterations and a minimum value for convergence.
term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 35, 0.00099)

# array of object points that correspond to the location of the corners of a chessboard calibration pattern in 3D space
object_points = np.zeros((9*6,3), np.float32)
object_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# empty lists to store the object points and image points
object_points_list = []                                      
image_points_list = []                                       

images_path = "C:/Users/ankur/OneDrive/Desktop/M.Eng Robotics/Spring 2023/Perception/Projects/Project 3/Calibration_Imgs"
images_for_3D_pts = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')]

for each_image in images_for_3D_pts:

    # Read the image
    image = cv.imread(each_image)
    
    # Converting to greyscale image
    greyscale_iamge = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, chess_square_corners = cv.findChessboardCorners(greyscale_iamge, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        # Adding object points to the list
        object_points_list.append(object_points)

        # Refining the corners
        fianl_corners = cv.cornerSubPix(greyscale_iamge,chess_square_corners, (13,13), (3,3), term_criteria)
        
        # Adding image points to the list
        image_points_list.append(fianl_corners)

        # Draw and display the corners
        cv.drawChessboardCorners(image, (9,6), fianl_corners, ret)

        # Create window and display image
        cv.namedWindow("image", cv.WINDOW_NORMAL)

        # Resize window 
        scale_percent = 30
        width = int(image.shape[1] * (scale_percent / 100))
        height = int(image.shape[0] * (scale_percent / 100))
        cv.resizeWindow("image", width, height)
        cv.imshow('image', image)
        cv.waitKey(500)

    else:
        break
cv.destroyAllWindows()

# obtain the camera matrix, distortion coefficients, and rotation & translation vectors
ret, K_matrix, distortion, rotation_vec, translation_vec = cv.calibrateCamera(object_points_list, image_points_list, greyscale_iamge.shape[::-1], None, None)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("K Matrix:")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(K_matrix)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Reprojection error for individual Images:")

n = 0
total_error = 0
for i in range(len(object_points_list)):
    final_image_points, _ = cv.projectPoints(object_points_list[i], rotation_vec[i], translation_vec[i], K_matrix, distortion)
    error = np.sqrt(np.sum((image_points_list[i] - final_image_points)**2))/len(final_image_points)
    n += 1
    print("-------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Reprojection error for image ",n, "is ", error)
    total_error += error

mean_error = total_error/len(object_points_list)    
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print( "Mean Reprojection error: ", mean_error )
print("-------------------------------------------------------------------------------------------------------------------------------------------------")