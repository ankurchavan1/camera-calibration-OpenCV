import numpy as np
from scipy.linalg import rq

# Given Image Points
image_points = [(757,213),(758,415),(758,686),(759,966),(1190,172),(329,1041),(1204,850),(340,159)]

# Given World Points
world_points = [(0,0,0),(0,3,0),(0,7,0),(0,11,0),(7,1,0),(0,11,7),(7,9,0),(0,1,7)]

#A Matrix
M_matrix = []
for i in range(0, len(image_points)):
    ui,vi = image_points[i]
    x_world,y_world,z_world = world_points[i]
    M_matrix.append([0, 0, 0, 0, -x_world, -y_world, -z_world, -1, vi*x_world, vi*y_world, vi*z_world, vi])
    M_matrix.append([x_world, y_world, z_world, 1, 0, 0, 0, 0, -ui*x_world, -ui*y_world, -ui*z_world, -ui]) 

M_matrix = np.asarray(M_matrix)

print(M_matrix.shape)

# Projection matrix using SVD
U,S,Vt = np.linalg.svd(M_matrix)

P = Vt[-1,:]  

# Final Projection Matrix
Projection_matrix = np.reshape(P,(3,4))
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Projection Matrix from World Frame to Image Frame: ")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(Projection_matrix)

# Left 3 Columns of Projection Matrix for RQ factorization
Proj_mat_left_3_columns = []
Proj_mat_left_3_columns = Projection_matrix[:, :3]

#RQ factorization
Intrinsic_mat, Rotation_mat = rq(Proj_mat_left_3_columns)

# Normalizing
Intrinsic_mat = Intrinsic_mat / Intrinsic_mat[2,2]

Intrinsic_mat_inverse = np.linalg.inv(Intrinsic_mat)

# Extrinsic Matrix
Extrinsic_mat = Intrinsic_mat_inverse @ Projection_matrix

# Translation Vector
Translation = Extrinsic_mat[:, 3]

#Printing the K and R matrices
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Intrinsic Matrix:")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(Intrinsic_mat)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Rotation Matrix:")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(Rotation_mat)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Translation Vector:")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(Translation)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("The reprojection error for individual Points")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
num_points = len(world_points)
reprojection_errors = np.zeros(num_points)

# Loop over each point
for i in range(num_points):
    # Convert world point to homogenous coordinates
    homogenous_world_point = np.hstack((world_points[i], 1))
    
    # Project world point onto image plane
    homogenous_image_point = np.dot(Projection_matrix, homogenous_world_point)
    
    # Convert to cartesian coordinates and normalize
    new_image_point = homogenous_image_point[:2] / homogenous_image_point[2]
    
    # Calculate reprojection error as the Euclidean distance between projected and actual image point
    dx, dy = image_points[i] - new_image_point 
    reprojection_errors[i] = np.sqrt(dx**2 + dy**2)
    
    # Print error for each point
    print(f"The reprojection error for point {i+1} is {reprojection_errors[i]}")
    print("-------------------------------------------------------------------------------------------------------------------------------------------------")

# Calculate mean reprojection error
mean_reprojection_error = np.mean(reprojection_errors)
print(f"The mean reprojection error is {mean_reprojection_error}")
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
