from utils.camera.kinect import KinectCamera
import cv2
import numpy as np
import time
from numpy.linalg import inv
left_intrinsic_parameters = np.array([[1.06459949e+03, 0.00000000e+00,   5.32365337e+02],
                                      [0.00000000e+00,   1.06636785e+03,   9.64761209e+02],
                                      [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
right_intrinsic_parameters = np.array([[1.06722499e+03, 0.00000000e+00, 5.47230766e+02],
                                      [0.00000000e+00, 1.06913729e+03, 9.53933309e+02],
                                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


transformation_matrix = np.array([  [-0.72296534, -0.27458619,  0.6339744, -655.97574223],
                                    [ 0.17913411,  0.81174473,  0.55586101, -671.33499921],
                                    [-0.66725714,  0.51543468, -0.53767556, 1783.01803744],
                                    [0, 0, 0, 1]])
# transformation_matrix = np.array([[-0.7294265,  -0.28553025,  0.62161842, -645.42868139],
#                                   [ 0.15763156,  0.81410438,  0.55891534, -658.04920594],
#                                   [-0.66564951,  0.50567434, -0.54882072, 1793.15555287],
#                                   [0, 0, 0, 1]])



transformation_matrix =inv(transformation_matrix)

def get_intrinsic_parameters(camera):
    if camera == 0:
        fx = left_intrinsic_parameters[0,0]
        fy = left_intrinsic_parameters[1,1]
        cx = left_intrinsic_parameters[0,2]
        cy = left_intrinsic_parameters[1,2]

    if camera == 1:
        fx = right_intrinsic_parameters[0,0]
        fy = right_intrinsic_parameters[1,1]
        cx = right_intrinsic_parameters[0,2]
        cy = right_intrinsic_parameters[1,2]
    return fx, fy, cx, cy

left = KinectCamera(0)
right = KinectCamera(1)

def get_world_coordinates(rgb_img, depth_img, camera):
    fx, fy, cx, cy= get_intrinsic_parameters(camera)
    world_coord = np.full([depth_img.shape[0], depth_img.shape[1], 3], np.nan)
    #world_img = np.full([4550,4550,3], 225)

    for i in range(depth_img.shape[0]):

        for j in range(depth_img.shape[1]):

            if not np.isnan(depth_img[i,j]) and not np.isinf(depth_img[i,j]):

                for k in range(3):

                    if k == 0:

                        world_coord[i, j, k] = int((i - cx) * depth_img[i,j] / fx)
                        #-2151 to 1346 -> 3497
                    if k == 1:
                        world_coord[i, j, k] = int((j - cy) * depth_img[i, j] / fy)

                        #-1523 to 1887 -> 3410
                    if k == 2:

                        world_coord[i, j, k] = depth_img[i, j]

                #print(rgb_img[i, j, 0])
                # world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 0] = rgb_img[i, j, 0]
                # world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 1] = rgb_img[i, j, 1]
                # world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 2] = rgb_img[i, j, 2]

                    #print(k, world_coord[i,j,k])

    return world_coord

point_clouds = []
#time.sleep(8)
left_img, left_depth = left.get_frames()  #did not flip.leftright



left_depth = left_depth[1:1081,:]

left_world_coordinates= get_world_coordinates(left_img, left_depth, 0)

# world_pic = cv2.resize(world_pic.astype(np.uint8),(1080,1080))


for i in range(left_depth.shape[0]):

    for j in range(left_depth.shape[1]):
        if not np.isnan(left_world_coordinates[i, j, 0]):

            homogenous_coordinate = np.array([left_world_coordinates[i,j, 1], left_world_coordinates[i,j, 0], left_world_coordinates[i,j, 2]])
            point_clouds.append(" ".join(str(a) for a in homogenous_coordinate)) # xyz
            point_clouds.append(" " + " ".join(str(e) for e in left_img[i, j]) + '\n')  # color

right_img, right_depth = right.get_frames()
right_depth = right_depth[1:1081, :]
right_world_coordinates = get_world_coordinates(right_img, right_depth, 1)

for i in range(right_depth.shape[0]):
    for j in range(right_depth.shape[1]):
        if not np.isnan(right_world_coordinates[i, j,0]):
            homogenous_coordinate = np.array([right_world_coordinates[i, j, 1], right_world_coordinates[i, j, 0], right_world_coordinates[i, j, 2],1])
            #homogenous_coordinate=np.append(right_world_coordinates[i,j],1)
            transformed_coordinate = np.matmul(transformation_matrix, homogenous_coordinate)
            point_clouds.append(" ".join(str(a) for a in transformed_coordinate[0:3])) #xyz
            point_clouds.append(" "+" ".join(str(e) for e in right_img[i, j])+'\n')  # color



#create transformed file
def write_ply():
    with open('shopper_synchronized_kinect.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n'%(len(point_clouds)/2))
        f.write('property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar blue\n'
                'property uchar green\n'
                'property uchar red\n'
                'end_header\n')
        for line in point_clouds:

            f.write(line)

print('writing ply to file')
write_ply()


left.__del__()
right.__del__()