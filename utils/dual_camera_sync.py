from utils.camera.kinect import KinectCamera
import cv2
import numpy as np
import time
left_intrinsic_parameters = np.array([[1.06459949e+03, 0.00000000e+00,   5.32365337e+02],
                                      [0.00000000e+00,   1.06636785e+03,   9.64761209e+02],
                                      [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])


def get_intrinsic_parameters(camera):
    if camera ==0:
        left_fx = left_intrinsic_parameters[0,0]
        left_fy = left_intrinsic_parameters[1,1]
        left_cx = left_intrinsic_parameters[0,2]
        left_cy = left_intrinsic_parameters[1,2]
        return left_fx, left_fy, left_cx, left_cy

left = KinectCamera(0)
right =KinectCamera(1)

def get_world_coordinates(rgb_img, depth_img, camera):
    fx, fy, cx, cy= get_intrinsic_parameters(camera)
    world_coord = np.full([depth_img.shape[0], depth_img.shape[1], 3], np.nan)
    world_img = np.full([3600,3600,3], 225)
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if not np.isnan(depth_img[i,j]) and not np.isinf(depth_img[i,j]):
                for k in range(3):
                    if k == 0:

                        world_coord[i, j, k] = int((i - cy) * depth_img[i,j] / fy)
                        #-2151 to 1346 -> 3497
                    if k == 1:
                        world_coord[i, j, k] = int((j - cx) * depth_img[i, j] / fx)

                        #-1523 to 1887 -> 3410
                    if k == 2:

                        world_coord[i, j, k] = depth_img[i, j]

                #print(rgb_img[i, j, 0])
                world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 0] = rgb_img[i, j, 0]
                world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 1] = rgb_img[i, j, 1]
                world_img[int(world_coord[i, j, 0] + 2151), int(world_coord[i, j, 1] + 1523), 2] = rgb_img[i, j, 2]

                    #print(k, world_coord[i,j,k])

    return world_coord, world_img

while True:
    time.sleep(5)
    left_img = left.get_frames()[0]
    left_depth = left.get_frames()[1][:,1:1081]



    left_world_coordinates, world_pic = get_world_coordinates(left_img, left_depth, 0)

    world_pic = cv2.resize(world_pic.astype(np.uint8),(1080,1080))

    cv2.imshow('left img', world_pic)
    cv2.imwrite('empty_point_cloud.png',world_pic)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

left.__del__()
right.__del__()