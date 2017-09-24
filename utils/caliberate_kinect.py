import numpy as np
import cv2
import glob
from numpy.linalg import inv
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
def caliberate_kinect(camera):
    h = 9
    w = 6
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = sorted(glob.glob('camera/%s/*.png'%camera))
    print(images)
    for fname in images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h,w),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (h,w), corners2,ret)
            # cv2.imshow(fname, cv2.resize(img, (int(1080/1.8), int(1920/1.8))))
            # cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return (mtx, dist, rvecs, tvecs)
l = caliberate_kinect('left')
r = caliberate_kinect('right')

rotation_vectors_left = l[2]
rotation_vectors_right = r[2]

translation_vectors_left = l[3]
translation_vectors_right = r[3]

for ind in range(len(translation_vectors_left)):

    left_rotation_matrix = cv2.Rodrigues(rotation_vectors_left[ind])[0]
    stacked_left = np.vstack((np.column_stack((left_rotation_matrix, translation_vectors_left[ind])),np.array([0,0,0,1])))

    right_rotation_matrix = cv2.Rodrigues(rotation_vectors_right[ind])[0]
    stacked_right = np.vstack((np.column_stack((right_rotation_matrix, translation_vectors_right[ind])),np.array([0,0,0,1])))


    print(np.matmul(stacked_left, inv(stacked_right)))

# print(cv2.Rodrigues(l[2][0])[0])


# ['camera/right/dual_right_10.png', 'camera/right/dual_right_13.png', 'camera/right/dual_right_14.png', 'camera/right/dual_right_15.png', 'camera/right/dual_right_16.png', 'camera/right/dual_right_17.png', 'camera/right/dual_right_18.png', 'camera/right/dual_right_19.png', 'camera/right/dual_right_20.png', 'camera/right/dual_right_21.png', 'camera/right/dual_right_22.png', 'camera/right/dual_right_23.png', 'camera/right/dual_right_24.png', 'camera/right/dual_right_25.png', 'camera/right/dual_right_27.png', 'camera/right/dual_right_28.png', 'camera/right/dual_right_29.png', 'camera/right/dual_right_30.png', 'camera/right/dual_right_4.png', 'camera/right/dual_right_5.png', 'camera/right/dual_right_6.png', 'camera/right/dual_right_7.png', 'camera/right/dual_right_8.png', 'camera/right/dual_right_9.png']
# ['camera/left/dual_left_10.png', 'camera/left/dual_left_13.png', 'camera/left/dual_left_14.png', 'camera/left/dual_left_15.png', 'camera/left/dual_left_16.png', 'camera/left/dual_left_17.png', 'camera/left/dual_left_18.png', 'camera/left/dual_left_19.png', 'camera/left/dual_left_20.png', 'camera/left/dual_left_21.png', 'camera/left/dual_left_22.png', 'camera/left/dual_left_23.png', 'camera/left/dual_left_24.png', 'camera/left/dual_left_25.png', 'camera/left/dual_left_27.png', 'camera/left/dual_left_28.png', 'camera/left/dual_left_29.png', 'camera/left/dual_left_30.png', 'camera/left/dual_left_4.png', 'camera/left/dual_left_5.png', 'camera/left/dual_left_6.png', 'camera/left/dual_left_7.png', 'camera/left/dual_left_8.png', 'camera/left/dual_left_9.png']
#dual_left
# [[  1.06459949e+03   0.00000000e+00   5.32365337e+02]
#  [  0.00000000e+00   1.06636785e+03   9.64761209e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

#dual_right
# [[  1.06722499e+03   0.00000000e+00   5.47230766e+02]
#  [  0.00000000e+00   1.06913729e+03   9.53933309e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]


# right exp 1
# [[  1.06843449e+03   0.00000000e+00   5.41203824e+02]
#  [  0.00000000e+00   1.06658844e+03   9.53953541e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
# right exp 2
# [[  1.07030342e+03   0.00000000e+00   5.49779380e+02]
#  [  0.00000000e+00   1.06959044e+03   9.55375064e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
# right exp 3
# [[  1.06751867e+03   0.00000000e+00   5.44552358e+02]
#  [  0.00000000e+00   1.06763614e+03   9.39975641e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

#left exp1
# [[  1.06328409e+03   0.00000000e+00   5.26410218e+02]
#  [  0.00000000e+00   1.06339060e+03   9.68641021e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]