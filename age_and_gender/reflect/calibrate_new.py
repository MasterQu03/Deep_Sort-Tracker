import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

square_size = 1
img_path = './Q4.jpg'
json_path = './Q4.json'
cam_no = 'c4_'

def load(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data


if __name__ == '__main__':

    obj_points = []
    img_points = []

    cali_data = load(json_path)['shapes']

    l = 0

    corners = []
    for data in cali_data:

        if int(data['label'])==l:
            corners.append([float(data['points'][0][0]), float(data['points'][0][1])])

        else:
            l = int(data['label'])
            img_points.append(np.array(corners, dtype='float32'))
            point_num = len(corners)

            pattern_size = (3, int(point_num/3))
            pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
            pattern_points *= square_size

            obj_points.append(pattern_points)

            corners = [[float(data['points'][0][0]), float(data['points'][0][1])]]

    point_num = len(corners)
    img_points.append(np.array(corners, dtype='float32'))
    pattern_size = (3, int(point_num/3))
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points.append(pattern_points)

    for i in range(len(img_points)):
        if len(img_points[i])!=len(obj_points[i]):
            print(len(img_points[i]), len(obj_points[i]), i)


    img_names_undistort = []
    img_names_undistort.append(img_path)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        nr_points = 1
        src = np.zeros(shape=(nr_points, 1, 2))
        src[0][0][0] = 248
        src[0][0][1] = 665
        undistort_point = cv2.undistortPoints(src, camera_matrix, dist_coefs, P=newcameramtx)
        print(undistort_point)
        print(roi)

        # crop and save the image
        x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        outfile = img_found + '_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

        img = Image.open(outfile)
        plt.imshow(img)
        plt.plot(undistort_point[0][0][0], undistort_point[0][0][1], marker='.')
        # plt.plot(undistort_point[0][0][0]*camera_matrix[0][0]+camera_matrix[0][2], undistort_point[0][0][1]*camera_matrix[1][1]+camera_matrix[1][2], marker='.')
        print(undistort_point[0][0][0]*camera_matrix[0][0]+camera_matrix[0][2], undistort_point[0][0][1]*camera_matrix[1][1]+camera_matrix[1][2])
        plt.show()
        # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, newcameramtx, (w, h), 5)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


    np.savetxt(cam_no + 'camera_matrix.txt', camera_matrix, fmt='%3.8f', delimiter=' ', newline='\n')
    np.savetxt(cam_no + 'newcameramtx.txt', newcameramtx, fmt='%3.8f', delimiter=' ', newline='\n')
    np.savetxt(cam_no + 'dist_coefs.txt', dist_coefs, fmt='%3.8f', delimiter=' ', newline='\n')

    cv2.destroyAllWindows()
