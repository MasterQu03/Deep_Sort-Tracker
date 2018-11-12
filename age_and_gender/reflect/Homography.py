import cv2
import numpy as np
import pylab as pl
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random


camera_matrix = []
newcameramtx = []
dist_coefs = []
h = []
num = 0

def load_matrix():
    camera_matrix.clear()
    newcameramtx.clear()
    dist_coefs.clear()
    h.clear()

    for i in range(1, 5):
        if i==2:
            camera_matrix.append([])
            newcameramtx.append([])
            dist_coefs.append([])
            h.append([])
            continue

        cam_no = 'c' + str(i) + '_'

        camera_matrix.append(np.loadtxt("./reflect/" + cam_no + "camera_matrix.txt", dtype='float32'))
        newcameramtx.append(np.loadtxt("./reflect/" + cam_no + "newcameramtx.txt", dtype='float32'))
        dist_coefs.append(np.loadtxt("./reflect/" + cam_no + "dist_coefs.txt", dtype='float32'))
        h.append(np.loadtxt("./reflect/" + cam_no + "homography.txt", dtype='float32'))

def undistortPoint(x, y, camera_matrix, dist_coefs, newcameramtx):
    src = np.zeros(shape=(1, 1, 2))
    src[0][0][0] = x
    src[0][0][1] = y
    undistort_point = cv2.undistortPoints(src, camera_matrix, dist_coefs, P=newcameramtx)
    return undistort_point[0][0][0], undistort_point[0][0][1]


############################################################################
#manufactured test
def manufactured_test():
    plt.subplot(1,2,1)
    plt.imshow(cam)
    plt.subplot(1,2,2)
    plt.imshow(para)

    def on_press(event):
        print("you pressed" ,event.button, event.xdata, event.ydata)

        point = hit_pos(event.xdata, event.ydata, num)
        print(point[0][0], point[1][0])

        ax = plt.gca()
        plt.plot(point[0][0], point[1][0], marker='.')
        fig.canvas.draw()
        # plt.show()


    fig.canvas.mpl_connect('button_press_event', on_press)


def manufactured_without_homograph_test():
    plt.subplot(1,2,1)
    plt.imshow(cam)
    plt.subplot(1,2,2)
    img = Image.open('Q' + str(num) + '.jpg_undistorted.png')
    plt.imshow(img)

    def on_press(event):
        print("you pressed" ,event.button, event.xdata, event.ydata)

        x, y = undistortPoint(event.xdata, event.ydata, camera_matrix, dist_coefs, newcameramtx)
        print(x, y)

        ax = plt.gca()
        plt.plot(x, y, marker='.')
        fig.canvas.draw()
        # plt.show()


    fig.canvas.mpl_connect('button_press_event', on_press)


def trans_pos(x, y, cam_no):
    world_pos = np.matrix([x, y, 1])
    world_pos = world_pos.reshape(-1,1)

    # print(world_pos)
    # print((camera_matrix*RT).I)
    point = h[cam_no]*world_pos
    return point, world_pos

def hit_pos(xdata, ydata, cam_no):
    if len(camera_matrix)==0:
        load_matrix()
    if cam_no > 1:
        cam_no += 1
    cam_no = int(cam_no)
    cam_no -= 1
    x, y = undistortPoint(xdata, ydata, camera_matrix[cam_no], dist_coefs[cam_no], newcameramtx[cam_no])
    point, world_pos = trans_pos(x, y, cam_no)

    # point, world_pos = trans_pos(event.xdata, event.ydata)
    point /= point[2][0]
    return point[0][0], point[1][0]

def load_data():
    data = {}
    with open('hypotheses(1).txt') as f:
        for line in f:
            line = line.split(',')

            frame = int(line[0])
            id = int(line[1])
            x = int(float(line[2])+float(line[4])/2)
            y = int(float(line[3])+float(line[5]))

            if int(line[0]) not in data:
                data[frame] = [(id,x,y)]
            else:
                data[frame].append((id,x,y))
    return data
def show_data():
    data = load_data()
    plt.ion()

    # 循环
    for frame, val in data.items():
        #清除原有图像
        plt.cla()

        plt.subplot(1, 2, 2)
        plt.imshow(para)

        if frame%5!=0:
            continue;

        for v in val:

            point = hit_pos(v[1], v[2])

            if 1:#point[0][0]>=0 and point[0][0]<=1000 and point[1][0]>=0 and point[1][0]<=1000:
                plt.plot(point[0][0], point[1][0], marker='.')
                fig.canvas.draw()
            else:
                print(point)

        # 暂停
        plt.pause(0.15)

    # 关闭交互模式
    plt.ioff()


if __name__ == '__main__':

    num = input('cam number: ')
    cam_no = 'c' + str(num) + '_'

    h = []
    cam = Image.open('Q' + str(num) + '.jpg')
    w, h = cam.size
    para = Image.open('map.jpg')

    camera_matrix = np.loadtxt("./" + cam_no + "camera_matrix.txt", dtype='float32')
    newcameramtx = np.loadtxt("./" + cam_no + "newcameramtx.txt", dtype='float32')
    dist_coefs = np.loadtxt("./" + cam_no + "dist_coefs.txt", dtype='float32')

    # Read source image.
    im_src = cv2.imread('Q'+str(num)+'.jpg')
    # Four corners of the book in frame_32.jpgsource image
    # pts_src = np.array([[167.0, 264.0], [482.0, 798.0], [1079.0, 403.0], [613.0, 84.0]])
    pts_src = np.loadtxt('c' + str(num) + '_2d_point.txt', dtype='float32')

    h, w = im_src.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    im_src = cv2.undistort(im_src, camera_matrix, dist_coefs, None, newcameramtx)

    for i in range(len(pts_src)):
        x, y = undistortPoint(pts_src[i][0], pts_src[i][1], camera_matrix, dist_coefs, newcameramtx)
        pts_src[i][0] = x
        pts_src[i][1] = y

    # Read destination image.
    im_dst = cv2.imread('Q'+str(num)+'.jpg_undistorted.png')
    # Four corners of the book in destination image.
    # pts_dst = np.array([[193.0, 742.0], [996.0, 874.0], [1059.0, 157.0], [266.0, 145.0]])
    pts_dst = np.loadtxt('c' + str(num) + '_3d_point.txt', dtype='float32')

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.LMEDS) # method or 0 or RANSAC
    np.savetxt(cam_no + 'homography.txt', h, fmt='%3.8f', delimiter=' ', newline='\n')

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    im = Image.fromarray(im_out[:, :, ::-1])
    im.save(cam_no + "Homography.jpg")
    para = Image.open('map.jpg')

    # pl.figure(), pl.imshow(im_src[:, :, ::-1]), pl.title('src'),
    # pl.figure(), pl.imshow(im_dst[:, :, ::-1]), pl.title('dst')
    # pl.figure(), pl.imshow(im_out[:, :, ::-1]), pl.title('out'), pl.show()  # show dst

    fig = plt.figure()
    load_matrix()
    manufactured_test()
    # manufactured_without_homograph_test()

    # show_data()

    plt.show()