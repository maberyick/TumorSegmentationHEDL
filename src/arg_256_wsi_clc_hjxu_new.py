#coding=utf-8
import argparse

caffe_root = '/home/sgu/git/caffe/' # 需要修改成caffe的路径
import sys
sys.path.insert(0, caffe_root + 'python')
import glob
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import os
import scipy
import cv2
import skimage
from skimage.filters import threshold_otsu, try_all_threshold, threshold_mean

import matplotlib.pyplot as plt
from scipy import misc
caffe.set_mode_gpu()
caffe.set_device(0)

start=time.time()

deploy_path = "./deploy_vgg16.prototxt"
model_path = "./vgg_iter_100000.caffemodel"
mean_proto_path = "./train_mean.binaryproto"

###========================================
'''
one wsi classify
'''
WSI = "/home/hjxu_disk/oral/data/CCF.tiff"  # 需要跑的图像的路径
path_of_heat_to_save = "/home/hjxu_disk/oral/pred/evaluatation/Pred2/CCF_16_ooo.jpg"  # 结果需要保存的路径,每跑一张图就需要修改

#==================
'''
batch classify wsi
'''
TUMOR_WSI_PATH = "/media/sgu/文档/oral_ccipd_hjxu/data/"
HEAT_MAP_SAVE_PATH = "/media/sgu/文档/oral_ccipd_hjxu/pred/"
num1 = 0 #批量测试时，如果文件夹内有100张，num1-num2代表 测试第0张到第20张
num2 = 40

# ======================================================##

##
'''
caffe model
'''
deploy = deploy_path
model = model_path
mean_proto_path = mean_proto_path

blob = caffe.proto.caffe_pb2.BlobProto()
data_mean = open(mean_proto_path, 'rb').read()
blob.ParseFromString(data_mean)
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean_npy.mean(1).mean(1))
#   transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
####===================================================================================


# ================================================
def get_arguments():
    parser = argparse.ArgumentParser(description='simple nodule segmentation')
    parser.add_argument('--deploy',type = str,default=deploy_path,help='path of caffe-deploy')
    parser.add_argument('--model',type = str,default=model_path,help='path of caffe-model')
    parser.add_argument('--mean_proto_path',type=str,default=mean_proto_path,help='path of caffe-mean.binaryproto')
    parser.add_argument('--WSI', type=str, default=WSI, help='path of whole slide image + name')
    parser.add_argument('--heatmap_saved', type=str, default=path_of_heat_to_save, help='path of heatmap-img + name')
    parser.add_argument('--num1', type=int, default=num1)
    parser.add_argument('--num2', type=int, default=num2)
    parser.add_argument('--TUMOR_WSI_PATH', type=str, default=TUMOR_WSI_PATH, help='folder of wsi')
    parser.add_argument('--HEAT_MAP_SAVE_PATH', type=str, default=HEAT_MAP_SAVE_PATH, help='the folder where you need to save the heatmap ')
    return parser.parse_args()


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def get_bbox(cont_img, rgb_image=None):
    # _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour


def read_wsi_tumor(wsi_path):
    try:
        wsi_image = OpenSlide(wsi_path)
        w, h = wsi_image.dimensions
        w, h = int(w / 256), int(h / 256)
        level_used = wsi_image.level_count - 1
        if (level_used >= 8):
            level_used = 8
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       (w,h)))
        else:
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            rgb_image = cv2.resize(rgb_image, (w, h))
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return wsi_image, rgb_image, level_used,w,h

def find_roi_bbox(rgb_image):
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([20, 20, 20])
    # upper_red = np.array([200, 200, 200])
    # # mask -> 1 channel
    # mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0
    thres = threshold_mean(hsv[..., 0])
    # fig, ax = try_all_threshold(hsv[..., 0])
    # mask = (hsv[..., 0] > thres).astype('uint8')
    _, mask = cv2.threshold(hsv[..., 0], thres, 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    # plt.imshow(image_open)
    # plt.show()
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open

def test(WSI_path,save_path):

    # wsi_name = get_filename_from_path(WSI_path)
    wsi_image, rgb_image, level,m,n=read_wsi_tumor(WSI_path)
    # image_heat_save_path = save_path + wsi_name + '_heatmap.jpg'
    bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
    image_heat_save = np.zeros((n, m))


    print('%s Classification is in progress' % WSI_path)
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        #        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
        #        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
        col_cords = np.arange(b_x_start, b_x_end)
        row_cords = np.arange(b_y_start, b_y_end)
        mag_factor = 256
        #        for x, y in zip(X, Y):
        #            if int(tumor_gt_mask[y, x]) != 0:
        for x in col_cords:
            for y in row_cords:
                if int(image_open[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (256, 256))
                    img_tmp = skimage.img_as_float(np.array(patch))
                    img1 = np.tile(img_tmp, (1, 1, 3))
                    img2 = img1[:, :, :3]
                    net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                    out = net.forward()
                    prob = out['prob'][0][0]
                    # print y, x
                    image_heat_save[y, x] = prob
    print save_path,'in save...'
    scipy.misc.imsave(save_path, image_heat_save)

def test_one_wsi():
# '''
# 仅测试一张wsi
# '''


    start=time.time()

    args = get_arguments()

    WSI = args.WSI
    saved_path = args.heatmap_saved


    wsi_image, rgb_image, level,m,n=read_wsi_tumor(WSI)
    plt.imshow(rgb_image)
    plt.show()
    # # rgb_image = cv2.resize(rgb_image, (cols/2, rows/2), interpolation=cv2.INTER_AREA)
    bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
    plt.imshow(image_open)
    plt.show()
    # image_heat_save = np.zeros((n+1, m+1))
    image_heat_save = np.zeros((n, m))

    print('%s Classification is in progress' % WSI)
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        # print b_x_start, b_x_end
        # print b_y_start, b_y_end
        col_cords = np.arange(b_x_start, b_x_end)
        row_cords = np.arange(b_y_start, b_y_end)
        mag_factor = 256
        for x in col_cords:
            for y in row_cords:
                # print y,x
                if int(image_open[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (256, 256))
                    img_tmp = skimage.img_as_float(np.array(patch))
                    img1 = np.tile(img_tmp, (1, 1, 3))
                    img2 = img1[:, :, :3]
                    net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                    out = net.forward()
                    prob = out['prob'][0][0]
                    # print prob
                    image_heat_save[y, x] = prob

    scipy.misc.imsave(saved_path, image_heat_save)
    end = time.time()
    print ('run time%s'%(end-start))
    print('has done...')


def test_batch_wsi():

# '''
# 测试文件夹内的图片
# '''
    args = get_arguments()
    # TUMOR_WSI_PATH = '/home/hjxu_disk/orcal/data/'
    # HEAT_MAP_SAVE_PATH = '/home/hjxu_disk/orcal/p/'
    TUMOR_WSI_PATH = args.TUMOR_WSI_PATH
    HEAT_MAP_SAVE_PATH = args.HEAT_MAP_SAVE_PATH
    # ===============================================
    wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH, '*.tiff'))
    wsi_paths.sort()
    WSI_path = list(wsi_paths)
    # print len(WSI_path)
    WSI_path1 = wsi_paths[args.num1: args.num2]
    for WSI_NAME in WSI_path1:
        wsi_name = get_filename_from_path(WSI_NAME)
        heat_map_save_path = HEAT_MAP_SAVE_PATH + wsi_name + '_heatmap.jpg'
        if os.path.exists(heat_map_save_path):
            print(heat_map_save_path,"has created, please check, ERROR!!!!!!!!!")
            continue
        test(WSI_NAME, heat_map_save_path)

if __name__ == "__main__":
    # test_one_wsi()
    test_batch_wsi()
