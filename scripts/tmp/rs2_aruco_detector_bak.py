#!/usr/bin/env python

'''
Author: Dianye Huang
Date: 2022-08-08 14:26:11
LastEditors: Dianye Huang
LastEditTime: 2023-01-17 08:08:25
Description: 
'''
# ros
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import tf
import tf.transformations as t

# opencv2
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class Rs2ArucoDetector:
    def __init__(self, 
                flag_run_alone = False,
                res_img_topic_name = 'rs2_aruco_detector/res_img'):
        if flag_run_alone:
            rospy.init_node('rs2_aruco_detector_node', anonymous=True)
        # camera's inner parameters -> realsense2
        self.mtx = np.array([[615.7256469726562, 0.0, 323.13262939453125], # TODO: to be replaced by the self._intrinsics params
                            [0.0, 616.17236328125, 237.86715698242188], 
                            [0.0, 0.0, 1.0]])
        self.dist =np.array([ 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]) # TODO: the same
        self._intrinsics = rs.intrinsics()
        
        # images subscription
        self.flag_detecting_aruco = False
        self.bridgeC = CvBridge()
        self.cinfo_sub = rospy.Subscriber("/camera/color/camera_info", 
                                            CameraInfo, self.cinfo_sub_cb)                 # intern parameters
        rospy.wait_for_message("/camera/color/camera_info", CameraInfo)    
        self.dimg_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",
                                                    Image , queue_size=20)                 # depth images
        self.cimg_sub = message_filters.Subscriber("/camera/color/image_raw", 
                                                    Image, queue_size=20, buff_size=2**24) # color images
        self.ts = message_filters.TimeSynchronizer([self.cimg_sub, self.dimg_sub], 1)
        self.ts.registerCallback(self.sub_imgs_cb)
        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        rospy.wait_for_message("/camera/color/image_raw", Image)

        # publish detected result img
        self.aruco_detect_res = dict()
        self.pub_res_img = rospy.Publisher(res_img_topic_name, Image, queue_size=2)

        # detect aruco 
        self.img_updataed = False
        self.marker_length = 0.028 # mind the setting
        self.rate = rospy.Rate(30)
    
    def set_aruco_length(self, val:float):
        self.marker_length = val

    def get_aruco_detect_res(self):
        if self.img_updataed:
            self.flag_detecting_aruco = True
            res_img, detect_res = self.detect_aruco(self.cv2_color_img)
            self.aruco_detect_res = dict()
            for id in detect_res.keys():
                ccorner = detect_res[id]['ccorner']
                dimg_arr = np.array(self.cv2_depth_img, dtype=np.float32)
                # tvec = detect_res[id]['tvec'] # or use translations from aruco, where marker length matters
                tvec = self.dimg_2_tras_vec(ccorner, dimg_arr) # tx, ty, tz, using depth info could avoid setting marker length
                qvec = self.rvec2quat(detect_res[id]['rvec'])  # qx, qy, qz, qw
                self.aruco_detect_res[id] = (tvec, list(qvec))
            self.flag_detecting_aruco = False
            self.img_updataed = False
            msg_img = self.bridgeC.cv2_to_imgmsg(res_img)
            self.pub_res_img.publish(msg_img)
        return self.aruco_detect_res

    def dimg_2_tras_vec(self, ccorner, dimg_arr):
        pixel2depth_marker = rs.rs2_deproject_pixel_to_point(self._intrinsics, [int(ccorner[0][0]), 
                                                int(ccorner[0][1])], dimg_arr[int(ccorner[0][1]),
                                                int(ccorner[0][0])])
        pixel2depth_marker[0] = float(pixel2depth_marker[0])/1000.0
        pixel2depth_marker[1] = float(pixel2depth_marker[1])/1000.0
        pixel2depth_marker[2] = float(pixel2depth_marker[2])/1000.0
        return pixel2depth_marker
    
    def rvec2quat(self, rvec):
        htfmtx = np.array([[0, 0, 0, 0], # homogenous transformation matrix
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]],
                            dtype=float)
        htfmtx[:3, :3], _ = cv2.Rodrigues(rvec)
        quat = tf.transformations.quaternion_from_matrix(htfmtx)
        return quat  # [qx, qy, qz, qw]
    
    def cinfo_sub_cb(self, ros_cinfo):
        if not self.flag_detecting_aruco:
            self._intrinsics.width = ros_cinfo.width
            self._intrinsics.height = ros_cinfo.height
            self._intrinsics.ppx = ros_cinfo.K[2] # 2
            self._intrinsics.ppy = ros_cinfo.K[5] # 5
            self._intrinsics.fx = ros_cinfo.K[0]  # 0
            self._intrinsics.fy = ros_cinfo.K[4]  # 4
            self._intrinsics.model = rs.distortion.inverse_brown_conrady
            self._intrinsics.coeffs = [i for i in ros_cinfo.D]

    def sub_imgs_cb(self, ros_cimg, ros_dimg):
        if not self.flag_detecting_aruco:
            self.img_updataed = True
            self.cv2_color_img = self.bridgeC.imgmsg_to_cv2(ros_cimg, "bgr8")
            self.cv2_depth_img = self.bridgeC.imgmsg_to_cv2(ros_dimg, desired_encoding="passthrough")   

    def detect_aruco(self, cv2_img):
        aruco_detect_res = dict()
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        center_corners = np.array([])
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, rejectedImgPoints = aruco.estimatePoseSingleMarkers(corners, self.marker_length, 
                                                                            (self.mtx), (self.dist))
            np_corners = np.array(corners)
            for idx, id in enumerate(ids):
                tmp_corners = np_corners[idx]
                center_corners = tmp_corners.mean(axis=1)
                center_corners = np.rint(center_corners)
                aruco_tmp_res = dict()
                aruco_tmp_res['rvec'] = rvec[idx][0]
                aruco_tmp_res['tvec'] = tvec[idx][0]
                aruco_tmp_res['corners'] = tmp_corners
                aruco_tmp_res['ccorner'] = center_corners
                aruco_detect_res[str(id[0])] = aruco_tmp_res

            for i in range(0, ids.size):
                aruco.drawAxis(cv2_img, self.mtx, self.dist, rvec[i], 
                                tvec[i], self.marker_length/2)
            aruco.drawDetectedMarkers(cv2_img, corners)
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '
            cv2.putText(cv2_img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(cv2_img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        return cv2_img, aruco_detect_res


def detect_aruco(cv2_img, marker_length=0.028):
    
    mtx = np.array([[615.7256469726562, 0.0, 323.13262939453125], # TODO: to be replaced by the self._intrinsics params
                            [0.0, 616.17236328125, 237.86715698242188], 
                            [0.0, 0.0, 1.0]])
    dist =np.array([ 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]) # TODO: the same
    
    aruco_detect_res = dict()
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10
    center_corners = np.array([])
    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, rejectedImgPoints = aruco.estimatePoseSingleMarkers(corners, marker_length, 
                                                                        (mtx), (dist))
        np_corners = np.array(corners)
        for idx, id in enumerate(ids):
            tmp_corners = np_corners[idx]
            center_corners = tmp_corners.mean(axis=1)
            center_corners = np.rint(center_corners)
            aruco_tmp_res = dict()
            aruco_tmp_res['rvec'] = rvec[idx][0]
            aruco_tmp_res['tvec'] = tvec[idx][0]
            aruco_tmp_res['corners'] = tmp_corners
            aruco_tmp_res['ccorner'] = center_corners
            aruco_detect_res[str(id[0])] = aruco_tmp_res

        for i in range(0, ids.size):
            aruco.drawAxis(cv2_img, mtx, dist, rvec[i], 
                            tvec[i], marker_length/2)
        aruco.drawDetectedMarkers(cv2_img, corners)
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '
        cv2.putText(cv2_img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    else:
        cv2.putText(cv2_img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    return cv2_img, aruco_detect_res

# if __name__ == '__main__':
#     rs2ad = Rs2ArucoDetector(flag_run_alone=True)
#     while not rospy.is_shutdown():
#         res_dict = rs2ad.get_aruco_detect_res()
#         print('------')
#         print('res_dict:', res_dict)
#         rs2ad.rate.sleep()


if __name__ == '__main__':
    img_path = '/home/hdy/Desktop/charucoboard.png'
    img = cv2.imread(img_path)
    res_img, _ = detect_aruco(img)
    cv2.imshow('aruco detection result', res_img)
    while True:
        cv2.waitKey(10)
    
