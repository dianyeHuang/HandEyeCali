#!/usr/bin/env python

'''
Author: Dianye Huang
Date: 2023-01-09 14:26:11
LastEditors: Dianye Huang
LastEditTime: 2023-01-24 11:20:49
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
        self.img_updated = False
        self.marker_length = 0.028 # mind the setting
        self.rate = rospy.Rate(30)
                
    
    def set_aruco_length(self, val:float):
        self.marker_length = val

    def get_aruco_detect_res(self, use_charuco=False):
        if self.img_updated:
            self.aruco_detect_res = dict()
            
            if use_charuco:
                res_img, detect_res = self.detect_charuco(self.cv2_color_img)
                if detect_res.keys():
                    qvec = self.rvec2quat(detect_res['qvec']) 
                    tvec = detect_res['trans'].reshape(-1)
                    self.aruco_detect_res['charuco'] = (list(tvec), list(qvec))
                
            else:
                self.flag_detecting_aruco = True
                res_img, detect_res = self.detect_aruco(self.cv2_color_img)
            
                for id in detect_res.keys():
                    ccorner = detect_res[id]['ccorner']
                    dimg_arr = np.array(self.cv2_depth_img, dtype=np.float32)
                    # tvec = detect_res[id]['tvec'] # or use translations from aruco, where marker length matters
                    tvec = self.dimg_2_tras_vec(ccorner, dimg_arr, charuco=use_charuco) # tx, ty, tz, using depth info could avoid setting marker length
                    qvec = self.rvec2quat(detect_res[id]['rvec'])  # qx, qy, qz, qw
                    self.aruco_detect_res[id] = (tvec, list(qvec))

            self.flag_detecting_aruco = False
            self.img_updated = False
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
            self.img_updated = True
            self.cv2_color_img = self.bridgeC.imgmsg_to_cv2(ros_cimg, "bgr8")
            self.cv2_depth_img = self.bridgeC.imgmsg_to_cv2(ros_dimg, desired_encoding="passthrough")   

    def detect_aruco(self, cv2_img):
        aruco_detect_res = dict()
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)  # TODO old version
        # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()  # TODO old version
        # parameters = aruco.DetectorParameters()
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
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, 
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
                cv2.drawFrameAxes(image=cv2_img, cameraMatrix=self.mtx, distCoeffs=self.dist, 
                                    rvec=rvec[i], tvec=tvec[i], length=self.marker_length/2)
            aruco.drawDetectedMarkers(cv2_img, corners)
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '
            cv2.putText(cv2_img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(cv2_img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        return cv2_img, aruco_detect_res
    
    def detect_charuco(self, cv2_img):
        charuco_detect_res = dict()
        self.flag_detecting_aruco = False
        
        # find the charuco board
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)  # TODO
        parameters = aruco.DetectorParameters_create()  # TODO old version
        
        # parameters = aruco.DetectorParameters()
        # constant to be passed into Aruco methods
        # charuco_board = aruco.CharucoBoard(
        #     size=(4, 4), squareLength=0.04, markerLength=0.02, 
        #     dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        # )
        
        charuco_board = aruco.CharucoBoard_create(
            squaresX=4, squaresY=4, squareLength=0.04, markerLength=0.02, # TODO length and number
            dictionary = aruco.Dictionary_get(aruco.DICT_4X4_100)  # TODO old version
        )
        
        # detect QR code (markers)
        corners, ids, rejectedCorners = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)
        corners, ids, rejectedCorners, _ = aruco.refineDetectedMarkers(image=cv2_img, board=charuco_board, detectedCorners=corners, detectedIds=ids, rejectedCorners=rejectedCorners)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        id_color = (0, 255, 0)
        # if any marker is detected
        if ids is not None:
            aruco.drawDetectedMarkers(image=cv2_img, corners=corners, ids=ids)
            # read chessboard corners between markers
            _, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                            markerIds=ids,
                                                                            image=gray,
                                                                            board=charuco_board)
            # if at least one charuco corner is detected
            if charucoIds is not None:
                aruco.drawDetectedCornersCharuco(image=cv2_img, charucoCorners=charucoCorners, charucoIds=charucoIds, cornerColor=id_color)
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners=charucoCorners, charucoIds=charucoIds, board=charuco_board,
                                                            cameraMatrix=self.mtx, distCoeffs=self.dist, rvec=None, tvec=None)
                if valid:
                    cv2.drawFrameAxes(image=cv2_img, cameraMatrix=self.mtx, distCoeffs=self.dist, rvec=rvec, tvec=tvec, length=0.1)
                    charuco_detect_res['qvec'] = rvec
                    charuco_detect_res['trans'] = tvec
                    
        else:
            cv2.putText(img=cv2_img, text="No id detected", org=(0,64), fontFace=font, fontScale=1, color=id_color, lineType=cv2.LINE_AA)
            
        return cv2_img, charuco_detect_res
            

if __name__ == '__main__':
    rs2ad = Rs2ArucoDetector(flag_run_alone=True)
    while not rospy.is_shutdown():
        res_dict = rs2ad.get_aruco_detect_res(use_charuco=True)
        print('------')
        print('res_dict', res_dict)
        rs2ad.rate.sleep()





This repository provide a ros package for eye in hand calibration of the Realsense D435 camera attached to the Franka Emika Panda robot.