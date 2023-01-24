#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-12 17:01:51
LastEditors: Dianye Huang
LastEditTime: 2022-11-12 15:21:00
Description: 
'''

import rospy
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
import tf
import tf2_ros
import tf.transformations as t
from geometry_msgs.msg import TransformStamped


import cv2
import numpy as np
import yaml


class CaliSampler():
    def __init__(self, 
                group_name="panda_arm", 
                base_link="panda_link0",
                endeffector_link="panda_link8") -> None:
        # wait until tf tree is set up
        print('CaliSampler: waiting_for_message: move_group/status')
        rospy.wait_for_message('move_group/status', GoalStatusArray)
        print('CaliSampler: move_group/status okay!')
        
        # use moveit to position the robot 
        self.group_name = group_name
        self.base_link = base_link
        self.ee_link = endeffector_link
        self.max_vel_limit = 0.1
        self.commander = MoveGroupCommander(self.group_name)
        self.commander.set_max_velocity_scaling_factor(self.max_vel_limit)

        # one sample to be recorded 
        self.jpos = list()
        self.ee_trans = list()  # w.r.t base frame, in the Frana case is "panda_link0"
        self.ee_quat = list()   
        self.ar_trans = list()  # w.r.t camera frame, in the realsense case is "camera_color_optical_frame"
        self.ar_quat = list()

        #         self.jpos = jpos
        # self.ee_trans = ee_trans
        # self.ee_quat = ee_quat
        # self.ar_trans = ar_trans
        # self.ar_quat = ar_quat

        # get ee pose from tf tree
        self.tf_listener = tf.TransformListener()
    
    def get_pose_from_tf_tree(self, 
                            child_frame='panda_link8', 
                            parent_frame='panda_link0'):
        '''
        Description: 
            get translation and quaternion of child frame w.r.t. the parent frame
        @ param : child_frame{string} 
        @ param : parent_frame{string} 
        @ return: trans{list} -- x, y, z
        @ return: quat{list}: -- x, y, z, w
        '''    
        self.tf_listener.waitForTransform(parent_frame, 
                                        child_frame, 
                                        rospy.Time(), 
                                        rospy.Duration(1.0))
        (trans,quat) = self.tf_listener.lookupTransform(parent_frame,  # parent frame
                                                        child_frame,   # child frame
                                                        rospy.Time(0))
        return trans, quat


class CaliCompute():
    '''
    Description: 
        The CaliCompute would load the sample from a file with predefined formatm or 
        obtains the sample during the acquisition, and then computes the calibration
        matrix, i.e. the homogeneous transformation matrix of the TCP coordinate 
        w.r.t. the base frame. [Eye in hand calibration program to be genralized]
    URLs for reference:
        https://blog.csdn.net/weixin_43994752/article/details/123781759
    @ param : {}: 
    @ return: {}: 
    param {*} self
    '''    
    def __init__(self):
        self.AVAILABLE_ALGORITHMS = {
            'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        # sample buffer for calibration purpose, each elements in the list are of np.array type
        self.ee2base_trans = list()  # end-effector to base 3x1 translation vector
        self.ee2base_rotmat = list() # end-effector to base 3x3 rotation matrix 
        self.ar2cam_trans = list()   # aruco to camera 3x1 translation vector
        self.ar2cam_rotmat = list()  # aruco to camera 3x3 rotation matrix

    def load_sample_buffer_from_file(self, filepath:str):
        with open(filepath, 'r') as f:
            txt_list = f.readlines()[1::]
            sample_list = []
            for txt_line in txt_list:
                txt_line = txt_line[0:-1]
                sample_line = [float(entry) for entry in txt_line.split('\t')]
                sample_list.append(sample_line)
        for s in np.array(sample_list):
            self.push_back_sample_buffer(s[7:10], s[10:14], s[14:17], s[17:21])

    def reset_sample_buffer(self):
        '''
        Description: 
            clear the sample buffer
        '''        
        self.ee2base_trans = list()  
        self.ee2base_rotmat = list() 
        self.ar2cam_trans = list()   
        self.ar2cam_rotmat = list() 

    def push_back_sample_buffer(self, ee2base_trans:list, 
                                    ee2base_quat:list, 
                                    ar2cam_trans:list,
                                    ar2cam_quat:list):
        '''
        Description: 
            get the translation (position) of the end-effector and aruco,
            and their orientation (quaternion) w.r.t. the base frame and
            the camera frame respectively.
        @ param : ee2base_trans{list} -- 3x1 end-effector position w.r.t. base frame
        @ param : ee2base_quat{list}  -- 4x1 end-effector orientation w.r.t. base frame
        @ param : ar2cam_trans{list}  -- 3x1 aruco position w.r.t. camera frame
        @ param : ar2cam_quat{list}   -- 4x1 aruco orientation w.r.t. camera frame
        '''
        self.ee2base_trans.append(np.array(ee2base_trans).reshape(3,1))         
        self.ee2base_rotmat.append(t.quaternion_matrix(ee2base_quat)[:3, :3])
        self.ar2cam_trans.append(np.array(ar2cam_trans).reshape(3,1))
        self.ar2cam_rotmat.append(t.quaternion_matrix(ar2cam_quat)[:3, :3])

    def compute(self, method='Tsai-Lenz', filepath=None):
        '''
        Description: 
            Computet the handeye calibration matrix from the sample buffer, with the help of cv2
            the default methods are 'Tsai-Lenz'
        @ param : method{string} -- some built-in method in opencv2 
        @ return: {}: 
        '''
        calires = np.identity(4)
        if not filepath is None:
            self.load_sample_buffer_from_file(filepath) 
        if len(self.ee2base_trans) > 3:
            cam2ee_rotmat, cam2ee_trans = cv2.calibrateHandEye(self.ee2base_rotmat, self.ee2base_trans,
                                                                    self.ar2cam_rotmat, self.ar2cam_trans,
                                                                    method=self.AVAILABLE_ALGORITHMS[method])
            calires[:3, :3] = cam2ee_rotmat
            calires[:3, 3] = cam2ee_trans.reshape(-1)
            print('CaliCompute --Calibration results:')
            print(calires)
        else:
            print('Sampling more before computing the calibration result ...')
        return calires

class CaliStore():
    '''
    Description: 
        CaliStore will store the calibration result and broadcast 
        the camera frame to the tf tree for visualization
    '''    
    def __init__(self, sample_filepath, result_filepath) -> None:
        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._result_filepath = result_filepath

        if not sample_filepath is None:
            self._log_sample_file = sample_filepath
            self._save_sample_init()


    def static_broadcast(self, tfmat, child_frame, parent_frame):
        # transform the calibration results into trans and quaternion
        trans = t.translation_from_matrix(tfmat)
        quat = t.quaternion_from_matrix(tfmat)
        # tf messages
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.transform.translation.x = trans[0]
        tf_msg.transform.translation.y = trans[1]
        tf_msg.transform.translation.z = trans[2]
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        self._static_broadcaster.sendTransform(tf_msg)

    def load_cali_res(self, filepath=None):
        '''
        Description: 
            the file that saves the calibration results is under the format:
            - parent_frame_id
            - child_frame_id
            - transformation (columnized 4x4 matrix)
        @ return : parent_frame_id{string} 
        @ return : child_frame_id{string}  
        @ return : cali_mtx{np.array}      --4x4 homogeneous translation matrix
        '''       
        if filepath is None:
            filepath=self._result_filepath
        with open(filepath, 'r') as f:
            cali_res = yaml.load(f.read(), Loader=yaml.FullLoader)
        parent_frame_id = cali_res['parent_frame_id']
        child_frame_id = cali_res['child_frame_id']
        cali_mtx = np.array(cali_res['cali_mtx']).reshape(4,4).transpose()
        print('CaliStore, Loading handeye calibration result')
        print('-- parent frame id :', parent_frame_id)
        print('-- child  frame id :', child_frame_id)
        print('-- transform matrix:')
        print(cali_mtx)
        return parent_frame_id, child_frame_id, cali_mtx

    def save_cali_res(self, res_mat, 
                    parent_frame_id="panda_link8", 
                    child_frame_id="camera_link", 
                    filepath=None):
        res_dict = dict()
        res_dict['parent_frame_id'] = parent_frame_id
        res_dict['child_frame_id'] = child_frame_id
        vec_res_mat = list()
        for cols in range(4):
            for rows in range(4):
                vec_res_mat.append(res_mat[rows, cols].tolist())
        res_dict["cali_mtx"] = vec_res_mat

        if filepath is None:
            filepath = self._result_filepath
        with open(filepath, 'w') as f:
            yaml.dump(res_dict, f)

    def _save_sample_init(self):
        '''
        Description: 
            clear the file and write a header line
        '''        
        with open(self._log_sample_file, 'w') as f:
            f.write('q1'+'\t' + 'q2'+'\t' + 'q3'+'\t' + \
                    'q4'+'\t' + 'q5'+'\t' + 'q6'+'\t' + 'q7'+'\t' +\
                    'ee_px'+'\t' + 'ee_py'+'\t' + 'ee_pz'+'\t' + 
                    'ee_qx'+'\t' + 'ee_qy'+'\t' + 'ee_qz'+'\t' + 'ee_qw'+'\t' + 
                    'ar_px'+'\t' + 'ar_py'+'\t' + 'ar_pz'+'\t' + 
                    'ar_qx'+'\t' + 'ar_qy'+'\t' + 'ar_qz'+'\t' + 'ar_qw'+'\t' + '\n')

    def save_sample(self, jpos, ee_trans, ee_quat, ar_trans, ar_quat):
        '''
        Description: 
            save samples into the log file, and update the class members
        @ param : jpos{list} --7x1 joint positions of the robot 
        @ param : ee_trans{} --3x1 translation of the end-effector w.r.t. base frame
        @ param : ee_quat{}  --4x1 quaternion of the end-effector w.r.t. base frame
        @ param : ar_trans{} --3x1 translation of the aruco w.r.t. camera frame
        @ param : ar_quat{}  --4x1 quaternion of the aruco w.r.t. camera frame
        ''' 
        with open(self._log_sample_file, 'a') as f:
                # joint states
                for j in jpos:
                    f.write(str(j)+'\t')
                # ee pose
                for t in ee_trans:
                    f.write(str(t)+'\t')
                for q in ee_quat:
                    f.write(str(q)+'\t')
                # aruco pose
                for t in ar_trans:
                    f.write(str(t)+'\t')
                for idx, q in enumerate(ar_quat):
                    if idx==3:
                        f.write(str(q)+'\n')
                    else:
                        f.write(str(q)+'\t')     

