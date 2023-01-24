#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-12 19:46:22
LastEditors: Dianye Huang
LastEditTime: 2022-10-10 11:27:21
Description: 
    This script loads the calibration result file and publish it to the rviz for visualization,
    - The relative location of the calibration result file are fixed.
'''

import os
import rospy
import numpy as np
import tf.transformations as t
from gui_handeye_utils import CaliStore


if  __name__ == '__main__':
    
    rospy.init_node('handeye_cali_result_node', anonymous=True)
    
    # load calibratin result
    cali_filename = '/handeye_cali_results.yaml'
    cali_filepath = (os.path.dirname(os.path.dirname(__file__))
                        +'/config'+cali_filename)
    calistore = CaliStore(sample_filepath=None, result_filepath=cali_filepath)
    
    # get calibration matrix from panda_link8 to camera_color_optical_frame
    parent_frame, child_frame, cali_mtx = calistore.load_cali_res() 
    
    # calculate the calibration matrix to get tf 
    # from panda_link8 to camera_link. 
    # camera__link to camera_color_optical_frame
    pos = [-0.00054404, 0.0147895, 0.000158515]
    quat = [0.501967, -0.497588, 0.500452, -0.499983]
    cam_T_optical = t.quaternion_matrix(quat)
    cam_T_optical[:3, 3] = np.array(pos)
    child_frame = 'camera_link'
    cali_mtx = np.dot(cali_mtx, np.linalg.inv(cam_T_optical))
    
    # publish static frame 
    calistore.static_broadcast(tfmat=cali_mtx, parent_frame=parent_frame, 
                            child_frame=child_frame)
    
    rospy.spin()

