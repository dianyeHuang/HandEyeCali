#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-12 19:46:22
LastEditors: Dianye Huang
LastEditTime: 2022-08-12 20:46:27
Description: 
    This script loads the calibration result file and publish it to the rviz for visualization,
    - The relative location of the calibration result file are fixed.
'''

import os
import rospy
from gui_handeye_utils import CaliStore

if  __name__ == '__main__':
    rospy.init_node('handeye_cali_result_node', anonymous=True)
    cali_filename = '/handeye_cali_results.yaml'
    cali_filepath = (os.path.dirname(os.path.dirname(__file__))
                        +'/config'+cali_filename)
    calistore = CaliStore(sample_filepath=None, result_filepath=cali_filepath)
    parent_frame, child_frame, cali_mtx = calistore.load_cali_res()
    calistore.static_broadcast(tfmat=cali_mtx, parent_frame=parent_frame, 
                            child_frame=child_frame) # TODO
    # point cloud topics are publish under the camera_color_optical_frame
    calistore.static_broadcast(tfmat=cali_mtx, parent_frame=parent_frame, 
                            child_frame='camera_color_optical_frame') 
    rospy.spin()

