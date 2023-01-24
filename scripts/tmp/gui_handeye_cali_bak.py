#!/usr/bin/env python

'''
Author: Dianye Huang
Date: 2022-08-08 14:23:56
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-03 12:38:16
Description: 

'''

# qt
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QMainWindow, QApplication, QWidget
from PyQt5.QtWidgets import QPushButton, QLabel, QGraphicsView, QGraphicsScene, QLineEdit
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout

# ros
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

# others
import cv2
from cv_bridge import CvBridge
from rs2_aruco_detector import Rs2ArucoDetector
from gui_handeye_utils import CaliSampler, CaliCompute, CaliStore


class CalibGUI(QMainWindow):
    '''
    Description: 
        CalibGUI serves as a handeye calibration manager, that assembling
        the samples by processing messages from ros, and utilized the 
        handeyecalibration api in cv2 to get the calibration result. The 
        caliration matrix is then broadcasted to rviz for an intuitive
        verification.
    @ param : sample_path{string} --filepath to save the samples for calibration
    @ param : result_path{string} --filepath to save the calibration results
    @ param : width{int}  --width of the GUI
    @ param : height{int} --height of the GUI
    '''
    def __init__(self, 
                sample_path, 
                result_path, 
                group_name="panda_arm", 
                base_link="panda_link0", 
                ee_link="panda_link8", 
                camera_link="camera_link", 
                pointcloud_link='camera_color_optical_frame',
                aruco_id = '582',
                aruco_resimg_topic = 'rs2_aruco_detector/res_img',
                cali_method='Tsai-Lenz',
                width=800, 
                height=800):
        super().__init__()
        # 1. ROS init
        rospy.init_node('gui_handeye_calibration_node', anonymous=True)
        self.rate = rospy.Rate(50)

        self.group_name = group_name
        self.base_link = base_link
        self.ee_link = ee_link
        self.camera_link = camera_link
        self.pointcloud_link = pointcloud_link
        self.cali_method = cali_method

        # 2. Calibration modules
        self.sampler = CaliSampler(group_name=group_name, base_link=self.base_link, endeffector_link=self.ee_link)
        self.compute = CaliCompute()
        self.store = CaliStore(sample_path, result_path)

        # -- Aruco detector
        self.aruco_cali_id = aruco_id
        vis_img_topic_name = aruco_resimg_topic
        self.aruco_res = dict()
        self.aruco_detector = Rs2ArucoDetector(res_img_topic_name = vis_img_topic_name)
        self.img = None
        self.bridgeC = CvBridge()
        self.cimg_sub = rospy.Subscriber(vis_img_topic_name, Image, 
                                        self.sub_img_cb)
        rospy.wait_for_message('/camera/color/image_raw', Image, timeout=3)
        #  -- Joint states subscribe
        self.jpos_sub = rospy.Subscriber("/franka_state_controller/joint_states", 
                                        JointState, self.sub_jpos_cb)
        rospy.wait_for_message("/franka_state_controller/joint_states", JointState, timeout=3)

        # 3. GUI init
        # -- layout
        self.log_cnt = 0
        self.prompt_str = "Do you want to record the current joint and ee states, and aruco results?\n" + \
                        "- Press sample to confirm recording joint states\n" + \
                        "- Press Quit to exit the program"
        self.width = width
        self.height = height
        self.initUI()
        # -- timers
        self.aruco_timer = QTimer()
        self.aruco_timer.timeout.connect(self.show_aruco_res)  # connect signals and slot
        self.aruco_timer.start(30)
        self.robot_timer = QTimer()
        self.robot_timer.timeout.connect(self.show_robot_states)
        self.robot_timer.start(30) 


    def show_aruco_res(self):
        self.aruco_res = self.aruco_detector.get_aruco_detect_res()
        if not self.img is None:
            y, x = self.img.shape[:2]
            frame = QImage(self.img, x, y, QImage.Format.Format_RGB888)
            self.pix = QPixmap.fromImage(frame)
            self.item=QGraphicsPixmapItem(self.pix)
            self.scene = QGraphicsScene()
            self.scene.addItem(self.item)
            self.imgshow.setScene(self.scene)

        if not self.aruco_res is None:
            if self.aruco_res.__contains__(self.aruco_cali_id):
                astr = '<b>Aruco '+ self.aruco_cali_id + ' pose (px, py, pz, qx, qy, qz, qw):</b>'
                (self.sampler.ar_trans, self.sampler.ar_quat) = self.aruco_res[self.aruco_cali_id]
                for p in self.sampler.ar_trans: # trans
                    p = round(p, 2)
                    astr += '\t' + str(p)
                for p in self.sampler.ar_quat: # quat
                    p = round(p, 3) 
                    astr += '\t' + str(p)
                self.ainfo.setText(astr)
        
    def show_robot_states(self):
        # joint states
        jstr = '<b>Current joint position (rad):</b>'
        for j in self.sampler.jpos:
            j = round(j, 3)
            jstr += '\t' + str(j)
        self.jinfo.setText(jstr)
        # end-effector states
        estr = '<b>End-effector pose (px, py, pz, qx, qy, qz, qw):</b>'
        self.sampler.ee_trans, self.sampler.ee_quat = self.sampler.get_pose_from_tf_tree(child_frame=self.ee_link, 
                                                                                        parent_frame=self.base_link)
        for t in self.sampler.ee_trans:
            t = round(t, 3)
            estr += '\t' + str(t)
        for q in self.sampler.ee_quat:
            q = round(q, 3)
            estr += '\t' + str(q)
        self.einfo.setText(estr)

    def sub_img_cb(self, ros_img):
        img = self.bridgeC.imgmsg_to_cv2(ros_img, "passthrough")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def sub_jpos_cb(self, ros_msg):
        self.sampler.jpos = ros_msg.position

    def on_click_sample(self):
        self.log_cnt += 1
        self.prompt.setText('Have recorded '+ str(self.log_cnt) + 
                            ' samples<jpos, aruco pose and ee pose>\n' + self.prompt_str)
        if not self.aruco_res is None:
            self.compute.push_back_sample_buffer(self.sampler.ee_trans, self.sampler.ee_quat, 
                                                    self.sampler.ar_trans, self.sampler.ar_quat)
            self.store.save_sample(self.sampler.jpos, self.sampler.ee_trans, self.sampler.ee_quat, 
                                    self.sampler.ar_trans, self.sampler.ar_quat)
    
    def on_click_arucolen(self):
        # self.aruco
        pass

    def on_click_arucoid(self):
        self.aruco_cali_id = self.lnedt_arucoid.text() 
        astr = '<b>Aruco '+ self.aruco_cali_id + ' pose (px, py, pz, qx, qy, qz, qw):</b>'
        self.ainfo.setText(astr)

    def on_click_compute(self):
        # get results
        # self.load_sample_buffer_from_file('/home/hdy/grasping_ws/franka_ws/src/franka_handeye_cali/scripts/hi2.txt') 
        cali_mat = self.compute.compute(method=self.cali_method) 
        # store reults
        self.store.save_cali_res(cali_mat, parent_frame_id=self.sampler.ee_link, child_frame_id=self.camera_link)
        # publish static frames
        for _ in range(3):
            self.store.static_broadcast(tfmat=cali_mat,
                                    child_frame=self.camera_link, 
                                    parent_frame=self.ee_link)
            self.store.static_broadcast(tfmat=cali_mat,
                                    child_frame=self.pointcloud_link, 
                                    parent_frame=self.ee_link)
            self.rate.sleep()
    
    def initUI(self):
        # setting init geometry of GUI
        self.resize(self.width, self.height)

        # init widgets
        # 1. menu
        # pass

        # 2. buttons
        # -1 aruco length button
        # self.btn_arucolen = QPushButton('Set aruco len', self)
        # self.btn_arucolen.clicked.connect(self.on_click_arucolen)
        # -2 aruco id button
        self.btn_arucoid = QPushButton('Set aruco id', self)
        self.btn_arucoid.clicked.connect(self.on_click_arucoid)
        # -3 sample button
        self.btn_sample = QPushButton('Sample', self)
        self.btn_sample.clicked.connect(self.on_click_sample)
        # -3 compute button
        self.btn_compute = QPushButton('Compute', self)
        self.btn_compute.clicked.connect(self.on_click_compute)
        # -4 quit button
        self.btn_quit = QPushButton('Quit', self)
        self.btn_quit.clicked.connect(QApplication.instance().quit)

        # 3. labels
        self.prompt = QLabel(self.prompt_str)
        self.jinfo = QLabel('<b>Current joint positions (rad):</b>')
        self.ainfo = QLabel('<b>Aruco '+ self.aruco_cali_id + ' pose (px, py, pz, qx, qy, qz, qw):</b>')
        self.einfo = QLabel('<b>End-effector pose (px, py, pz, qx, qy, qz, qw):</b>')

        # 4. graphics
        self.imgshow = QGraphicsView()
        self.imgshow.setObjectName('realsense_raw_color_imgs')

        # 5. line editor
        self.lnedt_arucoid = QLineEdit(self.aruco_cali_id)
        # self.lnedt_arucolen = QLineEdit("0.098")

        # add status bar
        self.statusBar().showMessage('Ready')

        # 3. setup layout
        self.hl_settings = QHBoxLayout()
        self.hl_settings.addSpacing(10)
        self.hl_settings.addStretch(1)
        self.hl_settings.addWidget(self.lnedt_arucoid)
        self.hl_settings.addWidget(self.btn_arucoid)
        # self.hl_settings.addWidget(self.lnedt_arucolen)
        # self.hl_settings.addWidget(self.btn_arucolen)

        self.hl_button = QHBoxLayout()
        self.hl_button.addSpacing(10)
        self.hl_button.addStretch(1)
        self.hl_button.addWidget(self.btn_sample)
        self.hl_button.addWidget(self.btn_compute)
        self.hl_button.addWidget(self.btn_quit)

        self.layout = QGridLayout()
        self.layout.setSpacing(5)
        self.layout.addWidget(self.imgshow, 1, 0)
        self.layout.addWidget(self.jinfo,  2, 0)
        self.layout.addWidget(self.ainfo,  3, 0)
        self.layout.addWidget(self.einfo,  4, 0)
        self.layout.addWidget(self.prompt, 5, 0)
        self.layout.addLayout(self.hl_settings, 6, 0)
        self.layout.addLayout(self.hl_button, 7, 0)

        # setup Qwidget in mainWindow
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('Franka Handeye Calibration Log cmd GUI')
        self.show()


import os
if  __name__ == '__main__':
    app = QApplication([])
    sample_filename = '/handeye_cali_samples.txt'
    result_filename = '/handeye_cali_results.yaml'
    config_path = os.path.dirname(os.path.dirname(__file__))+'/config'
    gui = CalibGUI( sample_path=config_path+sample_filename,
                    result_path=config_path+result_filename)
    sys.exit(app.exec())





