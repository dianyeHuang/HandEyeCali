<!--
 * @Author: Dianye Huang
 * @Date: 2022-08-08 13:57:11
 * @LastEditors: Dianye Huang
 * @LastEditTime: 2022-08-08 14:19:25
 * @Description: 
-->


The core algorithm used to identify camera external params, in cv2 module

refer to: https://github.com/IFL-CAMP/easy_handeye

the whole process involves 
- sampling
- computing using opencv2's calibrateHandEye
- storing results
- publishing results

```python
import cv2
hand_camera_rot, hand_camera_tr = cv2.calibrateHandEye(hand_world_rot, hand_world_tr, marker_camera_rot, marker_camera_tr, method=method)
result = tfs.affines.compose(np.squeeze(hand_camera_tr), hand_camera_rot, [1, 1, 1])
```

# sampling
mind that the roation is stored in such sequence: w, x, y, z; and translation in: x, y, z
```python
def _msg_to_opencv(transform_msg):
    cmt = transform_msg.translation
    tr = np.array((cmt.x, cmt.y, cmt.z))
    cmq = transform_msg.rotation
    rot = tfs.quaternions.quat2mat((cmq.w, cmq.x, cmq.y, cmq.z))
    return rot, tr
```

# computing
methods are listed as below:
```python
AVAILABLE_ALGORITHMS = {
    'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
    'Park': cv2.CALIB_HAND_EYE_PARK,
    'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}
```

# storing


# publishing

