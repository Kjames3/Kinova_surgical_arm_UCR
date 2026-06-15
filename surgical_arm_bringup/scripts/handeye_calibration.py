#!/usr/bin/env python3
import sys, os, threading, time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pymoveit2 import MoveIt2
import tf2_ros

IMAGE_TOPIC = "/global_camera/global_camera/color/image_raw"
CAMERA_MATRIX = np.array([[909.9,0.,640.0],[0.,909.8,360.0],[0.,0.,1.0]],dtype=np.float64)
DIST_COEFFS = np.zeros(5,dtype=np.float64)
MARKER_ID = 8
MARKER_SIZE = 0.2032
MARKER_OBJ_PTS = np.array([[-MARKER_SIZE/2,MARKER_SIZE/2,0],[MARKER_SIZE/2,MARKER_SIZE/2,0],[MARKER_SIZE/2,-MARKER_SIZE/2,0],[-MARKER_SIZE/2,-MARKER_SIZE/2,0]],dtype=np.float32)
SAVE_DIR = os.path.expanduser("~/Calibration_data/auto_run")
SETTLE_SECS = 3.0
CAMERA_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE,history=HistoryPolicy.KEEP_LAST,depth=1)
JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"]
BASE_LINK = "base_link"
EE_LINK = "end_effector_link"
GROUP = "manipulator"
POSES_RAD = [
    [-0.0362,  0.6621, -3.1087, -1.6009, -0.0051, -0.7185,  1.5692],
    [ 0.1140,  0.6758, -3.1302, -1.5740, -0.0140, -0.7321,  1.7096],
    [-0.3068,  0.7043, -3.0695, -1.5116,  0.0081, -0.7722,  1.3172],
    [-0.2472,  0.9510, -3.0724, -0.9866, -0.0066, -1.0487,  1.3724],
    [-0.2803,  0.7136, -3.0699, -1.2297,  0.0046, -1.0437,  1.3454],
    [-0.0631,  0.6741, -3.0865, -1.3071, -0.0160, -1.0007,  1.5613],
    [-0.0630,  0.6901, -3.0872, -1.3319, -0.0167, -0.9599,  1.5617],
    [ 0.2117,  0.7650, -3.1144, -1.4222, -0.0484, -0.7973,  1.8393],
    [-0.0841,  0.3948, -3.0784, -2.0248, -0.0180, -0.5622,  1.5611],
    [-0.2344,  0.3807, -3.0627, -2.0491,  0.0085, -0.5549,  1.4040],
]

class AutoCalibNode(Node):
    def __init__(self):
        super().__init__("auto_handeye_calib")
        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        self.moveit2 = MoveIt2(node=self,joint_names=JOINT_NAMES,base_link_name=BASE_LINK,end_effector_name=EE_LINK,group_name=GROUP,callback_group=self.cb_group)
        self.moveit2.planner_id = "PTP"
        self.moveit2.max_velocity = 0.15
        self.moveit2.max_acceleration = 0.10
        self.create_subscription(Image,IMAGE_TOPIC,self._image_cb,CAMERA_QOS,callback_group=self.cb_group)
        self.R_g2b=[];self.t_g2b=[];self.R_t2c=[];self.t_t2c=[];self.n=0
        os.makedirs(SAVE_DIR,exist_ok=True)
        self.get_logger().info("Node ready.")

    def _image_cb(self,msg):
        try:
            with self.lock: self.latest_frame=self.bridge.imgmsg_to_cv2(msg,"bgr8")
        except Exception as e: self.get_logger().error(str(e))

    def detect(self,frame):
        corners,ids,_=self.detector.detectMarkers(frame)
        if ids is None:
            vis=frame.copy()
            cv2.putText(vis,"Marker NOT found",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            return False,None,None,vis
        vis=frame.copy()
        cv2.aruco.drawDetectedMarkers(vis,corners,ids)
        for i,mid in enumerate(ids.flatten()):
            if mid==MARKER_ID:
                img_pts=corners[i][0].astype(np.float32)
                ok,rvec,tvec=cv2.solvePnP(MARKER_OBJ_PTS,img_pts,CAMERA_MATRIX,DIST_COEFFS,flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    cv2.drawFrameAxes(vis,CAMERA_MATRIX,DIST_COEFFS,rvec,tvec,0.05)
                    cv2.putText(vis,"MARKER OK",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                    return True,rvec,tvec,vis
        vis2=frame.copy()
        cv2.putText(vis2,"Marker NOT found",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        return False,None,None,vis2

    def get_tf(self):
        try:
            tf=self.tf_buffer.lookup_transform(BASE_LINK,EE_LINK,rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=2.0))
            t=tf.transform.translation;q=tf.transform.rotation
            return True,R.from_quat([q.x,q.y,q.z,q.w]).as_matrix(),np.array([t.x,t.y,t.z])
        except Exception as e:
            self.get_logger().warn(str(e));return False,None,None

    def move(self,joints_rad):
        self.moveit2.move_to_configuration(joints_rad)
        self.moveit2.wait_until_executed()

    def capture(self,idx):
        with self.lock: frame=self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None: print("  X No frame"); return False
        ok_m,rvec,tvec,vis=self.detect(frame)
        if not ok_m: print("  X Marker not detected"); cv2.imwrite(os.path.join(SAVE_DIR,"pose_%02d_NO_MARKER.png"%(idx+1)),frame); return False
        ok_tf,rot,t_ee=self.get_tf()
        if not ok_tf: print("  X TF failed"); return False
        cv2.imwrite(os.path.join(SAVE_DIR,"pose_%02d_ok.png"%(idx+1)),vis)
        R_t2c,_=cv2.Rodrigues(rvec)
        self.R_g2b.append(rot);self.t_g2b.append(t_ee)
        self.R_t2c.append(R_t2c);self.t_t2c.append(tvec.flatten())
        self.n+=1
        print("  OK Sample %d EE=(%.3f,%.3f,%.3f)"%(self.n,t_ee[0],t_ee[1],t_ee[2]))
        return True

    def solve(self):
        print("Computing from %d samples..."%self.n)
        if self.n<4: print("Need at least 4."); return
        best_R=best_t=best_name=None
        for name,method in [("TSAI",cv2.CALIB_HAND_EYE_TSAI),("PARK",cv2.CALIB_HAND_EYE_PARK),("HORAUD",cv2.CALIB_HAND_EYE_HORAUD),("ANDREFF",cv2.CALIB_HAND_EYE_ANDREFF)]:
            try:
                Rc,tc=cv2.calibrateHandEye(self.R_g2b,self.t_g2b,self.R_t2c,self.t_t2c,method=method)
                if Rc is not None: best_R,best_t,best_name=Rc,tc,name; print("  %s: SUCCESS"%name); break
            except Exception as e: print("  %s FAILED"%name)
        if best_R is None: print("All failed."); return
        tf=best_t.flatten()
        re=R.from_matrix(best_R).as_euler("xyz",degrees=True)
        rq=R.from_matrix(best_R).as_quat()
        yaml_path=os.path.join(SAVE_DIR,"eye_to_hand_calib.yaml")
        with open(yaml_path,"w") as f:
            f.write("translation:\n")
            f.write("  x: %.8f\n" % tf[0])
            f.write("  y: %.8f\n" % tf[1])
            f.write("  z: %.8f\n" % tf[2])
            f.write("\n")
            f.write("rotation_quaternion:\n")
            f.write("  x: %.8f\n" % rq[0])
            f.write("  y: %.8f\n" % rq[1])
            f.write("  z: %.8f\n" % rq[2])
            f.write("  w: %.8f\n" % rq[3])
        print("Saved:",yaml_path)
        print("ros2 run tf2_ros static_transform_publisher %.6f %.6f %.6f %.6f %.6f %.6f %.6f base_link global_camera_color_optical_frame"%(tf[0],tf[1],tf[2],rq[0],rq[1],rq[2],rq[3]))

    def run(self):
        rate=self.create_rate(10)
        while self.latest_frame is None and rclpy.ok(): rate.sleep()
        self.get_logger().info("Camera ready. Starting.")
        cv2.namedWindow("Calibration",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration",848,480)
        for i,joints in enumerate(POSES_RAD):
            if not rclpy.ok(): break
            print("Pose %d/10"%(i+1))
            self.move(joints)
            t0=time.time()
            while time.time()-t0<SETTLE_SECS:
                with self.lock: f=self.latest_frame
                if f is not None:
                    _,_,_,vis=self.detect(f)
                    cv2.putText(vis,"Pose %d/10 settling..."%(i+1),(10,vis.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    cv2.imshow("Calibration",vis)
                    cv2.waitKey(30)
            self.capture(i)
            with self.lock: f=self.latest_frame
            if f is not None:
                _,_,_,vis=self.detect(f)
                cv2.putText(vis,"Pose %d done Samples:%d"%(i+1,self.n),(10,vis.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.imshow("Calibration",vis)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        self.solve()

def main():
    rclpy.init(args=sys.argv)
    node=AutoCalibNode()
    executor=MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread=threading.Thread(target=executor.spin,daemon=True)
    spin_thread.start()
    print("AUTO HAND-EYE CALIBRATION - PTP")
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)

if __name__=="__main__": main()
