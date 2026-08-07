import rclpy
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import xml.etree.ElementTree as ET
import time

def get_robot_info(node, timeout_sec=5.0):
    """
    Subscribes to /robot_description and /robot_description_semantic.
    Returns (continuous_joints: set, max_vel: dict, home_joints: dict)
    If not available within timeout, returns empty structures.
    """
    urdf_xml = None
    srdf_xml = None

    def urdf_cb(msg):
        nonlocal urdf_xml
        urdf_xml = msg.data

    def srdf_cb(msg):
        nonlocal srdf_xml
        srdf_xml = msg.data

    qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    
    # Use temporary callback group to avoid interfering with node's main group
    cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
    sub1 = node.create_subscription(String, "/robot_description", urdf_cb, qos, callback_group=cb_group)
    sub2 = node.create_subscription(String, "/robot_description_semantic", srdf_cb, qos, callback_group=cb_group)

    node.get_logger().info("Waiting for /robot_description and /robot_description_semantic topics...")
    
    start = time.time()
    # Use an executor to spin just this callback group
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    while rclpy.ok() and time.time() - start < timeout_sec:
        executor.spin_once(timeout_sec=0.1)
        if urdf_xml and srdf_xml:
            break

    node.destroy_subscription(sub1)
    node.destroy_subscription(sub2)

    continuous_joints = set()
    max_vel = {}
    home_joints = {}

    if urdf_xml:
        try:
            root = ET.fromstring(urdf_xml)
            for joint in root.findall('joint'):
                if joint.get('type') == 'continuous':
                    continuous_joints.add(joint.get('name'))
                limit = joint.find('limit')
                if limit is not None and limit.get('velocity'):
                    max_vel[joint.get('name')] = float(limit.get('velocity'))
            node.get_logger().info("Successfully parsed URDF limits and continuous joints.")
        except Exception as e:
            node.get_logger().error(f"Error parsing URDF: {e}")
    else:
        node.get_logger().warn("Did not receive /robot_description topic.")

    if srdf_xml:
        try:
            root = ET.fromstring(srdf_xml)
            for gs in root.findall('group_state'):
                if gs.get('name') in ['home', 'ready']:
                    for j in gs.findall('joint'):
                        home_joints[j.get('name')] = float(j.get('value'))
                    break # prioritize first found
            if home_joints:
                node.get_logger().info(f"Successfully parsed SRDF home pose with {len(home_joints)} joints.")
            else:
                node.get_logger().warn("No 'home' or 'ready' pose found in SRDF.")
        except Exception as e:
            node.get_logger().error(f"Error parsing SRDF: {e}")
    else:
        node.get_logger().warn("Did not receive /robot_description_semantic topic.")

    return continuous_joints, max_vel, home_joints
