import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
import math
from tf_transformations import euler_from_quaternion

class PurePursuitPIDController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_pid_controller')
        self.logger = self.get_logger()
        
        # Vehicle parameters
        self.wheelbase = 1.8  # Distance between front and rear wheels
        
        # Pure Pursuit parameters
        self.k = 1.0  # Lookahead gain
        self.lookahead_distance = 3.0  # Default lookahead distance
        self.linear_velocity = 3.0  # Desired forward velocity (m/s)

        # PID Controller parameters for longitudinal control
        self.kp = 1.0  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.1  # Derivative gain
        self.previous_error = 0.0
        self.integral = 0.0

        # ROS 2 Subscriptions and Publishers
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/pose_msg',
            self.pose_callback,
            10
        )
        self.car_pose_subscription = self.create_subscription(
            PoseArray,
            '/pose_info',
            self.car_pose_callback,
            10
        )
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.logger.info("Pure Pursuit + PID Controller node started")

        # State variables
        self.target_pose = None
        self.car_pose = None
        self.yaw = None

    def pose_callback(self, msg: PoseStamped):
        """Callback to receive the target pose."""
        self.target_pose = msg.pose
        self.logger.info(f'Received target pose: {self.target_pose}')

    def car_pose_callback(self, msg: PoseArray):
        """Callback to receive the current car pose and execute control logic."""
        if len(msg.poses) > 0:
            self.car_pose = msg.poses[0]
            self.yaw = self.get_yaw_from_pose(self.car_pose)
            self.logger.info(f'Received car pose: {self.car_pose}, yaw: {self.yaw}')
            self.control_vehicle()

    def get_yaw_from_pose(self, pose):
        """Extract yaw from the quaternion of the car's pose."""
        orientation_q = pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        return yaw

    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def pid_control(self, target_velocity, current_velocity):
        """PID controller to adjust vehicle's linear velocity."""
        error = target_velocity - current_velocity
        self.integral += error
        derivative = error - self.previous_error

        # PID control output
        control_output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error
        return max(0.0, min(control_output, 10.0))  # Limit velocity within [0, 10] m/s

    def steering_angle_to_angular_velocity(self, steering_angle):
        """Convert the steering angle to angular velocity."""
        angular_velocity = math.tan(steering_angle) * self.linear_velocity / self.wheelbase
        return angular_velocity

    def control_vehicle(self):
        """Control the vehicle using Pure Pursuit for steering and PID for velocity."""
        if self.target_pose is None or self.car_pose is None or self.yaw is None:
            return

        # Extract current and target positions
        current_x = self.car_pose.position.x
        current_y = self.car_pose.position.y
        target_x = self.target_pose.position.x
        target_y = self.target_pose.position.y

        # Calculate lookahead distance dynamically
        dx = target_x - current_x
        dy = target_y - current_y
        lookahead_distance = self.distance((current_x, current_y), (target_x, target_y))
        lookahead_distance = max(self.k * self.linear_velocity, lookahead_distance)

        # Calculate the steering angle using Pure Pursuit
        alpha = math.atan2(dy, dx) - self.yaw
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), lookahead_distance)

        # Assume current velocity is the linear.x value (you can modify based on actual implementation)
        current_velocity = math.sqrt(
            self.car_pose.position.x ** 2 + self.car_pose.position.y ** 2
        )

        # Use PID controller to adjust linear velocity
        adjusted_velocity = self.pid_control(self.linear_velocity, current_velocity)

        # Create and publish the Twist message
        twist = Twist()
        twist.linear.x = adjusted_velocity
        twist.angular.z = self.steering_angle_to_angular_velocity(steering_angle)

        self.cmd_vel_publisher.publish(twist)
        self.logger.info(f'Published cmd_vel: linear.x={twist.linear.x}, angular.z={twist.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    controller = PurePursuitPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
