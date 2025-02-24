# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
import numpy as np
from board_controller import BoardController
from servo_bus_controller import ServoBusController
import utils as ut

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters

class HiwonderRobot:
    def __init__(self):
        """
        Initializes the HiwonderRobot, including:
        - Hardware controllers (board and servo bus)
        - Default joint states and control parameters
        - Kinematic parameters for the 5-DOF robotic arm
        - Denavit-Hartenberg (DH) parameters and transformation matrices for forward kinematics
        - Moves the robot to its home position
        """
        # ----------------------------
        # Hardware Controllers
        # ----------------------------
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        # ----------------------------
        # Joint States & Control Settings
        # ----------------------------
        self.joint_values = [0, 0, 90, -30, 0, 0]  # Current joint angles (in degrees)
        self.home_position = [0, 0, 90, -30, 0, 0]   # Home position (in degrees)
        self.joint_limits = [
            [-120, 120],  # Joint 1 limits
            [-90, 90],    # Joint 2 limits
            [-120, 120],  # Joint 3 limits
            [-100, 100],  # Joint 4 limits
            [-90, 90],    # Joint 5 limits
            [-120, 30]    # Gripper/EE limits
        ]
        self.joint_control_delay = 0.2  # Delay for individual joint control commands (secs)
        self.speed_control_delay = 0.2  # Delay for base speed control (secs)

        # ----------------------------
        # Kinematic Parameters for the 5-DOF Arm
        # ----------------------------
        # Link lengths (in meters)
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12
        self.num_dof = 5              # Number of degrees-of-freedom for the arm
        self.theta = [0, 0, 0, 0, 0]    # Internal joint angles for kinematics (in radians)
        self.ee = ut.EndEffector()      # End-effector state (position and orientation)
        self.points = [None] * (self.num_dof + 1)  # To store robot points for visualization / kinematics

        # ----------------------------
        # Denavit-Hartenberg Parameters & Transformation Matrices
        # ----------------------------
        # Each row represents [theta, d, a, alpha] for a joint.
        # Initially, we use self.theta (which are all zeros) for the joint angles.
        self.DH = np.array([
            [self.theta[0], self.l1, 0,         np.pi/2],
            [self.theta[1], 0,         self.l2, 0],
            [self.theta[2], 0,         self.l3, 0],
            [self.theta[3], 0,         0,         np.pi/2],
            [self.theta[4], self.l5,  0,         0]
        ])
        # Pre-allocate transformation matrices (one for each joint)
        self.T = np.zeros((self.num_dof, 4, 4))

        # ----------------------------
        # Move to Home Position
        # ----------------------------
        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f'---------------------------------------------------------------------')
        
        # Compute forward kinematics using utils
        self.calc_forward_kinematics(self.joint_values, radians=False)
        
        # Get end-effector position
        position = [self.ee.x, self.ee.y, self.ee.z]
        print(f'[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n')

        # Update joint velocities
        self.set_arm_velocity(cmd)


    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """ Computes wheel speeds based on joystick input and sends them to the board """
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        # insert your code for finding "speed"

        speed = [0]*4
        
        ######################################################################

        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def calc_forward_kinematics(self, theta=None, radians=False):
        """Computes forward kinematics to determine the end-effector position.

        Args:
            theta (list, optional): Joint angles in degrees or radians. Defaults to self.theta.
            radians (bool, optional): Whether input angles are in radians. Defaults to False.
        """
        if theta is None:
            theta = self.theta  # Use current joint angles
        
        if not radians:
            theta = np.radians(theta)  # Convert degrees to radians

        # Define Denavit-Hartenberg parameters (theta, d, a, alpha) for the 5-DOF arm
        DH_params = [
            [theta[0], self.l1, 0, np.pi/2],  # Joint 1
            [theta[1], 0, self.l2, 0],        # Joint 2
            [theta[2], 0, self.l3, 0],        # Joint 3
            [theta[3], 0, 0, np.pi/2],        # Joint 4
            [theta[4], self.l5, 0, 0]         # Joint 5
        ]

        # Compute transformation matrices using DH parameters
        T_final = np.eye(4)  # Initialize as identity matrix
        for dh in DH_params:
            T_final = np.dot(T_final, ut.dh_to_matrix(dh))  # Multiply transformations

        # Extract end-effector position from final transformation matrix
        self.ee.x, self.ee.y, self.ee.z = T_final[:3, 3]

        # Extract end-effector orientation (roll, pitch, yaw)
        rpy = ut.rotm_to_euler(T_final[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy

        print(f"[DEBUG] FK Computed EE Position: X={self.ee.x:.3f}, Y={self.ee.y:.3f}, Z={self.ee.z:.3f}")
        print(f"[DEBUG] FK Computed EE Orientation: RotX={self.ee.rotx:.3f}, RotY={self.ee.roty:.3f}, RotZ={self.ee.rotz:.3f}")

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel (list): Desired end-effector velocity [vx, vy, vz, ...].
                        (This implementation uses only the linear velocity components.)
        """
        n = self.num_dof

        # Update forward kinematics to get current EE position and transformation matrices
        self.calc_forward_kinematics(self.theta, radians=True)

        # Initialize Jacobian (3 x n matrix for linear velocity)
        J = np.zeros((3, n))
        
        # Extract current end-effector position
        p_e = np.array([self.ee.x, self.ee.y, self.ee.z])
        
        # Compute Jacobian by accumulating transformations
        T_cumulative = np.eye(4)
        for i in range(n):
            T_cumulative = np.dot(T_cumulative, self.T[i])
            p_i = T_cumulative[:3, 3]  # Position of the i-th joint
            z_i = T_cumulative[:3, 2]  # Rotation axis of the i-th joint
            J[:, i] = np.cross(z_i, (p_e - p_i))
        
        # Compute joint velocity updates using the pseudo-inverse of the Jacobian
        joint_v = np.linalg.pinv(J) @ np.array(vel)
        
        # Update joint angles using a small time step dt
        dt = 0.05  # Time step (seconds)
        self.theta = [self.theta[i] + joint_v[i] * dt for i in range(n)]
        
        # Recompute forward kinematics after updating the joint angles
        self.calc_forward_kinematics(self.theta, radians=True)

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities and individual joint inputs.

        Args:
            cmd (GamepadCmds): Contains linear velocities and individual joint control inputs.
        """
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]  # Extract linear velocity input

        # Compute joint velocity updates (this updates self.theta)
        self.calc_velocity_kinematics(vel)

        # Convert the updated joint angles (in radians) to degrees for further processing
        computed_angles_deg = [np.rad2deg(theta) for theta in self.theta]

        print(f'[DEBUG] Current joint angles (deg) = {self.joint_values}')
        print(f'[DEBUG] Linear velocity: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}')
        print(f'[DEBUG] Computed joint angles (deg) before individual control = {computed_angles_deg}')

        # Time step and gain for individual joint control
        dt = 0.5  # Fixed time step
        K = 10    # Mapping gain for individual joint control

        # Initialize new joint angle list (6 joints total)
        new_thetalist = [0.0] * 6

        # Apply velocity-based control: update joints 0-4 based on previous joint values and computed change
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * self.theta[i]  # self.theta[i] now holds the computed delta

        # Apply individual joint control modifications
        new_thetalist[0] += dt * K * cmd.arm_j1
        new_thetalist[1] += dt * K * cmd.arm_j2
        new_thetalist[2] += dt * K * cmd.arm_j3
        new_thetalist[3] += dt * K * cmd.arm_j4
        new_thetalist[4] += dt * K * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K * cmd.arm_ee  # Separate control for the gripper/EE

        # Round values for cleaner output
        new_thetalist = [round(theta, 2) for theta in new_thetalist]
        print(f'[DEBUG] Final commanded joint angles (deg) = {new_thetalist}')

        # Set the new joint angles into the system (this method should update both simulation and hardware if needed)
        self.set_joint_values(new_thetalist, radians=False)


    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle """
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.rad2deg(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)
        
        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)


    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles.

        Args:
            thetalist (list): Target joint angles in degrees.
            duration (int): Movement duration in milliseconds.
        """
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.rad2deg(theta) for theta in thetalist]

        thetalist = self.enforce_joint_limits(thetalist)
        self.joint_values = thetalist # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(thetalist) # remap the joint values from software to hardware

        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            self.servo_bus.move_servo(joint_id, pulse, duration)


    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)]


    def move_to_home_position(self):
        print(f'Moving to home position...')
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f'Arrived at home position: {self.joint_values} \n')
        time.sleep(1.0)
        print(f'------------------- System is now ready!------------------- \n')


    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """ Converts degrees to servo pulse value """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int((x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min)


    def pulse_to_angle(self, x: float):
        """ Converts servo pulse value to degrees """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round((x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2)


    def stop_motors(self):
        """ Stops all motors safely """
        self.board.set_motor_speed([0]*4)
        print("[INFO] Motors stopped.")


    def remap_joints(self, thetalist: list):
        """Reorders angles to match hardware configuration.

        Args:
            thetalist (list): Software joint order.

        Returns:
            list: Hardware-mapped joint angles.

        Note: Joint mapping for hardware
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5] 
            joint[2] = joint[4] 
            joint[3] = joint[3] 
            joint[4] = joint[2] 
            joint[5] = joint[1] 
        """
        return thetalist[::-1]