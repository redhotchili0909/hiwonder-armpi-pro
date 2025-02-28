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
        # Hardware Controllers
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        # Joint states (in degrees) and control parameters
        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees
        self.joint_limits = [
            [-120, 120], [-90, 90], [-120, 120],
            [-100, 100], [-90, 90], [-120, 30]
        ]
        self.joint_control_delay = 0.2  # secs
        self.speed_control_delay = 0.2

        # Kinematic Parameters for the 5-DOF Arm
        # (Our parameters remain unchanged)
        self.l1 = 0.155
        self.l2 = 0.099
        self.l3 = 0.095
        self.l4 = 0.055
        self.l5 = 0.105        
        self.num_dof = 5  # number of arm joints (ignoring gripper/EE as a separate DOF)
        self.thetalist_dot = [0]*5

        # End-effector state from our utility class (if needed)
        self.ee = ut.EndEffector()
        # Pre-allocate storage for individual transformation matrices (one per joint)
        self.T = np.zeros((self.num_dof, 4, 4))
        
        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base (unchanged)
    # -------------------------------------------------------------
    def set_base_velocity(self, cmd: ut.GamepadCmds):
        speed = [0]*4  # Replace with your base speed computation
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Forward Kinematics
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f'---------------------------------------------------------------------')
        
        # self.set_base_velocity(cmd)
        self.set_arm_velocity(cmd)

        ######################################################################

        self.calc_forward_kinematics()
        
        ######################################################################

    def calc_forward_kinematics(self, theta=None, radians=False):
        """Computes forward kinematics to determine the end-effector pose.
        
        Args:
            theta (list, optional): Joint angles in degrees or radians. Defaults to self.theta.
            radians (bool, optional): Whether input angles are in radians. Defaults to False.
        """
        if theta is None:
            theta = self.joint_values
        else:
            if not radians:
                theta = np.deg2rad(theta)

        # Here we define our DH parameters inline.
        # Our convention: [theta, alpha, a, d] for each joint.
        # Adjust offsets as needed; these values use our key parameters.
        DH = np.array([
            [theta[0], np.pi/2, 0,         self.l1],
            [theta[1], 0,         self.l2,   0],
            [theta[2], 0,         self.l3,   0],
            [theta[3], np.pi/2,   self.l4,   0],
            [theta[4], 0,         self.l5,   0]
        ])

        T_final = np.eye(4)
        for i in range(self.num_dof):
            ct = np.cos(DH[i, 0])
            st = np.sin(DH[i, 0])
            ca = np.cos(DH[i, 1])
            sa = np.sin(DH[i, 1])
            a = DH[i, 2]
            d = DH[i, 3]

            # Construct the transformation matrix for joint i:
            Ti = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            self.T[i] = Ti  # Store individual joint transform
            T_final = T_final @ Ti  # Accumulate overall transform

        # Extract end-effector position and orientation
        self.ee.x, self.ee.y, self.ee.z = T_final[0:3, 3]
        rpy = ut.rotm_to_euler(T_final[0:3, 0:3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy

        print(f"[DEBUG] FK: EE Position: X={self.ee.x:.3f}, Y={self.ee.y:.3f}, Z={self.ee.z:.3f}")
        print(f"[DEBUG] FK: EE Orientation: RotX={self.ee.rotx:.3f}, RotY={self.ee.roty:.3f}, RotZ={self.ee.rotz:.3f}")

    # -------------------------------------------------------------
    # Jacobian and Velocity Kinematics (custom style)
    # -------------------------------------------------------------
    def compute_Jacobian(self):
        """Compute the 3x5 Jacobian matrix for the linear velocity of the end-effector."""
        J = np.zeros((3, self.num_dof))
        p_e = np.array([self.ee.x, self.ee.y, self.ee.z])
        T_cumulative = np.eye(4)
        for i in range(self.num_dof):
            T_cumulative = T_cumulative @ self.T[i]
            p_i = T_cumulative[0:3, 3]
            z_i = T_cumulative[0:3, 2]
            J[:, i] = np.cross(z_i, (p_e - p_i))
        return J

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate joint velocity updates (delta theta) required to achieve the given end-effector velocity.
        This implementation considers only the linear velocity components.
        """
        # Update forward kinematics so that self.T and self.ee are current.
        J = self.compute_Jacobian()
        # Compute pseudoinverse of the Jacobian
        J_inv = np.linalg.pinv(J)
        vel = np.array(vel)
        # Compute joint velocity updates (in rad/s)
        self.thetalist_dot = J_inv @ np.array(vel*0.2)
        print(self.thetalist_dot)


    # -------------------------------------------------------------
    #  Control for the Arm
    # -------------------------------------------------------------
    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """
        Calculates and sets new joint angles purely from linear velocity commands.
        Args:
            cmd (GamepadCmds): Contains linear velocity inputs for the arm.
        """
        # Extract linear velocity inputs (e.g., in m/s)
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
        # Compute joint velocity updates from the Jacobian.

        print(f"[DEBUG] Current joint angles (deg): {self.joint_values}")
        print(f"[DEBUG] Linear velocity: {[round(v, 3) for v in vel]}")

        self.calc_velocity_kinematics(vel)
        print(f"[DEBUG] Joint Velocity: {[round(angel_vel,2) for angel_vel in self.thetalist_dot]}")

        # Update joint angles
        dt = 0.5 # Fixed time step
        K = 5 # mapping gain for individual joint control
        new_thetalist = [0.0]*6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * self.thetalist_dot[i]
        # individual joint control
        new_thetalist[0] += dt * K * cmd.arm_j1
        new_thetalist[1] += dt * K * cmd.arm_j2
        new_thetalist[2] += dt * K * cmd.arm_j3
        new_thetalist[3] += dt * K * cmd.arm_j4
        new_thetalist[4] += dt * K * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K * cmd.arm_ee

        new_thetalist = [round(theta,2) for theta in new_thetalist]
        print(f'[DEBUG] Commanded thetalist (deg) = {new_thetalist}')       
        
        # set new joint angles
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