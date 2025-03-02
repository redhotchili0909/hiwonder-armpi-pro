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
            [-120, 120],  # Joint 1
            [-90, 90],    # Joint 2
            [-120, 120],  # Joint 3
            [-100, 100],  # Joint 4
            [-90, 90],    # Joint 5
            [-120, 30]    # Gripper
        ]
        self.joint_control_delay = 0.2  # Delay for individual joint control commands (secs)
        self.speed_control_delay = 0.2  # Delay for base speed control (secs)

        # ----------------------------
        # Kinematic Parameters for the 5-DOF Arm
        # (Link lengths in meters)
        # ----------------------------
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12

        # Move to Home Position at startup
        self.move_to_home_position()

    # -------------------------------------------------------------
    # Interfacing with the mobile base
    # -------------------------------------------------------------
    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands."""
        if cmd.arm_home:
            self.move_to_home_position()

        print('---------------------------------------------------------------------')

        # For demonstration, you can do base control here if desired:
        # self.set_base_velocity(cmd)

        # Directly set the arm velocity (which also updates joint angles)
        self.set_arm_velocity(cmd)

    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """Computes wheel speeds based on joystick input and sends them to the board."""
        # Example wheel-speed array (replace with actual logic).
        speed = [0]*4
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Arm Control (Inline FK & Jacobian)
    # -------------------------------------------------------------
    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """
        Calculates new joint angles from:
          - Linear velocity commands (cmd.arm_vx, cmd.arm_vy, cmd.arm_vz)
          - Individual joint overrides (cmd.arm_j1 ... cmd.arm_j5, cmd.arm_ee)
        And applies them directly.
        """
        # --------------------------
        # 1) Build Jacobian
        # --------------------------
        theta_radians = np.radians(self.joint_values[:5])  # first 5 are the arm (ignore gripper)
        # Setup Denavit-Hartenberg parameters inline for each of the 5 DOFs.
        # Each row is [theta, d, a, alpha].
        DH = [
            [theta_radians[0], self.l1, 0,         np.pi/2],
            [theta_radians[1], 0,        self.l2,  0      ],
            [theta_radians[2], 0,        self.l3,  0      ],
            [theta_radians[3]-np.pi/2, 0,        0,        -np.pi/2],
            [theta_radians[4], self.l5+self.l4,  0,        0      ]
        ]

        T_cumulative = np.eye(4)
        positions = []
        z_axes = []

        for i in range(5):
            T_i = ut.dh_to_matrix(DH[i])
            T_cumulative = T_cumulative @ T_i

            positions.append(T_cumulative[:3, 3].copy())  # x, y, z
            z_axes.append(T_cumulative[:3, 2].copy())     # unit vector of z

        # End-effector position (the last T_cumulative gives us the final pose)
        p_e = positions[-1]

        # Construct a 3x5 Jacobian for linear velocity
        J = np.zeros((3, 5))
        for i in range(5):
            # cross(z_i, (p_e - p_i))
            J[:, i] = np.cross(z_axes[i], (p_e - positions[i]))

        # --------------------------
        # 2) Solve for joint velocities
        # --------------------------
        vel = np.array([cmd.arm_vx, cmd.arm_vy, cmd.arm_vz])
        J_inv = np.linalg.pinv(J)
        joint_vel_radians = J_inv @ vel
        thetalist_dot = np.degrees(joint_vel_radians)

        print(f'[DEBUG] Current thetalist (deg) = {self.joint_values}') 
        print(f'[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}')
        print(f'[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}')

        # --------------------------
        # 3) Update joints
        # --------------------------        

        dt = 0.5 # Fixed time step
        K = 1600 # mapping gain for individual joint control
        new_thetalist = [0.0]*6

        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * thetalist_dot[i]
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

    # -------------------------------------------------------------
    # Joint Setting and Utility
    # -------------------------------------------------------------
    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle (in degrees by default). """
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.degrees(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)
        
        print(f"[DEBUG] Moving joint {joint_id} to {theta}° (pulse={pulse})")
        time.sleep(self.joint_control_delay)

    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles (degrees by default)."""
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.degrees(t) for t in thetalist]

        thetalist = self.enforce_joint_limits(thetalist)

        self.joint_values = thetalist

        mapped = self.remap_joints(thetalist)

        for joint_id, angle_deg in enumerate(mapped, start=1):
            pulse = self.angle_to_pulse(angle_deg)
            self.servo_bus.move_servo(joint_id, pulse, duration)

    def enforce_joint_limits(self, thetalist, joint_id=None):
        """Clamps joint angles to their hardware limits."""
        if joint_id is not None:
            # Single angle case
            index = joint_id - 1
            return float(np.clip(thetalist, *self.joint_limits[index]))
        else:
            # List of angles
            return [
                float(np.clip(a, *lim))
                for a, lim in zip(thetalist, self.joint_limits)
            ]

    def move_to_home_position(self):
        print("Moving to home position...")
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f"Arrived at home position: {self.joint_values}")
        time.sleep(1.0)
        print("------------------- System is now ready!-------------------\n")

    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------
    def angle_to_pulse(self, x: float):
        """ Converts degrees to servo pulse value (0..1000). """
        hw_min, hw_max = 0, 1000
        joint_min, joint_max = -150, 150
        return int((x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min)

    def pulse_to_angle(self, x: float):
        """ Converts servo pulse (0..1000) back to degrees. """
        hw_min, hw_max = 0, 1000
        joint_min, joint_max = -150, 150
        return round((x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2)

    def stop_motors(self):
        """ Stops all motors safely. """
        self.board.set_motor_speed([0]*4)
        print("[INFO] Motors stopped.")

    def remap_joints(self, thetalist: list):
        """
        Reorders angles to match hardware configuration.

        Note: Joint mapping for hardware vs. software:
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5]
            joint[2] = joint[4]
            joint[3] = joint[3]
            joint[4] = joint[2]
            joint[5] = joint[1]
        """
        return thetalist[::-1]