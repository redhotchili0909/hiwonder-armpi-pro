"""
Main Application Script
-----------------------
Coordinates gamepad input and robot control.
"""

import sys, os
import time
import threading
import traceback
from queue import Queue

# Extend system path to include script directory
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from hiwonder import HiwonderRobot
from gamepad_control import GamepadControl
import utils


# Initialize components
cmdlist = []    # Stores recent gamepad commands
gpc = GamepadControl()
robot = HiwonderRobot()

console_queue = Queue()

def monitor_gamepad():
    """ Continuously reads gamepad inputs and stores the latest command. """
    try:
        while True:
            if len(cmdlist) > 2:
                cmdlist.pop(0)  # Retain only the latest two commands
            cmdlist.append(gpc.get_gamepad_cmds())
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("[INFO] Gamepad monitoring stopped.")


def console_input():
    """
    Thread that waits for console (stdin) input.
    You can type IK commands, mode changes, etc. without blocking the main loop.
    """
    print("[INFO] Console input thread started. Type commands like:")
    print("       ik numeric 100 200 50   (numeric IK)")
    print("       ik analytic 100 200 50  (analytic IK)")
    try:
        while True:
            raw = input("> ")
            raw = raw.strip()
            if not raw:
                continue
            
            parts = raw.split()
            command = parts[0].lower()

            if command == "exit" or command == "quit":
                print("[INFO] Exiting console thread...")
                console_queue.put(("mode", "exit"))
                break
            
            elif command == "ik":
                if len(parts) < 5:
                    print("[WARN] Usage: ik <numeric|analytic> x y z")
                    continue
                sub_cmd = parts[1].lower()
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                except ValueError:
                    print("[WARN] x/y/z must be numeric.")
                    continue

                if sub_cmd == "numeric":
                    console_queue.put(("ik-numeric", (x, y, z)))
                elif sub_cmd == "analytic":
                    console_queue.put(("ik-analytic", (x, y, z)))
                else:
                    print("[WARN] IK mode must be 'numeric' or 'analytic'.")

            elif command == "stop":
                console_queue.put(("stop", None))

            else:
                print(f"[WARN] Unrecognized command: {raw}")

    except KeyboardInterrupt:
        print("[INFO] Console input thread stopping due to Ctrl-C.")

def shutdown_robot():
    print("\n[INFO] Shutting down the robot safely...")

    # Stop motors and reset servos to a safe position
    robot.stop_motors()
    robot.set_joint_values(robot.home_position, duration=600)
    time.sleep(1.5)  # Allow time for servos to reposition

    # Close communication interfaces
    print("[INFO] Closing hardware interfaces...")
    robot.board.close()
    robot.servo_bus.close()

    print("[INFO] Shutdown complete. Safe to power off.")


def main():
    """ Main loop that reads gamepad commands and console commands, then updates the robot. """
    try:
        # Start the gamepad monitoring thread
        gamepad_thread = threading.Thread(target=monitor_gamepad, daemon=True)
        gamepad_thread.start()

        # Start a console input thread
        console_thread = threading.Thread(target=console_input, daemon=True)
        console_thread.start()
        
        control_interval = 0.25  # Seconds per control cycle
        
        keep_running = True
        while keep_running:
            cycle_start = time.time()

            if cmdlist:
                latest_cmd = cmdlist[-1]
                robot.set_robot_commands(latest_cmd)

            while not console_queue.empty():
                mode, payload = console_queue.get_nowait()
                
                if mode == "ik-numeric":
                    (x, y, z) = payload
                    print(f"[INFO] Received IK command to move to x={x}, y={y}, z={z}.")
                    joint_angles, calc_time = robot.inverse_kinematics(x, y, z, mode = "numeric")
                    robot.set_joint_values(joint_angles)
                    print(f"[INFO] Joint angles calculated: {joint_angles}")
                    print(f"Time taken for IK calculation: {calc_time} seconds")
                    continue

                elif mode == "ik-analytic":
                    (x, y, z) = payload
                    print(f"[INFO] Received IK command to move to x={x}, y={y}, z={z}.")
                    joint_angles, calc_time = robot.inverse_kinematics(x, y, z, mode = "analytic")
                    robot.set_joint_values(joint_angles)
                    print(f"[INFO] Joint angles calculated: {joint_angles}")
                    print(f"Time taken for IK calculation: {calc_time} seconds")
                    continue

                elif mode == "stop":
                    print("[INFO] Stopping all movement from console command.")
                    robot.stop_motors()
                    continue

                elif mode == "exit":
                    print("[INFO] Exiting main loop based on console request.")
                    keep_running = False
                    continue

                else:
                    print(f"[WARN] Unhandled console mode '{mode}'.")

            elapsed = time.time() - cycle_start
            remaining_time = control_interval - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        shutdown_robot()


if __name__ == "__main__":
    main()
