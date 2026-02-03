#!/usr/bin/env python3
"""
DualSense (PS5) controller test script.

Reads joystick inputs via pygame and prints them continuously at ~10 Hz.
No ROS dependency — standalone sanity check for controller connectivity.

Reference mapping (from data_gather_teleop.py ROS Joy message):
    axes[0] = left stick X    axes[1] = left stick Y
    axes[2] = right stick X   axes[3] = right stick Y
    axes[4] = d-pad L/R       axes[5] = d-pad U/D
    buttons[0] = X (keystep)  buttons[1] = A (close gripper)
    buttons[2] = B (open gripper) buttons[3] = Y (save)
    buttons[4/5] = LB/RB (x-rot)  buttons[6/7] = LT/RT click (y-rot)
    buttons[8] = Back (discard) buttons[9] = Start (finish)

Note: pygame DualSense mapping differs from ROS joy_node (Xbox-style).
      The indices below are for DualSense on Linux via pygame/SDL.
      D-pad is reported as a hat (not axes) by pygame.

Usage:
    python3 test_new_joystick.py
    Press Ctrl+C to exit.
"""

import sys
import time

import pygame

# DualSense axis names (Linux/pygame via SDL — indices may shift per driver)
AXIS_NAMES = {
    0: "Left Stick X",
    1: "Left Stick Y",
    2: "Right Stick X",
    3: "Right Stick Y",
    4: "L2 Trigger",
    5: "R2 Trigger",
}

# DualSense button names (Linux/pygame via SDL)
BUTTON_NAMES = {
    0: "Cross (X)",
    1: "Circle",
    2: "Triangle",
    3: "Square",
    4: "L1",
    5: "R1",
    6: "L2 Click",
    7: "R2 Click",
    8: "Share",
    9: "Options",
    10: "PS",
    11: "L3",
    12: "R3",
    13: "Touchpad",
}


def main():
    pygame.init()
    pygame.joystick.init()

    num_joysticks = pygame.joystick.get_count()
    if num_joysticks == 0:
        print("No joystick detected. Is the controller connected?")
        pygame.quit()
        sys.exit(1)

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print("=" * 60)
    print(f"Controller : {joystick.get_name()}")
    print(f"Axes       : {joystick.get_numaxes()}")
    print(f"Buttons    : {joystick.get_numbuttons()}")
    print(f"Hats (D-pad): {joystick.get_numhats()}")
    print("=" * 60)
    print("Reading inputs at ~10 Hz. Press Ctrl+C to exit.\n")

    try:
        while True:
            pygame.event.pump()

            # --- Axes ---
            num_axes = joystick.get_numaxes()
            axes_lines = []
            for i in range(num_axes):
                val = joystick.get_axis(i)
                name = AXIS_NAMES.get(i, f"Axis {i}")
                bar = "#" * int(abs(val) * 10)
                sign = "+" if val >= 0 else "-"
                axes_lines.append(f"  [{i}] {name:>15s}: {val:+.3f} {sign}{bar}")

            # --- Buttons ---
            num_buttons = joystick.get_numbuttons()
            pressed = []
            for i in range(num_buttons):
                if joystick.get_button(i):
                    name = BUTTON_NAMES.get(i, f"Btn {i}")
                    pressed.append(f"[{i}] {name}")

            # --- Hats (D-pad) ---
            num_hats = joystick.get_numhats()
            hat_lines = []
            for i in range(num_hats):
                hx, hy = joystick.get_hat(i)
                directions = []
                if hy > 0:
                    directions.append("Up")
                elif hy < 0:
                    directions.append("Down")
                if hx < 0:
                    directions.append("Left")
                elif hx > 0:
                    directions.append("Right")
                label = "+".join(directions) if directions else "Neutral"
                hat_lines.append(f"  Hat {i}: ({hx:+d}, {hy:+d}) {label}")

            # --- Print ---
            print("\033[2J\033[H", end="")  # clear screen
            print("=== DualSense Controller Test ===\n")
            print("Axes:")
            print("\n".join(axes_lines))
            print()
            print(f"Buttons pressed: {', '.join(pressed) if pressed else '(none)'}")
            print()
            print("D-pad:")
            for line in hat_lines:
                print(line)
            print("\n[Ctrl+C to exit]")

            time.sleep(0.1)  # ~10 Hz

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
