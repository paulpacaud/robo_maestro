#!/usr/bin/env python3
"""
DualSense joystick drift debugger.

Displays all axes and buttons in real-time so you can see exactly which
inputs are non-zero when the controller is idle.  Highlights values that
exceed configurable thresholds.

Usage:
    python3 joystick_debug.py                 # default deadzone 0.20
    python3 joystick_debug.py --deadzone 0.10 # custom deadzone
    python3 joystick_debug.py --log drift.csv # also write to CSV

Controls:
    r       – reset min/max tracking
    q / ESC – quit
"""

import argparse
import csv
import os
import sys
import time

import pygame


def main():
    parser = argparse.ArgumentParser(description="Joystick drift debugger")
    parser.add_argument(
        "--deadzone", type=float, default=0.20,
        help="Deadzone threshold to highlight (default: 0.20)",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Optional CSV file to log raw values",
    )
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Connect a controller and retry.")
        sys.exit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()

    n_axes = joy.get_numaxes()
    n_buttons = joy.get_numbuttons()
    n_hats = joy.get_numhats()

    print(f"Controller : {joy.get_name()}")
    print(f"Axes       : {n_axes}")
    print(f"Buttons    : {n_buttons}")
    print(f"Hats       : {n_hats}")
    print(f"Deadzone   : {args.deadzone}")
    print()
    print("Leave the controller untouched and watch for non-zero values.")
    print("Press 'r' to reset min/max, 'q' or ESC to quit.")
    print("-" * 72)

    # Min/max trackers per axis
    axis_min = [0.0] * n_axes
    axis_max = [0.0] * n_axes

    # Button event counters (how many JOYBUTTONDOWN events per button)
    btn_events = [0] * n_buttons

    csv_file = None
    csv_writer = None
    if args.log:
        csv_file = open(args.log, "w", newline="")
        header = ["time_s"]
        header += [f"axis_{i}" for i in range(n_axes)]
        header += [f"btn_{i}" for i in range(n_buttons)]
        header += [f"hat_{i}_x" for i in range(n_hats)]
        header += [f"hat_{i}_y" for i in range(n_hats)]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    t0 = time.monotonic()
    try:
        while True:
            pygame.event.pump()

            # Handle keyboard / controller quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        raise KeyboardInterrupt
                    if event.key == pygame.K_r:
                        axis_min[:] = [0.0] * n_axes
                        axis_max[:] = [0.0] * n_axes
                        btn_events[:] = [0] * n_buttons
                if event.type == pygame.JOYBUTTONDOWN:
                    btn_events[event.button] += 1

            elapsed = time.monotonic() - t0

            # --- Gather current values ---
            axes = [joy.get_axis(i) for i in range(n_axes)]
            buttons = [joy.get_button(i) for i in range(n_buttons)]
            hats = [joy.get_hat(i) for i in range(n_hats)]

            for i, v in enumerate(axes):
                axis_min[i] = min(axis_min[i], v)
                axis_max[i] = max(axis_max[i], v)

            # --- CSV log ---
            if csv_writer:
                row = [f"{elapsed:.3f}"]
                row += [f"{v:.4f}" for v in axes]
                row += [str(b) for b in buttons]
                for h in hats:
                    row += [str(h[0]), str(h[1])]
                csv_writer.writerow(row)

            # --- Terminal output ---
            # Move cursor to top of output area (after the header)
            sys.stdout.write(f"\033[{n_axes + n_buttons + n_hats + 5}A")

            dz = args.deadzone
            sys.stdout.write(
                f"{BOLD}t={elapsed:7.1f}s{RESET}  "
                f"(values exceeding deadzone {dz} shown in {RED}red{RESET})"
                f"{CLEAR_LINE}\n"
            )
            sys.stdout.write(f"{CLEAR_LINE}\n")

            # Axes
            sys.stdout.write(f"{BOLD}{'AXIS':>8}  {'VALUE':>8}  {'MIN':>8}  {'MAX':>8}  BAR{RESET}{CLEAR_LINE}\n")
            for i, v in enumerate(axes):
                exceeds = abs(v) > dz
                color = RED if exceeds else (YELLOW if abs(v) > 0.05 else GREEN)
                bar_w = 20
                bar_pos = int((v + 1) / 2 * bar_w)
                bar = "." * bar_w
                bar = bar[:bar_pos] + "|" + bar[bar_pos + 1:]
                label = ""
                if i == 0:   label = " (LX)"
                elif i == 1: label = " (LY)"
                elif i == 2: label = " (RX)"
                elif i == 3: label = " (RY)"
                elif i == 4: label = " (L2)"
                elif i == 5: label = " (R2)"
                sys.stdout.write(
                    f"  ax {i:>2}{label:<5} "
                    f"{color}{v:>+8.4f}{RESET}  "
                    f"{axis_min[i]:>+8.4f}  "
                    f"{axis_max[i]:>+8.4f}  "
                    f"[{bar}]"
                    f"{' *** DRIFT' if exceeds else ''}"
                    f"{CLEAR_LINE}\n"
                )

            sys.stdout.write(f"{CLEAR_LINE}\n")

            # Buttons — show in a compact grid
            sys.stdout.write(f"{BOLD}BUTTONS (poll / events){RESET}{CLEAR_LINE}\n")
            for i, b in enumerate(buttons):
                color = RED if b else GREEN
                label = ""
                if i == 0:   label = "Cross"
                elif i == 1: label = "Circle"
                elif i == 2: label = "Tri"
                elif i == 3: label = "Square"
                elif i == 4: label = "L1"
                elif i == 5: label = "R1"
                elif i == 6: label = "L2"
                elif i == 7: label = "R2"
                elif i == 8: label = "Share"
                elif i == 9: label = "Opts"
                elif i == 10: label = "PS"
                elif i == 11: label = "L3"
                elif i == 12: label = "R3"
                else:         label = f"b{i}"
                evt_warn = f" ({btn_events[i]} events)" if btn_events[i] > 0 else ""
                drift = " *** STUCK" if b else ""
                sys.stdout.write(
                    f"  btn {i:>2} {label:<7} "
                    f"{color}{b}{RESET}"
                    f"{evt_warn}{drift}"
                    f"{CLEAR_LINE}\n"
                )

            # Hats
            for i, h in enumerate(hats):
                color = RED if h != (0, 0) else GREEN
                sys.stdout.write(
                    f"  hat {i:>2}       "
                    f"{color}{h}{RESET}"
                    f"{CLEAR_LINE}\n"
                )

            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
            print(f"\nLog saved to {args.log}")
        pygame.quit()
        print("\nDone.")


if __name__ == "__main__":
    # We need a tiny display surface for pygame key events
    os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENLESS", "1")
    pygame.display.init()
    try:
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
    except pygame.error:
        pass  # headless is fine, keyboard quit won't work but Ctrl-C will
    main()
