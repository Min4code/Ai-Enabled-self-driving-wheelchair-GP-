#!/usr/bin/env python3
import socket
import time
import tkinter as tk
from tkinter import font as tkFont, simpledialog, messagebox
import numpy as np
import math
import threading
import queue

# --- Configuration ---
DEFAULT_LISTENER_PORT = 9876

# Robot Control Configuration
ROBOT_COMMAND_IP = "192.168.250.225"
ROBOT_COMMAND_PORT = 9000
ROBOT_COMMAND_TIMEOUT_S = 2.0  # Timeout for sending TCP command to robot

# Tkinter display settings
APP_WIDTH = 850
APP_HEIGHT = 750
DEFAULT_FONT_FAMILY = "Arial"
DEFAULT_FONT_SIZE = 10
DEFAULT_FONT = (DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE)
TITLE_FONT = (DEFAULT_FONT_FAMILY, 14, "bold")
BUTTON_FONT = (DEFAULT_FONT_FAMILY, 10, "bold")

# Colors
BG_COLOR = "#F0F0F0"
ACCENT_COLOR = "#D32F2F"  # Reddish accent
TEXT_COLOR_DARK = "#1A1A1A"
CANVAS_BG_COLOR = "#FFFFFF"
ROBOT_BODY_COLOR = "#00796B"  # Teal
ROBOT_ACCENT_COLOR = "#004D40"  # Darker Teal
GRID_COLOR_HIGH = (230, 0, 0)  # Obstacle color (high intensity)
GRID_COLOR_LOW = (255, 150, 150)  # Obstacle color (low intensity)

# Lidar Data Interpretation
# CRITICAL: Set this based on your Lidar's physical mounting.
# Angle to ADD to Lidar's raw angle to make 0 degrees = ROBOT'S PHYSICAL FORWARD.
# E.g., If Lidar's 0-degree is robot's physical FRONT: set to 0.0
# E.g., If Lidar's 0-degree is robot's physical REAR: set to 180.0
LIDAR_MOUNTING_OFFSET_DEG = 0.0  # <<<< USER: VERIFY AND SET THIS!

# Occupancy Grid settings for 4x4m map
MAP_SIZE_METERS = 4.0
CELL_SIZE_MM = 50  # Each grid cell is 5cm x 5cm
GRID_DIMENSION = int(MAP_SIZE_METERS * 1000 / CELL_SIZE_MM)  # 4000mm / 50mm = 80x80 grid
MAX_LIDAR_RANGE_MM_DISPLAY = 4000  # Max Lidar range to consider for display/mapping

# Occupancy grid dynamics
HIT_INCREMENT = 25  # How much to increase cell value on Lidar hit
DECAY_RATE = 0.90  # How fast cell values decay (0.85 is faster, 0.95 is slower)
MAX_CELL_VALUE = 150  # Max value a cell can reach
RENDER_THRESHOLD = 40  # Cell values >= this are considered obstacles for display/planning
CLEAR_BELOW_VALUE = 10  # Cell values < this are reset to 0 during decay

# Update rate
DISPLAY_UPDATE_MS = 70  # GUI refresh rate (milliseconds). Lower is faster but more CPU.

# --- Navigation States ---
NAV_STATE_IDLE = "IDLE"
NAV_STATE_PLANNING = "PLANNING"
NAV_STATE_AWAITING_NEXT_STEP = "AWAITING_NEXT_STEP"  # Waiting for action timer or next decision
NAV_STATE_TURNING_LEFT = "TURNING_LEFT"
NAV_STATE_TURNING_RIGHT = "TURNING_RIGHT"
NAV_STATE_MOVING_FORWARD = "MOVING_FORWARD"
NAV_STATE_REACHED_GOAL = "REACHED_GOAL"
NAV_STATE_PATH_BLOCKED = "PATH_BLOCKED"
NAV_STATE_NO_PATH_FOUND = "NO_PATH_FOUND"
NAV_STATE_ROBOT_CMD_ERROR = "ROBOT_CMD_ERROR"  # Error sending command to robot

# --- Motion Durations ---
# CRITICAL: This duration should be slightly longer than the time it takes your ROBOT
# to complete a single 'F' (move CELL_SIZE_MM), 'L' (90deg turn), or 'R' (90deg turn) action.
# After this duration, an 'S' (STOP) command is sent.
ACTION_DURATION_S = 1.2  # <<<< USER: CALIBRATE THIS with your robot's physical movement time.


class LidarMapViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("RPLidar Networked Map Viewer (4x4m Navigation with Control)")
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.configure(bg=BG_COLOR)

        self.occupancy_grid = np.zeros((GRID_DIMENSION, GRID_DIMENSION), dtype=np.float32)
        self.grid_center_x_cell = GRID_DIMENSION // 2
        self.grid_center_y_cell = GRID_DIMENSION // 2
        self.data_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent excessive memory if processing lags

        self.udp_socket = None
        self.udp_thread = None
        self.is_udp_running = False
        self.listener_ip_ref = "N/A"  # IP of the Lidar data source (RPi)
        self.listener_port = DEFAULT_LISTENER_PORT

        # Navigation attributes
        self.navigation_goal_cell = None  # (row, col) of the target
        self.current_path = []  # List of (row, col) tuples from A*
        # current_logical_robot_cell_on_path: Where the script *thinks* the robot is.
        # This is THE CENTER OF THE MAP because Lidar data is always relative to Lidar.
        self.current_logical_robot_cell_on_path = (self.grid_center_y_cell, self.grid_center_x_cell)
        self.navigation_state = NAV_STATE_IDLE
        # robot_orientation_map_deg: Robot's logical orientation on the map.
        # 0=Up(North), 90=Right(East), 180=Down(South), 270=Left(West).
        self.robot_orientation_map_deg = 0.0  # Initial orientation: Up
        self.current_action_info = {"text": "Idle", "end_time": 0}  # For timed actions
        self.action_timer_id = None  # For tk.after cancellation
        self.path_planner_thread = None
        self.last_command_sent_successful = True  # Track success of F/L/R commands

        self._setup_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(DISPLAY_UPDATE_MS, self._periodic_update)

        self.last_data_time = time.time()
        self.received_packets = 0
        self.points_in_last_scan = 0
        self.map_update_counter = 0  # For less frequent full redraw if needed

        self._update_static_status_labels()
        print("--- Welcome to Lidar Map Viewer ---")
        print(f"CRITICAL: Ensure LIDAR_MOUNTING_OFFSET_DEG ({LIDAR_MOUNTING_OFFSET_DEG}°) is correct for your robot!")
        print(
            f"CRITICAL: Ensure ACTION_DURATION_S ({ACTION_DURATION_S}s) matches your robot's movement time for one step/turn!")
        print("Robot 'L'/'R' commands MUST perform ~90 degree turns.")
        print("Robot 'F' command MUST move ~CELL_SIZE_MM forward.")

    def _setup_ui(self):
        # ... (UI setup mostly unchanged)
        control_frame = tk.Frame(self.root, bg=BG_COLOR, pady=5)
        control_frame.pack(fill=tk.X, side=tk.TOP)
        tk.Label(control_frame, text="Lidar Src IP:", font=DEFAULT_FONT, bg=BG_COLOR, fg=TEXT_COLOR_DARK).pack(
            side=tk.LEFT, padx=(10, 0))
        self.ip_entry = tk.Entry(control_frame, font=DEFAULT_FONT, width=15)
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        self.ip_entry.insert(0, "127.0.0.1")
        self.connect_button = tk.Button(control_frame, text="Connect", font=BUTTON_FONT, bg=ACCENT_COLOR, fg="white",
                                        command=self._toggle_connection, relief=tk.RAISED, bd=2, padx=10)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        self.connection_status_label = tk.Label(control_frame, text="Status: Disconnected", font=DEFAULT_FONT,
                                                bg=BG_COLOR, fg="red")
        self.connection_status_label.pack(side=tk.LEFT, padx=10)

        nav_status_frame = tk.Frame(self.root, bg=BG_COLOR)
        nav_status_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))
        tk.Label(nav_status_frame, text="Nav Action:", font=DEFAULT_FONT, bg=BG_COLOR, fg=TEXT_COLOR_DARK).pack(
            side=tk.LEFT, padx=(10, 5))
        self.nav_action_label = tk.Label(nav_status_frame, text="Idle",
                                         font=(DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE, "bold"), bg=BG_COLOR,
                                         fg=TEXT_COLOR_DARK, width=50, anchor="w")  # Increased width
        self.nav_action_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        main_content_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 10))
        canvas_outer_frame = tk.Frame(main_content_frame, bg=ACCENT_COLOR, bd=1, relief=tk.SUNKEN)
        canvas_outer_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))
        # Calculate map display size based on available space
        self.map_display_size_px = min(APP_WIDTH - 230, APP_HEIGHT - 160)  # Adjusted for padding/widget sizes
        self.cell_display_size_px = self.map_display_size_px / GRID_DIMENSION
        self.cell_display_size_px = max(1.0, self.cell_display_size_px)  # Ensure at least 1px
        self.canvas = tk.Canvas(canvas_outer_frame, width=int(self.map_display_size_px),
                                height=int(self.map_display_size_px), bg=CANVAS_BG_COLOR, highlightthickness=0)
        self.canvas.pack(padx=2, pady=2, anchor="center", expand=True)
        self.canvas.bind("<Button-1>", self._handle_map_click)

        stats_frame = tk.Frame(main_content_frame, bg=BG_COLOR, width=200)  # Slightly wider stats
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
        stats_frame.pack_propagate(False)
        tk.Label(stats_frame, text="--- Status ---", font=(DEFAULT_FONT_FAMILY, 11, "bold"), bg=BG_COLOR,
                 fg=TEXT_COLOR_DARK).pack(pady=5, anchor="w")
        self.status_labels = {}
        status_keys = ["grid_info", "map_size", "cell_size_disp", "lidar_offset", "action_dur", "last_pkt_time",
                       "pkt_count", "points_last", "robot_ip"]
        status_descs = ["Grid:", "Map Area:", "Cell Disp:", "Lidar Mount:", "Action Time:", "Last Data:", "Packets Rx:",
                        "Points/Pkt:", "Robot IP:"]
        for i, key in enumerate(status_keys):
            s_frame = tk.Frame(stats_frame, bg=BG_COLOR)
            s_frame.pack(fill="x", pady=1)  # Reduced pady
            desc_label = tk.Label(s_frame, text=status_descs[i], font=DEFAULT_FONT, bg=BG_COLOR, fg=TEXT_COLOR_DARK,
                                  width=11, anchor="w")  # Adjusted width
            desc_label.pack(side=tk.LEFT)
            self.status_labels[key] = tk.Label(s_frame, text="N/A", font=DEFAULT_FONT, bg=BG_COLOR, fg=TEXT_COLOR_DARK,
                                               anchor="w")
            self.status_labels[key].pack(side=tk.LEFT, fill="x", expand=True)

    def _update_static_status_labels(self):
        self.status_labels["grid_info"].config(text=f"{GRID_DIMENSION}x{GRID_DIMENSION} ({CELL_SIZE_MM}mm)")
        self.status_labels["map_size"].config(text=f"{MAP_SIZE_METERS:.1f}x{MAP_SIZE_METERS:.1f} m")
        self.status_labels["robot_ip"].config(text=f"{ROBOT_COMMAND_IP}:{ROBOT_COMMAND_PORT}")
        self.status_labels["lidar_offset"].config(text=f"{LIDAR_MOUNTING_OFFSET_DEG}°")
        self.status_labels["action_dur"].config(text=f"{ACTION_DURATION_S}s")
        self.status_labels["cell_size_disp"].config(text=f"{self.cell_display_size_px:.2f} px")

    def send_command(self, command):
        # ... (send_command method logic remains largely the same)
        if not self.is_udp_running and command != 'S':
            print(f"WARN: Lidar stream not connected. Sending '{command}' blind. Only 'S' is generally safe.")

        if self.navigation_state == NAV_STATE_IDLE and command not in ['S']:
            print(f"INFO: Command '{command}' not sent: Navigation is IDLE. Only 'S' allowed.")
            return False

        ip = ROBOT_COMMAND_IP
        port = ROBOT_COMMAND_PORT
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(ROBOT_COMMAND_TIMEOUT_S)
                sock.connect((ip, port))
                sock.send(command.encode())
            print(f"ROBOT_CMD: Sent '{command}' to {ip}:{port}")
            self.last_command_sent_successful = True
            return True
        except socket.timeout:
            print(f"ROBOT_CMD_FAIL: Timeout sending '{command}' to {ip}:{port}")
            self._handle_command_send_error(command)
            return False
        except ConnectionRefusedError:
            print(f"ROBOT_CMD_FAIL: Connection refused for '{command}' to {ip}:{port}")
            self._handle_command_send_error(command)
            return False
        except Exception as e:
            print(f"ROBOT_CMD_FAIL: Error sending '{command}': {e}")
            self._handle_command_send_error(command)
            return False

    def _handle_command_send_error(self, command_that_failed=""):
        self.last_command_sent_successful = False
        if self.action_timer_id:
            self.root.after_cancel(self.action_timer_id)
            self.action_timer_id = None
        # Only transition to ROBOT_CMD_ERROR if not already in a terminal state
        if self.navigation_state not in [NAV_STATE_IDLE, NAV_STATE_REACHED_GOAL, NAV_STATE_NO_PATH_FOUND]:
            self.navigation_state = NAV_STATE_ROBOT_CMD_ERROR
            self._update_nav_action_label(f"Robot Cmd Err ({command_that_failed})! Check robot.", "red")
        print(f"ERROR: Robot command '{command_that_failed}' failed. Navigation may be compromised.")

    def _toggle_connection(self):
        if self.is_udp_running:
            # Try to stop robot if it was moving due to navigation
            if self.navigation_state not in [NAV_STATE_IDLE, NAV_STATE_REACHED_GOAL, NAV_STATE_NO_PATH_FOUND,
                                             NAV_STATE_ROBOT_CMD_ERROR]:
                print("INFO: Disconnecting Lidar stream. Sending STOP to robot.")
                self.send_command('S')

            self._stop_udp_listener()  # Stops the UDP thread and closes socket
            self.navigation_state = NAV_STATE_IDLE
            self.navigation_goal_cell = None
            self.current_path = []
            # self.current_logical_robot_cell_on_path remains center
            self.robot_orientation_map_deg = 0.0  # Reset orientation on disconnect
            self._update_nav_action_label("Idle (Lidar Disconnected)")
            if self.action_timer_id:
                self.root.after_cancel(self.action_timer_id)
                self.action_timer_id = None
        else:
            ip_from_user = self.ip_entry.get().strip()
            if not ip_from_user:
                messagebox.showerror("Error", "Please enter a Lidar source (e.g., RPi) IP address.")
                return
            self.listener_ip_ref = ip_from_user
            self._start_udp_listener("0.0.0.0", self.listener_port)  # Listen on all interfaces for Lidar data

    def _start_udp_listener(self, bind_ip, port):
        if self.is_udp_running: return
        try:
            self.occupancy_grid.fill(0)  # Clear grid on new connection
            self.received_packets = 0
            self.points_in_last_scan = 0
            self.last_data_time = time.time()
            if hasattr(self, 'canvas'): self.canvas.delete("all")
            # self.current_logical_robot_cell_on_path is always map center
            self.robot_orientation_map_deg = 0.0  # Reset orientation on new connection

            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((bind_ip, port))
            self.udp_socket.settimeout(1.0)  # Timeout for recvfrom
            print(f"LIDAR_NET: Listening on UDP {bind_ip}:{port} (expecting data from: {self.listener_ip_ref})")
            self.is_udp_running = True
            self.udp_thread = threading.Thread(target=self._udp_receive_loop, daemon=True)
            self.udp_thread.start()
            self.connect_button.config(text="Disconnect")
            self.connection_status_label.config(text=f"Status: Listening on {port}...", fg="green")
            self.ip_entry.config(state=tk.DISABLED)
            self._update_nav_action_label("Idle (Lidar Connected)")
        except Exception as e:
            print(f"LIDAR_NET_ERROR: Could not start UDP listener: {e}")
            messagebox.showerror("Connection Error", f"Could not start listener: {e}")
            self.connection_status_label.config(text="Status: Error", fg="red")
            self.is_udp_running = False

    def _stop_udp_listener(self):
        if not self.is_udp_running: return
        self.is_udp_running = False  # Signal thread to stop
        if self.udp_thread and self.udp_thread.is_alive():
            self.udp_thread.join(timeout=1.5)  # Wait for thread to finish
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None
        self.connect_button.config(text="Connect")
        self.connection_status_label.config(text="Status: Disconnected", fg="red")
        self.ip_entry.config(state=tk.NORMAL)
        print("LIDAR_NET: UDP listener stopped.")

    def _udp_receive_loop(self):
        print("LIDAR_NET: UDP receive thread started.")
        while self.is_udp_running:
            try:
                raw_data, addr = self.udp_socket.recvfrom(65535)  # Buffer size
                if raw_data:
                    try:
                        # Drop older packets if queue is full to prioritize fresh data
                        if self.data_queue.full():
                            self.data_queue.get_nowait()  # Discard oldest
                        self.data_queue.put_nowait(raw_data.decode('utf-8'))
                    except queue.Full:
                        print(
                            "LIDAR_NET_WARN: Data queue full, packet dropped.")  # Should be rare with get_nowait above
                    except UnicodeDecodeError:
                        print("LIDAR_NET_WARN: Received non-UTF-8 data, packet dropped.")
            except socket.timeout:
                continue  # Normal if no data arrives
            except Exception as e:
                if self.is_udp_running:  # Only log if we are supposed to be running
                    print(f"LIDAR_NET_ERROR: UDP Receive error: {e}")
                    time.sleep(0.01)  # Brief pause on error
        print("LIDAR_NET: UDP receive thread ended.")

    def _process_queued_data(self):
        if not self.is_udp_running and self.data_queue.empty(): return  # No connection and no backlog

        new_data_processed_this_cycle = False
        try:
            # Process a limited number of packets per cycle to keep GUI responsive
            # This is a simple form of load shedding if Lidar data rate is very high.
            packets_to_process_this_cycle = 2  # Process up to 2 full scans per GUI update
            for _ in range(packets_to_process_this_cycle):
                if self.data_queue.empty():
                    break
                data_str = self.data_queue.get_nowait()
                self.received_packets += 1
                self.points_in_last_scan = data_str.count(';') + 1 if data_str else 0
                self.last_data_time = time.time()
                self._update_occupancy_grid(data_str)
                new_data_processed_this_cycle = True
        except queue.Empty:
            pass  # Normal if queue becomes empty during processing

        # Decay map if no new Lidar data was processed in this cycle
        # (or if UDP is off but there was old data in queue that got processed)
        if not new_data_processed_this_cycle and self.is_udp_running:  # Only decay if connected and no new data
            self.occupancy_grid *= DECAY_RATE
            self.occupancy_grid[self.occupancy_grid < CLEAR_BELOW_VALUE] = 0.0

    def _polar_to_cartesian_map_frame(self, angle_deg_raw, distance_mm):
        # angle_deg_raw: Raw angle from Lidar (Lidar's 0 degrees).
        # LIDAR_MOUNTING_OFFSET_DEG: Corrects Lidar's 0 to robot's physical FRONT.
        # self.robot_orientation_map_deg: Robot's current orientation on the map (0=Up, 90=Right, etc).

        angle_relative_to_robot_front_deg = (angle_deg_raw + LIDAR_MOUNTING_OFFSET_DEG) % 360.0
        angle_in_map_coords_deg = (angle_relative_to_robot_front_deg + self.robot_orientation_map_deg) % 360.0
        angle_in_map_coords_rad = math.radians(angle_in_map_coords_deg)

        # Map coordinates: +X is Right, +Y is UP (standard math, but screen Y is down)
        x_mm = distance_mm * math.sin(angle_in_map_coords_rad)  # sin for X when 0 deg is Up
        y_mm = distance_mm * math.cos(angle_in_map_coords_rad)  # cos for Y when 0 deg is Up
        return x_mm, y_mm

    def _update_occupancy_grid(self, points_data_str):
        # Apply decay once before adding new hits for this scan packet
        self.occupancy_grid *= DECAY_RATE
        self.occupancy_grid[self.occupancy_grid < CLEAR_BELOW_VALUE] = 0.0

        if not points_data_str: return

        for point_str in points_data_str.split(';'):
            try:
                parts = point_str.split(',')
                if len(parts) != 2: continue  # Skip malformed
                angle_deg_raw, dist_mm = float(parts[0]), float(parts[1])

                if not (0 < dist_mm <= MAX_LIDAR_RANGE_MM_DISPLAY): continue

                x_map_mm, y_map_mm = self._polar_to_cartesian_map_frame(angle_deg_raw, dist_mm)

                # Convert mm map coordinates to grid cell indices
                # Robot is at grid_center_x_cell, grid_center_y_cell
                # Positive x_map_mm goes right, positive y_map_mm goes up (on map)
                grid_col = int(self.grid_center_x_cell + (x_map_mm / CELL_SIZE_MM) + 0.5)  # Add 0.5 for rounding
                grid_row = int(self.grid_center_y_cell - (
                            y_map_mm / CELL_SIZE_MM) + 0.5)  # -y because grid rows increase downwards, +0.5 for rounding

                if 0 <= grid_row < GRID_DIMENSION and 0 <= grid_col < GRID_DIMENSION:
                    current_val = self.occupancy_grid[grid_row, grid_col]
                    self.occupancy_grid[grid_row, grid_col] = min(current_val + HIT_INCREMENT, MAX_CELL_VALUE)
            except ValueError:
                # print(f"MAP_WARN: Skipping malformed Lidar point: {point_str}")
                continue
            except Exception as e:
                print(f"MAP_ERROR: Processing Lidar point {point_str}: {e}")

    def _handle_map_click(self, event):
        if not self.is_udp_running:
            messagebox.showinfo("Not Connected", "Connect to Lidar stream first to enable navigation.");
            return

        # Check if navigation is busy with an action that shouldn't be interrupted lightly
        busy_states = [NAV_STATE_TURNING_LEFT, NAV_STATE_TURNING_RIGHT, NAV_STATE_MOVING_FORWARD, NAV_STATE_PLANNING]
        if self.navigation_state in busy_states:
            if messagebox.askyesno("Navigation Busy", "Navigation is active. Cancel current action and set new goal?"):
                print("NAV: User cancelled current action. Sending STOP to robot.")
                self.send_command('S')
                if self.action_timer_id: self.root.after_cancel(self.action_timer_id); self.action_timer_id = None
                self.navigation_state = NAV_STATE_IDLE  # Go to idle before planning new
                self.current_path = []
                # Keep current orientation and logical position for new plan
                self._update_nav_action_label("Idle (User Cancelled Action)")
            else:
                return  # User chose not to cancel

        # Convert click coordinates to grid cell
        clicked_col = int(event.x / self.cell_display_size_px)
        clicked_row = int(event.y / self.cell_display_size_px)

        if 0 <= clicked_row < GRID_DIMENSION and 0 <= clicked_col < GRID_DIMENSION:
            if self.occupancy_grid[clicked_row, clicked_col] >= RENDER_THRESHOLD:
                messagebox.showwarning("Invalid Goal", "Selected goal is in an Cccupied/Blocked cell. Choose another.");
                return

            self.navigation_goal_cell = (clicked_row, clicked_col)
            print(f"NAV: New goal set by map click: {self.navigation_goal_cell}")

            # Reset path and state for new planning
            self.current_path = []
            self.navigation_state = NAV_STATE_PLANNING
            # Path planning uses current_logical_robot_cell_on_path (map center) and current robot_orientation_map_deg
            self._update_nav_action_label(f"Planning path to ({clicked_row},{clicked_col})...")

            if self.path_planner_thread and self.path_planner_thread.is_alive():
                print(
                    "NAV_WARN: Path planner already running. New request will be processed after current one (if state allows).")
                # This behavior is okay; the state change to PLANNING will be picked up.
                return
            self.path_planner_thread = threading.Thread(target=self._plan_path_threaded, daemon=True)
            self.path_planner_thread.start()
        else:
            print(f"NAV_INFO: Map click ({event.x}, {event.y}) was outside grid boundaries.")

    def _update_nav_action_label(self, text, color=None):
        # ... (no changes)
        self.nav_action_label.config(text=text)
        default_fg = TEXT_COLOR_DARK
        if self.navigation_state == NAV_STATE_REACHED_GOAL:
            default_fg = "green"
        elif self.navigation_state in [NAV_STATE_NO_PATH_FOUND, NAV_STATE_PATH_BLOCKED, NAV_STATE_ROBOT_CMD_ERROR]:
            default_fg = "red"
        elif self.navigation_state not in [NAV_STATE_IDLE, NAV_STATE_AWAITING_NEXT_STEP]:
            default_fg = ACCENT_COLOR  # Active states
        self.nav_action_label.config(fg=color if color else default_fg)

    def _is_valid_and_clear(self, r, c):
        # Check bounds and if cell is below obstacle threshold
        return 0 <= r < GRID_DIMENSION and \
            0 <= c < GRID_DIMENSION and \
            self.occupancy_grid[r, c] < RENDER_THRESHOLD

    def _heuristic(self, cell, goal):  # Manhattan distance
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    def _plan_path_threaded(self):
        # A* Path Planning
        # The robot's starting cell for planning is always its logical position on the map,
        # which is the map center, as Lidar data is relative.
        # The path generated is a sequence of cells relative to this fixed robot origin.
        start_cell = self.current_logical_robot_cell_on_path  # This is (grid_center_y, grid_center_x)

        # Ensure goal is still valid before starting intensive planning
        current_goal_cell = self.navigation_goal_cell
        if not current_goal_cell:
            print("PLANNER: No goal cell defined at planning start.")
            self.root.after(0, self._path_planning_failed, "Goal not set.")
            return
        if not self._is_valid_and_clear(current_goal_cell[0], current_goal_cell[1]):
            print(f"PLANNER: Goal cell {current_goal_cell} is blocked or invalid at planning start.")
            self.root.after(0, self._path_planning_failed, "Goal blocked.")
            return

        print(
            f"PLANNER: Starting A* from {start_cell} to {current_goal_cell} (Robot orientation: {self.robot_orientation_map_deg:.1f}°)")

        open_set = []  # Priority queue: (f_score, count, cell)
        import heapq
        count = 0  # Tie-breaker for heapq
        heapq.heappush(open_set, (self._heuristic(start_cell, current_goal_cell), count, start_cell))

        came_from = {}  # Stores parent of each cell in path
        g_score = {cell: float('inf') for r in range(GRID_DIMENSION) for cell_c in range(GRID_DIMENSION) for cell in
                   [(r, cell_c)]}
        g_score[start_cell] = 0

        f_score = {cell: float('inf') for r in range(GRID_DIMENSION) for cell_c in range(GRID_DIMENSION) for cell in
                   [(r, cell_c)]}
        f_score[start_cell] = self._heuristic(start_cell, current_goal_cell)

        open_set_hash = {start_cell}  # For quick check if cell is in open_set

        path_found = False
        max_iterations = GRID_DIMENSION * GRID_DIMENSION * 1.5  # Safety break for A*
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            current_f_val, _, current_cell = heapq.heappop(open_set)
            open_set_hash.remove(current_cell)

            if current_cell == current_goal_cell:
                path_found = True
                break

            # Explore neighbors (Up, Down, Left, Right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current_cell[0] + dr, current_cell[1] + dc)

                if self._is_valid_and_clear(neighbor[0], neighbor[1]):
                    tentative_g_score = g_score[current_cell] + 1  # Cost of 1 for each step

                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current_cell
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, current_goal_cell)
                        if neighbor not in open_set_hash:
                            count += 1
                            heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                            open_set_hash.add(neighbor)

        if iterations >= max_iterations:
            print("PLANNER_WARN: A* reached max iterations.")

        if path_found:
            # Reconstruct path
            path = []
            temp = current_goal_cell
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            # path.append(start_cell) # Optional: include start cell
            path.reverse()  # Path from start to goal
            if path and path[0] == start_cell: path.pop(0)  # A* often includes start, remove if so.
            self.root.after(0, self._path_planning_succeeded, path)
        else:
            self.root.after(0, self._path_planning_failed, "No path found by A*.")

    def _path_planning_succeeded(self, path):
        self.current_path = path
        if self.current_path:
            print(
                f"PLANNER_SUCCESS: Path found (len {len(self.current_path)}). From {self.current_logical_robot_cell_on_path} to {self.navigation_goal_cell}. Path: {self.current_path}")
            self.navigation_state = NAV_STATE_AWAITING_NEXT_STEP
            self._update_nav_action_label("Path found. Executing...")
        else:  # Path is empty
            if self.navigation_goal_cell == self.current_logical_robot_cell_on_path:  # Already at goal
                self.navigation_state = NAV_STATE_REACHED_GOAL
                self._update_nav_action_label("Already at goal!", "green")
                print("PLANNER_INFO: Already at goal. Sending STOP.")
                self.send_command('S')
            else:  # Path planning resulted in empty path but not at goal (should be rare)
                print(
                    f"PLANNER_WARN: Path planning yielded empty path but not at goal {self.navigation_goal_cell}. Logical pos: {self.current_logical_robot_cell_on_path}")
                self.navigation_state = NAV_STATE_NO_PATH_FOUND
                self._update_nav_action_label("Path planning error.", "red")

    def _path_planning_failed(self, reason="Unknown"):
        print(f"PLANNER_FAIL: Path planning failed: {reason}")
        # Do not send 'S' here, as no motion was initiated from this failed plan
        self.navigation_state = NAV_STATE_NO_PATH_FOUND
        # self.navigation_goal_cell = None # Keep goal visible for user context
        self.current_path = []  # Clear any partial path
        self._update_nav_action_label(f"No path: {reason}", "red")

    def _execute_navigation_step(self):
        # This is the main navigation state machine driver
        # It decides the next robot command based on current_path and robot_orientation_map_deg

        # States where no action should be taken here
        if self.navigation_state in [NAV_STATE_IDLE, NAV_STATE_PLANNING, NAV_STATE_REACHED_GOAL,
                                     NAV_STATE_NO_PATH_FOUND, NAV_STATE_PATH_BLOCKED, NAV_STATE_ROBOT_CMD_ERROR]:
            return

        # If an action (F, L, R) is currently "in progress" (waiting for ACTION_DURATION_S)
        if time.time() < self.current_action_info.get("end_time", 0):
            return  # Wait for action timer to complete

        # At this point, any previous timed action has completed (and 'S' was sent).
        # We are in NAV_STATE_AWAITING_NEXT_STEP or a motion state (which should transition to AWAITING_NEXT_STEP)

        current_robot_pos_cell = self.current_logical_robot_cell_on_path  # This is always map center

        # Check if path is consumed or empty
        if not self.current_path:
            if self.navigation_goal_cell and current_robot_pos_cell == self.navigation_goal_cell:  # Should have been caught by planner
                print("NAV_EXEC: Path empty, at goal (logic check).")
                self.navigation_state = NAV_STATE_REACHED_GOAL
                self._update_nav_action_label("Goal Reached!", "green")
                # No 'S' here, _on_action_completed or planner success should handle it
            else:  # Path ended, but not at goal (e.g. path cleared by blockage, or error)
                print("NAV_EXEC: Path empty, not at goal. Current state: " + self.navigation_state)
                if self.navigation_state != NAV_STATE_PATH_BLOCKED:  # Avoid redundant message if already path_blocked
                    self.navigation_state = NAV_STATE_NO_PATH_FOUND  # Or some other error state
                    self._update_nav_action_label("Path ended unexpectedly.", "orange")
            # self.navigation_goal_cell = None # Clear goal only when truly done or failed completely
            self.current_path = []
            return

        # Get next cell from path
        next_cell_in_world = self.current_path[0]  # This is relative to map center
        print(
            f"NAV_EXEC: Current logical pos: {current_robot_pos_cell}, Orientation: {self.robot_orientation_map_deg:.1f}°, Next path cell: {next_cell_in_world}")

        # Check if next cell is blocked (dynamic obstacle appeared after planning)
        if not self._is_valid_and_clear(next_cell_in_world[0], next_cell_in_world[1]):
            print(
                f"NAV_EXEC_WARN: Path blocked at {next_cell_in_world}! Map value: {self.occupancy_grid[next_cell_in_world[0], next_cell_in_world[1]]}. Sending STOP and replanning.")
            # STOP command is crucial here as robot might have been moving
            self.send_command('S')
            self._update_nav_action_label("Path Blocked! Replanning...", "orange")
            self.navigation_state = NAV_STATE_PATH_BLOCKED  # Indicate blockage
            # Set to PLANNING to trigger replan. Keep goal. Clear current path as it's invalid.
            self.current_path = []  # Clear bad path before replanning

            # Trigger replan immediately if planner not busy
            if not (self.path_planner_thread and self.path_planner_thread.is_alive()):
                self.navigation_state = NAV_STATE_PLANNING  # Switch to planning state
                self.path_planner_thread = threading.Thread(target=self._plan_path_threaded, daemon=True)
                self.path_planner_thread.start()
            else:
                print("NAV_EXEC_INFO: Planner busy, replan for blockage will occur after.")
            return

        # Determine required movement: dr, dc are relative to robot's current position (map center)
        # Since current_robot_pos_cell is the map center, dr/dc calculation simplifies.
        # Example: if next_cell_in_world is (center_y - 1, center_x), dr = -1, dc = 0 (Move Up)
        dr = next_cell_in_world[0] - current_robot_pos_cell[0]
        dc = next_cell_in_world[1] - current_robot_pos_cell[1]

        target_orientation_deg = -1.0  # Target orientation to reach the next cell

        if dr == -1 and dc == 0:
            target_orientation_deg = 0.0  # Move Up (North on map)
        elif dr == 1 and dc == 0:
            target_orientation_deg = 180.0  # Move Down (South)
        elif dr == 0 and dc == -1:
            target_orientation_deg = 270.0  # Move Left (West)
        elif dr == 0 and dc == 1:
            target_orientation_deg = 90.0  # Move Right (East)
        elif dr == 0 and dc == 0:  # Next step in path is current logical position (shouldn't happen if path planned correctly)
            print("NAV_EXEC_WARN: Path step is current logical location. Consuming & re-evaluating.")
            self.current_path.pop(0)  # Consume this redundant step
            self.navigation_state = NAV_STATE_AWAITING_NEXT_STEP  # Re-evaluate immediately
            if self.action_timer_id: self.root.after_cancel(self.action_timer_id)
            # No physical robot command. Use a very short timer or direct call.
            self.action_timer_id = self.root.after(10,
                                                   self._on_action_completed_no_command)  # Simulate quick completion
            return
        else:  # Diagonal or invalid step, current planner only does cardinal
            self._update_nav_action_label(f"Invalid path step d({dr},{dc})", "red")
            print(
                f"NAV_EXEC_ERROR: Invalid path step dr={dr}, dc={dc}. From {current_robot_pos_cell} to {next_cell_in_world}. Sending STOP.")
            self.send_command('S')
            self.navigation_state = NAV_STATE_PATH_BLOCKED  # Treat as error
            self.current_path = []  # Clear path
            return

        action_text = ""
        command_to_send = None  # 'F', 'L', 'R'
        action_end_time = time.time() + ACTION_DURATION_S

        # Compare current robot orientation with target orientation
        orientation_diff = (target_orientation_deg - self.robot_orientation_map_deg + 360.0) % 360.0

        # Tolerance for float comparisons of angles
        angle_tolerance = 1.0  # degrees
        needs_turn = True
        if abs(orientation_diff) < angle_tolerance or abs(orientation_diff - 360.0) < angle_tolerance:
            needs_turn = False  # Already facing target or close enough

        print(
            f"NAV_DECISION: Target orient: {target_orientation_deg:.1f}, Current orient: {self.robot_orientation_map_deg:.1f}, Diff: {orientation_diff:.1f}. Needs turn: {needs_turn}")

        if not needs_turn:  # Robot is (assumed to be) facing the correct direction for the next cell
            self.navigation_state = NAV_STATE_MOVING_FORWARD
            action_text = f"Forward to {next_cell_in_world} (Map Cell)..."
            command_to_send = 'F'
            # Logical position update (popping from path) happens AFTER command success
        else:  # Robot needs to turn
            # This logic assumes 'L' and 'R' robot commands perform ~90 degree turns.
            # If your robot turns differently, this control logic needs adjustment.
            if (orientation_diff > (90.0 - angle_tolerance) and orientation_diff < (90.0 + angle_tolerance)) or \
                    (orientation_diff > (180.0 - angle_tolerance) and orientation_diff < (
                            180.0 + angle_tolerance) and command_to_send is None):  # Prefer R for 180 if first choice
                # Turn Right (90 deg), or first part of 180 deg turn
                self.navigation_state = NAV_STATE_TURNING_RIGHT
                command_to_send = 'R'
                if orientation_diff > (180.0 - angle_tolerance) and orientation_diff < (180.0 + angle_tolerance):
                    action_text = f"Turn Right (1/2 of 180° to {target_orientation_deg:.0f}°)..."
                else:
                    action_text = f"Turn Right (to {target_orientation_deg:.0f}°)..."

            elif orientation_diff > (270.0 - angle_tolerance) and orientation_diff < (270.0 + angle_tolerance):
                # Turn Left (effectively -90 deg)
                self.navigation_state = NAV_STATE_TURNING_LEFT
                command_to_send = 'L'
                action_text = f"Turn Left (to {target_orientation_deg:.0f}°)..."

            elif orientation_diff > (180.0 - angle_tolerance) and orientation_diff < (
                    180.0 + angle_tolerance) and command_to_send is None:  # Should be caught by R first
                # This case is for a 180-degree turn.
                # We'll make two 90-degree turns (e.g., Right then Right).
                # This execution step handles the FIRST turn.
                self.navigation_state = NAV_STATE_TURNING_RIGHT  # Arbitrarily pick Right for the first 90 deg
                action_text = f"Turn Right (1/2 of 180° to {target_orientation_deg:.0f}°)..."
                command_to_send = 'R'
            else:  # Unexpected orientation difference.
                # This might happen if robot turns are not exactly 90 deg.
                action_text = f"ERR:Unhandled turn: {orientation_diff:.1f}° to reach {target_orientation_deg:.0f}°"
                print(
                    f"NAV_EXEC_ERROR: {action_text}. Current orientation {self.robot_orientation_map_deg:.1f}°. Sending STOP.")
                self.send_command('S')
                self.navigation_state = NAV_STATE_PATH_BLOCKED  # Treat as path issue
                self.current_path = []
                return

        # Send the determined command to the robot
        if command_to_send:
            print(
                f"NAV_COMMANDING: State: {self.navigation_state}, Command: '{command_to_send}', Action: {action_text}")
            if self.send_command(command_to_send):  # If command sent successfully (TCP ack)
                # Update logical state based on the command sent
                if self.navigation_state == NAV_STATE_MOVING_FORWARD:
                    # Assume robot moved one cell forward. Path is relative to robot's Lidar frame.
                    # The 'next_cell_in_world' effectively becomes the new 'current_robot_pos_cell' (map center)
                    # after the world "shifts" due to robot's movement.
                    # So, we just pop the cell from the path.
                    self.current_path.pop(0)
                    print(f"NAV_LOGIC: Moved forward. Path remaining: {len(self.current_path)}")
                elif self.navigation_state == NAV_STATE_TURNING_LEFT:
                    self.robot_orientation_map_deg = (self.robot_orientation_map_deg - 90.0 + 360.0) % 360.0
                    print(f"NAV_LOGIC: Turned Left. New orientation: {self.robot_orientation_map_deg:.1f}°")
                elif self.navigation_state == NAV_STATE_TURNING_RIGHT:
                    self.robot_orientation_map_deg = (self.robot_orientation_map_deg + 90.0) % 360.0
                    print(f"NAV_LOGIC: Turned Right. New orientation: {self.robot_orientation_map_deg:.1f}°")

                self.current_action_info = {"text": action_text, "end_time": action_end_time}
                self._update_nav_action_label(action_text)
                if self.action_timer_id: self.root.after_cancel(self.action_timer_id)
                # Start timer for ACTION_DURATION_S, after which _on_action_completed sends 'S'
                self.action_timer_id = self.root.after(int(ACTION_DURATION_S * 1000), self._on_action_completed)
            else:
                # send_command failed and should have set NAV_STATE_ROBOT_CMD_ERROR
                # _handle_command_send_error was called by send_command
                print(
                    f"NAV_EXEC_ERROR: Command '{command_to_send}' failed to send. Navigation should be in error state.")
                # No further action needed here; error state is set by _handle_command_send_error.
        else:
            # This case should ideally not be reached if logic is correct
            print(
                "NAV_EXEC_ERROR: No command determined for navigation step. This is unexpected. State: " + self.navigation_state)
            self.navigation_state = NAV_STATE_PATH_BLOCKED  # Fallback to a safe error state
            self.send_command('S')  # Send stop just in case

    def _on_action_completed(self):
        """Called after ACTION_DURATION_S for a physical F/L/R robot command."""
        self.action_timer_id = None  # Clear the timer ID

        if not self.last_command_sent_successful:
            # The F/L/R command itself failed. _handle_command_send_error already set state.
            print(
                "NAV_TIMER: Action timer expired, but last F/L/R command failed. Not sending STOP. State should be ROBOT_CMD_ERROR.")
            return

        print(
            f"NAV_TIMER: Action duration ({ACTION_DURATION_S}s) for '{self.current_action_info.get('text', 'Unknown')}' ended. Sending STOP.")

        # Send 'S' (STOP) command to the robot.
        # If this 'S' command fails, _handle_command_send_error will be called.
        stop_success = self.send_command('S')

        if not stop_success:
            print("NAV_TIMER_ERROR: STOP command FAILED after action. Navigation state is now ROBOT_CMD_ERROR.")
            # _handle_command_send_error has updated the state.
            return  # Do not proceed to AWAITING_NEXT_STEP if STOP failed.

        # If STOP was successful, and we are in a state that expects further steps:
        if self.navigation_state in [NAV_STATE_MOVING_FORWARD, NAV_STATE_TURNING_LEFT, NAV_STATE_TURNING_RIGHT,
                                     NAV_STATE_AWAITING_NEXT_STEP]:
            # Check if path is now empty (goal reached after this step)
            if not self.current_path and self.navigation_goal_cell == self.current_logical_robot_cell_on_path:
                # This condition means the 'F' move just completed was to the goal cell
                self.navigation_state = NAV_STATE_REACHED_GOAL
                self._update_nav_action_label("Goal Reached!", "green")
                print("NAV_TIMER: Goal reached after final move/turn.")
                # self.navigation_goal_cell = None # Clear goal once reached
            else:
                self.navigation_state = NAV_STATE_AWAITING_NEXT_STEP
                self._update_nav_action_label("Awaiting next step...")
        # If already in a terminal (REACHED_GOAL, NO_PATH, etc.) or error state, do nothing further here.
        # The _execute_navigation_step will be called again in the next periodic update if state is AWAITING_NEXT_STEP.

    def _on_action_completed_no_command(self):
        """Called after a logical-only step (no physical robot command sent)."""
        self.action_timer_id = None
        # No 'S' command needed here.
        if self.navigation_state not in [NAV_STATE_REACHED_GOAL, NAV_STATE_PATH_BLOCKED, NAV_STATE_NO_PATH_FOUND,
                                         NAV_STATE_PLANNING, NAV_STATE_IDLE, NAV_STATE_ROBOT_CMD_ERROR]:
            self.navigation_state = NAV_STATE_AWAITING_NEXT_STEP
            # print("NAV_TIMER_LOGIC: Logical step completed. Awaiting next.")

    def _draw_map(self):
        self.canvas.delete("all")
        # Draw occupancy grid cells
        for r_idx in range(GRID_DIMENSION):
            for c_idx in range(GRID_DIMENSION):
                cell_value = self.occupancy_grid[r_idx, c_idx]
                if cell_value >= RENDER_THRESHOLD:  # Only draw "occupied" cells
                    # Calculate color based on cell_value intensity
                    denominator = (MAX_CELL_VALUE - RENDER_THRESHOLD)
                    denominator = 1 if denominator <= 0 else denominator  # Avoid division by zero
                    intensity_ratio = min(max(0, (cell_value - RENDER_THRESHOLD) / denominator), 1.0)

                    color_val = [int(GRID_COLOR_LOW[i] + intensity_ratio * (GRID_COLOR_HIGH[i] - GRID_COLOR_LOW[i])) for
                                 i in range(3)]
                    color_val = [max(0, min(255, val)) for val in color_val]  # Clamp to 0-255
                    color_hex = f"#{color_val[0]:02x}{color_val[1]:02x}{color_val[2]:02x}"

                    # Calculate screen coordinates for the cell
                    screen_x = c_idx * self.cell_display_size_px
                    screen_y = r_idx * self.cell_display_size_px
                    # Ensure cells are at least 1x1 px for visibility, and cover the area
                    cell_draw_w = math.ceil(self.cell_display_size_px)
                    cell_draw_h = math.ceil(self.cell_display_size_px)

                    self.canvas.create_rectangle(screen_x, screen_y,
                                                 screen_x + cell_draw_w, screen_y + cell_draw_h,
                                                 fill=color_hex, outline="")  # No outline for speed

        # Draw navigation goal 'G'
        if self.navigation_goal_cell:
            gr, gc = self.navigation_goal_cell
            gx_center = gc * self.cell_display_size_px + self.cell_display_size_px / 2
            gy_center = gr * self.cell_display_size_px + self.cell_display_size_px / 2
            radius = self.cell_display_size_px / 2.5  # Slightly smaller than cell
            # Dashed circle for goal
            self.canvas.create_oval(gx_center - radius, gy_center - radius,
                                    gx_center + radius, gy_center + radius,
                                    outline="blue", width=max(1, int(self.cell_display_size_px / 12)), dash=(3, 3))
            # Text "G"
            self.canvas.create_text(gx_center, gy_center, text="G", fill="blue",
                                    font=(DEFAULT_FONT_FAMILY, int(max(7, self.cell_display_size_px * 0.6)), "bold"))

        # Draw current path
        if self.current_path:
            path_pixel_coords = []
            # Path starts from robot's current logical position (center of map)
            start_px_x = self.current_logical_robot_cell_on_path[
                             1] * self.cell_display_size_px + self.cell_display_size_px / 2
            start_px_y = self.current_logical_robot_cell_on_path[
                             0] * self.cell_display_size_px + self.cell_display_size_px / 2
            path_pixel_coords.extend([start_px_x, start_px_y])

            # Add subsequent cells in the path
            for r_cell, c_cell in self.current_path:
                path_pixel_coords.extend([c_cell * self.cell_display_size_px + self.cell_display_size_px / 2,
                                          r_cell * self.cell_display_size_px + self.cell_display_size_px / 2])

            if len(path_pixel_coords) >= 4:  # Need at least two points for a line
                self.canvas.create_line(path_pixel_coords, fill="purple",
                                        width=max(1, int(self.cell_display_size_px / 10)),
                                        arrow=tk.LAST, dash=(4, 2), capstyle=tk.ROUND)

        # Draw robot icon at the center of the map, oriented by self.robot_orientation_map_deg
        robot_screen_x = self.grid_center_x_cell * self.cell_display_size_px + self.cell_display_size_px / 2
        robot_screen_y = self.grid_center_y_cell * self.cell_display_size_px + self.cell_display_size_px / 2

        car_icon_base_size = self.cell_display_size_px
        # Scale icon size: make it larger if cells are small, but not too tiny
        car_icon_size = max(10, car_icon_base_size * 1.5 if car_icon_base_size >= 7 else car_icon_base_size * 2.5)

        self._draw_car_icon(self.canvas, robot_screen_x, robot_screen_y, car_icon_size, self.robot_orientation_map_deg)

    def _draw_car_icon(self, canvas, x_c, y_c, size, angle_deg=0):
        # Defines a simple car-like shape with a "windshield"
        # Body points (counter-clockwise) - origin is center of robot
        body_width_ratio = 0.7
        body_height_ratio = 1.0

        bw = size * body_width_ratio
        bh = size * body_height_ratio
        hbw = bw / 2  # half body width
        hbh = bh / 2  # half body height

        # Polygon for car body (shape: wide rear, tapered front)
        # Front is at -Y direction (0 deg points "up" on canvas)
        body_points_local = [
            (-hbw, hbh * 0.7),  # Rear-left
            (hbw, hbh * 0.7),  # Rear-right
            (hbw * 0.8, -hbh * 0.5),  # Mid-front-right
            (0, -hbh),  # Front-center (nose)
            (-hbw * 0.8, -hbh * 0.5),  # Mid-front-left
        ]
        # Polygon for "windshield" (smaller, towards front)
        windshield_points_local = [
            (-hbw * 0.5, hbh * 0.1),
            (hbw * 0.5, hbh * 0.1),
            (hbw * 0.4, -hbh * 0.3),
            (-hbw * 0.4, -hbh * 0.3),
        ]

        rad = math.radians(angle_deg)  # Convert angle to radians for trig functions
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)

        def rotate_and_translate(local_points):
            transformed = []
            for lx, ly in local_points:
                # Rotation:
                # x' = x*cos(theta) - y*sin(theta)
                # y' = x*sin(theta) + y*cos(theta)
                # Screen Y is inverted for drawing, but local points are standard math (+Y up)
                # The angle_deg is map orientation (0=Up), so positive rotation is counter-clockwise.
                # Here, a positive angle_deg should rotate the icon clockwise on screen because
                # the car icon's "front" is defined as -Y local.
                # Or, more simply, use the standard rotation matrix and let the angle_deg dictate it.
                # angle_deg = 0 -> icon points UP on map.
                # angle_deg = 90 -> icon points RIGHT on map.

                # Standard rotation for coordinates where +angle is CCW:
                # x_rot = lx * cos_rad - ly * sin_rad
                # y_rot = lx * sin_rad + ly * cos_rad
                # However, our icon's "front" points towards local -Y.
                # And map's 0-degrees is "UP".
                # If angle_deg = 0, icon should point UP. Our local -Y should become map +Y (screen -Y).
                # If angle_deg = 90, icon should point RIGHT. Our local -Y should become map -X (screen -X).
                # Let's use the direct definition of the car icon where its "front" (nose) is at (0, -hbh) in its local frame.
                # angle_deg = 0 means it points up.
                rotated_x = lx * cos_rad - ly * sin_rad
                rotated_y = lx * sin_rad + ly * cos_rad

                # Translation to canvas coordinates
                transformed.extend([x_c + rotated_x, y_c + rotated_y])  # Screen Y is typically downwards from top-left
            return transformed

        final_body_points = rotate_and_translate(body_points_local)
        final_windshield_points = rotate_and_translate(windshield_points_local)

        canvas.create_polygon(final_body_points, fill=ROBOT_BODY_COLOR, outline=ROBOT_ACCENT_COLOR,
                              width=max(1, int(size / 15)))
        canvas.create_polygon(final_windshield_points, fill=ROBOT_ACCENT_COLOR, outline=ROBOT_BODY_COLOR,
                              width=max(1, int(size / 18)))

    def _update_dynamic_status_labels(self):
        # Update labels that change frequently
        if not self.is_udp_running and self.received_packets == 0:
            self.status_labels["last_pkt_time"].config(text="N/A", fg=TEXT_COLOR_DARK)
            self.status_labels["pkt_count"].config(text="0")
            self.status_labels["points_last"].config(text="0")
            return

        time_since_last_data = time.time() - self.last_data_time
        color_data_freshness = "green"
        if not self.is_udp_running:
            color_data_freshness = "gray"  # Disconnected
        elif time_since_last_data > 15:
            color_data_freshness = "red"  # Very stale
        elif time_since_last_data > 5:
            color_data_freshness = "orange"  # Stale

        last_pkt_text = "N/A"
        if self.received_packets > 0:
            last_pkt_text = f"{time_since_last_data:.1f}s ago"

        self.status_labels["last_pkt_time"].config(text=last_pkt_text, fg=color_data_freshness)
        self.status_labels["pkt_count"].config(text=f"{self.received_packets}")
        self.status_labels["points_last"].config(text=f"{self.points_in_last_scan}")

    def _periodic_update(self):
        # Main loop called by tk.after

        # 1. Process any queued Lidar data to update the occupancy grid
        self._process_queued_data()

        # 2. Execute next step in navigation state machine if conditions met
        #    (e.g., connected, not idle, not planning, action timer not active)
        if self.is_udp_running:  # Only navigate if Lidar is connected
            self._execute_navigation_step()

        # 3. Redraw the map display (occupancy grid, robot, path, goal)
        #    Can optimize by redrawing less frequently if map updates are slow
        # self.map_update_counter +=1
        # if self.map_update_counter % 2 == 0: # Example: redraw every other cycle
        self._draw_map()

        # 4. Update dynamic status labels (packet counts, data freshness)
        self._update_dynamic_status_labels()

        # 5. Schedule the next periodic update
        self.root.after(DISPLAY_UPDATE_MS, self._periodic_update)

    def _on_close(self):
        print("--- Application Shutting Down ---")
        # Attempt to stop robot if it was navigating
        if self.is_udp_running and self.navigation_state not in [NAV_STATE_IDLE, NAV_STATE_REACHED_GOAL,
                                                                 NAV_STATE_NO_PATH_FOUND, NAV_STATE_ROBOT_CMD_ERROR]:
            print("SYS_CLOSE: Sending STOP to robot as a precaution.")
            self.send_command('S')

        # Clean up resources
        if self.action_timer_id:
            self.root.after_cancel(self.action_timer_id)
            self.action_timer_id = None

        # Signal UDP thread to stop and wait for it
        self._stop_udp_listener()  # This now handles thread joining

        if self.path_planner_thread and self.path_planner_thread.is_alive():
            print("SYS_CLOSE_WARN: Path planner thread still active. Will be terminated by daemon exit.")
            # Daemon threads exit when main program exits.

        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass  # Tkinter might already be gone
        print("--- Application Closed ---")


if __name__ == '__main__':
    root = tk.Tk()
    app = LidarMapViewer(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected by main thread. Closing application.")
        app._on_close()
    except Exception as e:
        print(f"FATAL_ERROR: Unhandled exception in Tk main loop: {e}")
        import traceback

        traceback.print_exc()
        app._on_close()  # Attempt graceful shutdown