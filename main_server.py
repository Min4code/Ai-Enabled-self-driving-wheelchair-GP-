import tkinter as tk
import threading
import time
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

import serial
import serial.tools.list_ports
import sys
import glob
import socket  # Already present

matplotlib.use("TkAgg")

try:
    from rplidar import RPLidar, RPLidarException
except ImportError:
    print("RPLidar library not found. Install with: pip install rplidar-roboticia")
    RPLidar = None
    RPLidarException = None

# --- Device Identification ---
RPLIDAR_VID_PIDS = [(0x10C4, 0xEA60), (0x0403, 0x6001)]
RPLIDAR_DESCRIPTIONS = ['cp210x', 'rplidar', 'slamtec', 'ft232r usb uart', 'usb serial']
RPLIDAR_GLOB_PATTERNS = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/serial/by-id/*CP210*',
                         '/dev/serial/by-id/*USB_SERIAL*', '/dev/serial/by-id/*Silicon_Labs_CP210x*',
                         '/dev/serial/by-id/*FTDI_FT232R_USB_UART*']


def send_command(command):
    ip = "127.0.0.1"
    port = 9000
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout
        sock.connect((ip, port))
        sock.send(command.encode())
        sock.close()
        print(f"✓ Sent command: {command}")
    except socket.timeout:
        print(f"✗ Timeout connecting to {ip}:{port}")
    except ConnectionRefusedError:
        print(f"✗ Connection refused to {ip}:{port} - Is the server running?")
    except Exception as e:
        print(f"✗ Error sending command {command}: {e}")


class RPLidarCarControl:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.LIDAR_PORT_NAME = None

        self.lidar = None
        self.all_objects_detected = []
        self.angles = np.array([])  # For internal viz, uses offset
        self.distances = np.array([])  # For internal viz
        self.is_scanning = False
        self.scan_thread = None
        self.lidar_reconnect_thread = None
        self.is_reconnecting_lidar = False

        self.MAX_RANGE_MM = 2000  # Max range for internal visualization and object detection
        self.MIN_OBJECT_SIZE_POINTS = 3
        self.SCAN_INTERVAL = 0.2  # Interval for processing scans and updating UI
        self.MAX_BUF_MEAS = 300
        self.SCAN_POINT_SUBSAMPLE_RATIO = 3  # For internal processing/visualization
        self.LIDAR_ANGLE_OFFSET_DEG = 180.0  # For internal visualization consistency

        self.FRONT_ANGLE_CFG_DEG = 25
        self.PROXIMITY_THRESHOLD_MM = 350
        self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM = self.PROXIMITY_THRESHOLD_MM

        self.car_locked = False
        self.any_object_in_front_details = None

        self.is_fullscreen = True
        self.window.attributes('-fullscreen', self.is_fullscreen)
        self.window.bind('<Escape>', self.toggle_fullscreen)

        self.bg_color = "#212121";
        self.accent_color = "#3498db";
        self.text_color = "#FFFFFF"
        self.button_color = "#333333";
        self.button_hover_color = "#4CAF50"
        self.warning_color = "#e74c3c";
        self.caution_color = "#f39c12";
        self.info_color = "#2ecc71"

        self.window.configure(bg=self.bg_color)
        self._setup_ui_layout()

        self.ser = None  # For Arduino, if used directly (send_command uses socket)

        # --- UDP Streaming Configuration ---
        self.UDP_STREAM_ENABLED = True  # Master switch for UDP streaming
        self.UDP_RECEIVER_IP = '192.168.250.41'
        self.UDP_RECEIVER_PORT = 9876
        self.UDP_SEND_RATE_HZ = 4.0  # Target send rate
        self.UDP_MAX_POINTS_TO_SEND = 90  # Max points per UDP packet
        self.UDP_MIN_QUALITY_FOR_SEND = 10  # Min Lidar point quality for UDP
        self.UDP_MAX_DISTANCE_FOR_SEND = 4000  # mm, max distance for points in UDP stream

        self.udp_socket = None
        self.last_udp_send_time = 0
        self.udp_send_interval = 0

        if self.UDP_STREAM_ENABLED:
            if self.UDP_SEND_RATE_HZ > 0:
                self.udp_send_interval = 1.0 / self.UDP_SEND_RATE_HZ
            else:
                # If send rate is 0 or invalid, effectively disable rate limiting (send every scan)
                # or could be set to float('inf') to disable if not desired.
                self.udp_send_interval = 0

            try:
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(
                    f"✓ UDP socket created for Lidar data streaming to {self.UDP_RECEIVER_IP}:{self.UDP_RECEIVER_PORT}")
            except socket.error as e:
                print(f"✗ Error creating UDP socket: {e}. Disabling UDP streaming.")
                self.UDP_STREAM_ENABLED = False
            except Exception as e:
                print(f"✗ Unexpected error creating UDP socket: {e}. Disabling UDP streaming.")
                self.UDP_STREAM_ENABLED = False
        # --- End UDP Streaming Configuration ---

        self.initialize_rplidar_with_reconnect_logic()

        self.window.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.DEBUG_FRONT_DETECTION = False

    def _send_lidar_data_udp(self, raw_scan_data_tuples):
        """
        Sends Lidar scan data via UDP.
        raw_scan_data_tuples: list of (quality, angle_degrees_raw, distance_mm)
        """
        if not self.UDP_STREAM_ENABLED or not self.udp_socket:
            return

        current_time = time.time()
        if current_time - self.last_udp_send_time < self.udp_send_interval:
            return  # Not time to send yet, respecting UDP_SEND_RATE_HZ

        points_to_send = []
        # raw_scan_data_tuples is a list of (quality, angle_raw_degrees, distance_mm)
        for quality, angle_raw_deg, dist_mm in raw_scan_data_tuples:
            if quality >= self.UDP_MIN_QUALITY_FOR_SEND and dist_mm > 0:
                # Cap distance for UDP stream, use raw angle from Lidar
                capped_dist_mm = min(dist_mm, self.UDP_MAX_DISTANCE_FOR_SEND)
                points_to_send.append((angle_raw_deg, capped_dist_mm))

        if not points_to_send:
            self.last_udp_send_time = current_time  # Update time to maintain rate even if no points sent
            return

        # Select a subset if too many points (takes the first N points)
        selected_points = points_to_send[:self.UDP_MAX_POINTS_TO_SEND]

        # Format message: "angle1,dist1;angle2,dist2;..."
        # Angles are raw from Lidar (0-360), distances in mm.
        msg_parts = [f"{angle:.1f},{distance:.0f}" for angle, distance in selected_points]
        msg_payload = ";".join(msg_parts)

        try:
            self.udp_socket.sendto(msg_payload.encode('utf-8'), (self.UDP_RECEIVER_IP, self.UDP_RECEIVER_PORT))
            # For debugging:
            # print(f"✓ UDP: Sent {len(selected_points)} Lidar pts ({len(msg_payload)} bytes) to {self.UDP_RECEIVER_IP}:{self.UDP_RECEIVER_PORT}")
        except socket.error as se:
            print(f"✗ UDP send error to {self.UDP_RECEIVER_IP}:{self.UDP_RECEIVER_PORT}: {se}")
        except Exception as e:
            print(f"✗ Unexpected error during UDP send: {e}")
        finally:
            self.last_udp_send_time = current_time

    def _test_rplidar_port(self, port_to_test):
        if RPLidar is None: return False
        try:
            test_lidar = RPLidar(port_to_test, timeout=1)
            info = test_lidar.get_info()
            test_lidar.disconnect()
            return 'model' in info and info['model'] is not None
        except:
            return False

    def _find_device_port(self, target_vid_pids, target_descriptions, device_name_log, exclude_port=None,
                          glob_patterns=None):
        try:
            ports_info = serial.tools.list_ports.comports()
        except Exception as e:
            print(f"  Error getting serial ports: {e}");
            return None
        if not ports_info: print("  No serial ports found."); return None

        for p_info in ports_info:
            if p_info.device == exclude_port: continue
            if p_info.vid is not None and p_info.pid is not None and (p_info.vid, p_info.pid) in target_vid_pids:
                if device_name_log == "RPLidar" and self._test_rplidar_port(p_info.device):
                    print(f"  Found {device_name_log} by VID/PID: {p_info.device}");
                    return p_info.device
                elif device_name_log != "RPLidar":
                    print(f"  Found {device_name_log} by VID/PID: {p_info.device}");
                    return p_info.device
        for p_info in ports_info:
            if p_info.device == exclude_port: continue
            desc_lower = p_info.description.lower() if p_info.description else ""
            if any(keyword in desc_lower for keyword in target_descriptions):
                if device_name_log == "RPLidar" and self._test_rplidar_port(p_info.device):
                    print(f"  Found {device_name_log} by description: {p_info.device}");
                    return p_info.device
                elif device_name_log != "RPLidar":
                    print(f"  Found {device_name_log} by description: {p_info.device}");
                    return p_info.device
        if device_name_log == "RPLidar" and glob_patterns and RPLidar:
            possible_ports = []
            for pattern in glob_patterns: possible_ports.extend(glob.glob(pattern))
            for port_path in sorted(list(set(possible_ports))):
                if port_path == exclude_port or not any(p.device == port_path for p in ports_info): continue
                if self._test_rplidar_port(port_path):
                    print(f"  Found {device_name_log} by glob: {port_path}");
                    return port_path
        common_ports = []
        if device_name_log == "RPLidar":
            common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0']  # Common fallback for Linux
        for port_path in common_ports:
            if any(p.device == port_path for p in ports_info) and port_path != exclude_port:
                if device_name_log == "RPLidar" and self._test_rplidar_port(port_path):
                    print(f"  Found {device_name_log} by common (fallback): {port_path}");
                    return port_path
                elif device_name_log != "RPLidar":  # For other devices if used
                    print(f"  Found {device_name_log} by common (fallback): {port_path}");
                    return port_path
        return None

    def _initialize_lidar_port_only(self):
        print("--- Initializing Lidar Port ---")
        self.LIDAR_PORT_NAME = self._find_device_port(
            RPLIDAR_VID_PIDS, RPLIDAR_DESCRIPTIONS, "RPLidar",
            glob_patterns=RPLIDAR_GLOB_PATTERNS
        )
        if not self.LIDAR_PORT_NAME:
            print("!! WARNING: RPLidar port not found during re-scan. Will keep trying if in reconnect loop.")
        print("--- Lidar Port Initialization Complete ---")

    def _setup_serial_connection(self, port, baudrate=9600, timeout=1, device_name="Device"):
        # This method is for direct serial connections if needed, not used by send_command (socket)
        if port is None: print(f"No port for {device_name}."); return None
        try:
            ser_conn = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Connected to {device_name} on {port}")
            return ser_conn
        except serial.SerialException as e:
            print(f"Error connecting to {device_name} on {port}: {e}");
            return None

    def _setup_ui_layout(self):
        header_frame = tk.Frame(self.window, bg=self.bg_color, height=40);
        header_frame.pack(fill=tk.X, pady=(5, 2));
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="RPLIDAR NAVIGATOR (RPi)", font=("Arial", 16, "bold"), bg=self.bg_color,
                 fg=self.accent_color).pack(side=tk.TOP, pady=5)
        main_frame = tk.Frame(self.window, bg=self.bg_color);
        main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        main_frame.grid_rowconfigure(0, weight=1);
        main_frame.grid_columnconfigure(0, weight=2);
        main_frame.grid_columnconfigure(1, weight=1)
        lidar_frame = tk.Frame(main_frame, bg=self.bg_color);
        lidar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        lidar_border = tk.Frame(lidar_frame, bg=self.accent_color, bd=1, relief=tk.RAISED, padx=2, pady=2);
        lidar_border.pack(expand=True, fill=tk.BOTH, pady=5)
        self.setup_lidar_visualization(lidar_border)
        controls_frame = tk.Frame(main_frame, bg=self.bg_color);
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        controls_frame.grid_rowconfigure(0, weight=1);
        controls_frame.grid_columnconfigure(0, weight=1)
        btn_frame = tk.Frame(controls_frame, bg=self.bg_color);
        btn_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.create_control_buttons(btn_frame)
        self.object_info_label = tk.Label(self.window, text="Initializing...", font=("Arial", 10, "bold"),
                                          bg=self.bg_color, fg=self.info_color)
        self.object_info_label.pack(fill=tk.X, pady=(2, 2), side=tk.BOTTOM)
        self.create_status_panel(self.window)

    def setup_lidar_visualization(self, parent_frame):
        self.fig = Figure(figsize=(3, 3), dpi=70, facecolor=self.bg_color);
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_facecolor('black');
        self.ax.set_title('RPLidar Scan', color=self.text_color, fontsize=9)  # Will be updated in plot method
        self.ax.grid(color='gray', linestyle=':', alpha=0.6);
        self.ax.set_ylim(0, self.MAX_RANGE_MM)
        self.ax.set_theta_zero_location("N");
        self.ax.set_theta_direction('counterclockwise')
        self.ax.tick_params(axis='x', colors=self.text_color, labelsize=7);
        self.ax.tick_params(axis='y', colors=self.text_color, labelsize=7)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame);
        self.canvas.draw();
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.scan_info_text = self.fig.text(0.02, 0.02, "Initializing Lidar...", color=self.text_color, fontsize=7,
                                            va='bottom')

    def create_control_buttons(self, parent_frame):
        for i in range(3): parent_frame.grid_columnconfigure(i, weight=1,
                                                             uniform="dpad_col");parent_frame.grid_rowconfigure(i,
                                                                                                                weight=1,
                                                                                                                uniform="dpad_row")
        self.btn_forward = self._create_styled_button(parent_frame, "▲", 0, 1, self.forward_pressed,
                                                      self.forward_released)
        self.btn_left = self._create_styled_button(parent_frame, "◄", 1, 0, self.left_pressed, self.left_released)
        self.btn_stop_manual = self._create_styled_button(parent_frame, "■", 1, 1, self.stop_pressed_released,
                                                          self.stop_pressed_released)  # Same for press/release
        self.btn_right = self._create_styled_button(parent_frame, "►", 1, 2, self.right_pressed, self.right_released)
        self.btn_backward = self._create_styled_button(parent_frame, "▼", 2, 1, self.backward_pressed,
                                                       self.backward_released)

    def _create_styled_button(self, parent, text, r, c, press_func, release_func):
        btn = tk.Button(parent, text=text, font=("Arial", 14, "bold"), bg=self.button_color, fg=self.text_color,
                        activebackground=self.button_hover_color, activeforeground=self.text_color, relief=tk.RAISED,
                        bd=2)
        btn.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
        btn.bind("<Enter>", lambda e, b=btn: self.button_hover(b, True));
        btn.bind("<Leave>", lambda e, b=btn: self.button_hover(b, False))
        if press_func == release_func:  # For buttons like STOP that act on press
            btn.bind("<Button-1>", press_func)
        else:  # For press-and-hold buttons
            btn.bind("<Button-1>", press_func);
            btn.bind("<ButtonRelease-1>", release_func)
        return btn

    def create_status_panel(self, parent_window):
        status_frame = tk.Frame(parent_window, bg="#1a1a1a", height=30);
        status_frame.pack(side=tk.BOTTOM, fill=tk.X);
        status_frame.pack_propagate(False)
        self.status_label = tk.Label(status_frame, text="System Ready", fg=self.text_color, bg="#1a1a1a",
                                     font=("Arial", 9, "bold"));
        self.status_label.pack(side=tk.LEFT, padx=10)
        lidar_status_subframe = tk.Frame(status_frame, bg="#1a1a1a");
        lidar_status_subframe.pack(side=tk.RIGHT, padx=10)
        self.lidar_status_label = tk.Label(lidar_status_subframe, text="LIDAR: Initializing...", fg=self.text_color,
                                           bg="#1a1a1a", font=("Arial", 8));
        self.lidar_status_label.pack(side=tk.LEFT)
        self.lidar_indicator = tk.Canvas(lidar_status_subframe, width=10, height=10, bg="#1a1a1a",
                                         highlightthickness=0);
        self.lidar_indicator.pack(side=tk.LEFT, padx=(5, 0))
        self.lidar_indicator.create_oval(1, 1, 9, 9, fill="#f39c12", outline="#f39c12")  # Initializing color

    def initialize_rplidar_with_reconnect_logic(self):
        if self.is_reconnecting_lidar or (self.lidar_reconnect_thread and self.lidar_reconnect_thread.is_alive()):
            print("Lidar reconnection already in progress.")
            return

        if RPLidar is None:
            self.update_lidar_status("Library Missing", "red")
            self.update_scan_info_text("RPLidar library missing. DEMO MODE.")
            self.run_demo_mode()  # Fallback to demo if library is missing
            return

        if not self.LIDAR_PORT_NAME:
            self._initialize_lidar_port_only()

        if not self.LIDAR_PORT_NAME:
            self.update_lidar_status("Port Not Found", "red")
            self.update_scan_info_text("RPLidar port not found. Will attempt reconnect or DEMO.")
            self.attempt_lidar_reconnect(initial_attempt=True)  # This will run demo if reconnect fails
            return

        # If port found, attempt to connect
        if not self._connect_and_start_lidar_scan():
            # If initial connection fails, attempt reconnect which might lead to demo mode
            self.attempt_lidar_reconnect(initial_attempt=True)

    def _connect_and_start_lidar_scan(self):
        if not self.LIDAR_PORT_NAME:
            print("Lidar port name not set, cannot connect.")
            self.update_lidar_status("Port Invalid", self.warning_color)
            return False

        self.update_lidar_status(f"Connecting ({self.LIDAR_PORT_NAME.split('/')[-1]})", self.caution_color)
        self.update_scan_info_text(f"Connecting to RPLidar on {self.LIDAR_PORT_NAME}...")

        try:
            if self.lidar:  # Ensure any old instance is disconnected
                try:
                    self.lidar.disconnect()
                except:
                    pass
                self.lidar = None

            self.lidar = RPLidar(self.LIDAR_PORT_NAME, baudrate=115200, timeout=3)  # Baudrate common for many RPLidars
            info = self.lidar.get_info()
            health = self.lidar.get_health()
            print(f"RPLidar Info: {info}\nHealth: {health}")
            self.update_scan_info_text(
                f"Connected: {info.get('model', '?')} S/N:{info.get('serialnumber', 'N/A')} | Health: {health[0]}")

            if health[0].lower() in ['good', 'warning']:  # 'Warning' can sometimes be acceptable
                status_color = self.info_color if health[0].lower() == 'good' else self.caution_color
                self.update_lidar_status(f"Connected ({health[0]})", status_color)

                # Start motor (some models need explicit start)
                if hasattr(self.lidar, 'set_motor_pwm') and info.get('model', '').startswith('A'):  # e.g. A1, A2
                    try:
                        self.lidar.set_motor_pwm(660)  # A common PWM value
                    except Exception as e:
                        print(f"Note: Failed Lidar motor PWM set: {e}")
                if hasattr(self.lidar, 'start_motor'):
                    try:
                        self.lidar.start_motor()
                    except Exception as e:
                        print(f"Note: Failed Lidar motor start: {e}")

                time.sleep(1)  # Give motor a moment to spin up

                self.is_scanning = True
                if self.scan_thread and self.scan_thread.is_alive():
                    print("Warning: Old scan_thread still alive before starting new one.")
                    # Potentially join it, but daemon=True should handle exit
                self.scan_thread = threading.Thread(target=self.main_scan_loop, name="LidarScanThread", daemon=True)
                self.scan_thread.start()
                return True
            else:  # Health is bad
                self.update_lidar_status(f"Health Error: {health[0]}", self.warning_color)
                self.update_scan_info_text(f"RPLidar health: {health[0]}. Check device.")
                if self.lidar: self.lidar.disconnect(); self.lidar = None
                return False
        except RPLidarException as rpe:  # Specific Lidar exceptions
            print(f"RPLidar connection/setup error on {self.LIDAR_PORT_NAME}: {rpe}")
            self.update_lidar_status("Lidar Init Fail", self.warning_color)
            self.update_scan_info_text(f"Lidar Init Error: {rpe}.")
            if self.lidar:
                try:
                    self.lidar.disconnect()
                except:
                    pass
                self.lidar = None
            return False
        except Exception as e:  # Other errors (serial, etc.)
            print(f"General Lidar connection error on {self.LIDAR_PORT_NAME}: {e}")
            self.update_lidar_status("Connection Failed", self.warning_color)
            self.update_scan_info_text(f"Connection Error: {e}.")
            if self.lidar:
                try:
                    self.lidar.disconnect()
                except:
                    pass
                self.lidar = None
            return False

    def main_scan_loop(self):
        loop_start_time = time.time()
        try:
            self.update_scan_info_text("Lidar scan stream active...")
            # scan_gen yields: list of (quality, angle_degrees, distance_mm) tuples for a full 360 scan
            scan_gen = self.lidar.iter_scans(max_buf_meas=self.MAX_BUF_MEAS, min_len=5)

            last_processed_time = 0
            for i, raw_scan_points_tuples in enumerate(scan_gen):
                if not self.is_scanning: break  # Exit if scanning stopped

                current_time_loop = time.time()
                if current_time_loop - last_processed_time < self.SCAN_INTERVAL:
                    # Sleep to maintain approx SCAN_INTERVAL between processing
                    time.sleep(max(0, self.SCAN_INTERVAL - (current_time_loop - last_processed_time)))
                    continue  # Skip this scan iteration to throttle processing
                last_processed_time = current_time_loop

                # Process for internal logic and visualization (applies offset, subsamples)
                self.process_raw_scan_to_points_objects(raw_scan_points_tuples)
                self.detect_objects_in_front_path()  # Uses processed data
                self.manage_autonomous_maneuver()  # Uses detection results
                self.update_lidar_visualization_plot()  # Uses processed data

                # Send raw scan data (before offset/subsampling) via UDP
                if self.UDP_STREAM_ENABLED:
                    self._send_lidar_data_udp(raw_scan_points_tuples)

        except RPLidarException as rpe:
            print(f"RPLidarException in scan loop: {rpe}")
            self.update_lidar_status("Scan Error", self.warning_color)
            self.is_scanning = False;  # Stop scanning flag
            self.attempt_lidar_reconnect()  # Try to reconnect
        except serial.SerialException as se:  # USB disconnect often shows as SerialException
            print(f"SerialException in scan loop (likely Lidar disconnect): {se}")
            self.update_lidar_status("Disconnected", self.warning_color)
            self.is_scanning = False;
            self.attempt_lidar_reconnect()
        except Exception as e:  # Catch-all for other unexpected errors
            print(f"General error in Lidar scan loop: {e}");
            import traceback;
            traceback.print_exc()
            self.update_lidar_status("Runtime Error", self.warning_color)
            self.is_scanning = False
            if RPLidar:  # Only attempt reconnect if library was loaded
                self.attempt_lidar_reconnect()
            else:  # Fallback to demo if RPLidar lib is missing
                self.run_demo_mode()
        print(f"{threading.current_thread().name} finished.")

    def attempt_lidar_reconnect(self, initial_attempt=False):
        if self.is_reconnecting_lidar:
            # print("Reconnect already in progress.")
            return
        self.is_reconnecting_lidar = True
        self.is_scanning = False  # Stop current scan loop if any

        if self.scan_thread and self.scan_thread.is_alive():
            # print("Waiting for existing scan thread to join...")
            self.scan_thread.join(timeout=1.0)  # Wait a bit for it to exit

        if self.lidar:
            try:
                # print("Stopping and disconnecting current Lidar instance for reconnect...")
                self.lidar.stop();
                self.lidar.stop_motor();
                self.lidar.disconnect()
            except Exception as e:
                # print(f"Error stopping/disconnecting Lidar during reconnect attempt: {e}")
                pass  # Continue attempt even if cleanup fails
            self.lidar = None

        self.update_lidar_status("Reconnecting...", self.caution_color)
        self.update_scan_info_text("Attempting Lidar reconnect...")

        if self.lidar_reconnect_thread and self.lidar_reconnect_thread.is_alive():
            # print("Waiting for existing reconnect thread to join...")
            self.lidar_reconnect_thread.join(timeout=1.0)  # Should not happen if logic is correct

        self.lidar_reconnect_thread = threading.Thread(target=self._reconnect_worker,
                                                       name="LidarReconnectThread",
                                                       args=(initial_attempt,), daemon=True)
        self.lidar_reconnect_thread.start()

    def _reconnect_worker(self, initial_attempt=False):
        max_retries = 5 if not initial_attempt else 10;  # More retries on initial setup
        retry_delay_seconds = 5
        print(f"Reconnect worker started. Max retries: {max_retries}")

        for attempt in range(max_retries):
            if not self.is_reconnecting_lidar:  # Flag might be cleared by successful exit_app
                print("Reconnect worker: Reconnection flag cleared, exiting.")
                break

            print(f"Lidar reconnect attempt {attempt + 1}/{max_retries}...")
            self.update_lidar_status(f"Reconnect Att.{attempt + 1}", self.caution_color)

            self._initialize_lidar_port_only()  # Try to find the port again

            if self.LIDAR_PORT_NAME and self._connect_and_start_lidar_scan():
                print("Lidar reconnected successfully.")
                self.is_reconnecting_lidar = False;  # Clear flag on success
                return  # Exit worker thread

            if attempt < max_retries - 1:  # Don't sleep after last attempt
                print(f"Reconnect attempt {attempt + 1} failed. Waiting {retry_delay_seconds}s...")
                for _ in range(retry_delay_seconds):  # Sleep interruptibly
                    if not self.is_reconnecting_lidar: break
                    time.sleep(1)
            if not self.is_reconnecting_lidar:  # Check again after sleep
                print("Reconnect worker: Reconnection flag cleared during sleep, exiting.")
                break

        # If loop finishes without successful reconnect
        print("Lidar reconnect failed after all retries.")
        self.update_lidar_status("Reconnect Failed", self.warning_color)
        self.update_scan_info_text("Lidar reconnect failed. Switching to DEMO MODE.")
        self.is_reconnecting_lidar = False;  # Clear flag
        self.run_demo_mode()  # Fallback to demo mode
        print(f"{threading.current_thread().name} finished.")

    def process_raw_scan_to_points_objects(self, raw_scan_data_tuples):
        # Parameter: list of (quality, angle_raw_degrees, distance_mm)
        proc_angles_rad_list, proc_dists_mm_list = [], []  # For visualization (offset angles)
        detected_obj_list, current_obj_buffer = [], []  # For object detection (offset angles)

        # Subsample for internal processing to reduce load, if configured
        scan_subset_for_internal = raw_scan_data_tuples[::self.SCAN_POINT_SUBSAMPLE_RATIO] \
            if self.SCAN_POINT_SUBSAMPLE_RATIO > 1 else raw_scan_data_tuples

        MIN_QUALITY_INTERNAL = 10  # Min quality for points used in internal object detection/viz

        if self.DEBUG_FRONT_DETECTION and len(scan_subset_for_internal) > 0:
            print(f"--- InternalProcessing: {len(scan_subset_for_internal)} points (after subsample) ---")

        for q, raw_angle_deg, dist_mm in scan_subset_for_internal:
            # Apply Lidar angle offset for internal coordinate system consistency
            lidar_angle_deg_offset = (raw_angle_deg + self.LIDAR_ANGLE_OFFSET_DEG) % 360.0

            # Filter points for internal processing (range, quality)
            if not (0 < dist_mm <= self.MAX_RANGE_MM and q >= MIN_QUALITY_INTERNAL):
                # If an object was being tracked, finalize it
                if len(current_obj_buffer) >= self.MIN_OBJECT_SIZE_POINTS:
                    detected_obj_list.append(list(current_obj_buffer))
                current_obj_buffer.clear()
                continue  # Skip this point

            # For visualization plot (uses offset angles)
            plot_angle_rad_offset = np.radians(lidar_angle_deg_offset)
            proc_angles_rad_list.append(plot_angle_rad_offset)
            proc_dists_mm_list.append(dist_mm)

            # Debug print for points relevant to front detection (uses offset angles)
            if self.DEBUG_FRONT_DETECTION and \
                    ((lidar_angle_deg_offset <= self.FRONT_ANGLE_CFG_DEG) or \
                     (lidar_angle_deg_offset >= (360.0 - self.FRONT_ANGLE_CFG_DEG))) and \
                    dist_mm < self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM * 1.5:  # Check slightly beyond threshold
                print(
                    f"  [ProcScanInternal] Kept: Qual={q}, RawAng={raw_angle_deg:.1f}, Dist={dist_mm:.0f} -> OffsetAng={lidar_angle_deg_offset:.1f}, PlotRad={plot_angle_rad_offset:.2f}")

            # Simple object clustering based on angular proximity (uses offset angles)
            part_of_curr_obj = False
            if current_obj_buffer:
                # Last point in buffer: (quality, lidar_angle_deg_offset, distance_mm)
                last_pt_angle_offset = current_obj_buffer[-1][1]
                angle_diff = abs(last_pt_angle_offset - lidar_angle_deg_offset)
                # Handle angle wrap-around (e.g. 359 deg and 1 deg)
                if min(angle_diff, 360.0 - angle_diff) < 20.0:  # Heuristic: 20 deg max gap
                    part_of_curr_obj = True

            if part_of_curr_obj:
                current_obj_buffer.append((q, lidar_angle_deg_offset, dist_mm))
            else:  # New object starts
                if len(current_obj_buffer) >= self.MIN_OBJECT_SIZE_POINTS:
                    detected_obj_list.append(list(current_obj_buffer))
                current_obj_buffer = [(q, lidar_angle_deg_offset, dist_mm)]  # Start new object

        # Add last tracked object if it meets size criteria
        if len(current_obj_buffer) >= self.MIN_OBJECT_SIZE_POINTS:
            detected_obj_list.append(list(current_obj_buffer))

        self.all_objects_detected = detected_obj_list  # List of lists of (q, offset_angle_deg, dist_mm)
        self.angles = np.array(proc_angles_rad_list)  # For plotting (radians, with offset)
        self.distances = np.array(proc_dists_mm_list)  # For plotting

        if self.DEBUG_FRONT_DETECTION and len(self.angles) > 0:
            print(
                f"  [ProcScanInternal] Populated self.angles (size {len(self.angles)}) and self.distances (size {len(self.distances)})")

    def detect_objects_in_front_path(self):
        self.any_object_in_front_details = None
        closest_point_dist_in_front = float('inf')
        obstacle_point_found_in_cone = False

        # This detection uses self.angles and self.distances, which are already processed
        # (subsampled, offset applied, and filtered by MAX_RANGE_MM & MIN_QUALITY_INTERNAL)
        if self.DEBUG_FRONT_DETECTION and len(self.angles) > 0:
            print(
                f"--- FrontDetection: Cone +/- {self.FRONT_ANGLE_CFG_DEG} deg, StopDist: {self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM}mm ---")
            print(f"  Input points for detection (from self.angles/self.distances): {len(self.angles)}")

        if not (len(self.angles) > 0 and len(self.angles) == len(self.distances)):
            if self.DEBUG_FRONT_DETECTION: print("  FrontDetection: No points or mismatched arrays, skipping.")
            return

        for i in range(len(self.angles)):
            angle_rad_offset = self.angles[i]  # Radians, already has LIDAR_ANGLE_OFFSET_DEG applied
            dist_mm = self.distances[i]

            # Convert angle (which is offset) to degrees (0-360) for the cone check logic
            angle_deg_offset_for_check = math.degrees(angle_rad_offset) % 360.0

            # Check if this point (with offset angle) falls into the robot's front cone
            # Robot's front is 0 degrees in the offset coordinate system.
            is_in_front_cone = (angle_deg_offset_for_check <= self.FRONT_ANGLE_CFG_DEG) or \
                               (angle_deg_offset_for_check >= (360.0 - self.FRONT_ANGLE_CFG_DEG))

            if is_in_front_cone:
                if self.DEBUG_FRONT_DETECTION and dist_mm < self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM * 1.5:
                    print(
                        f"  FrontDetection: Point in cone: Idx={i}, AngleRadOffset={angle_rad_offset:.2f} -> AngleDegOffsetChk={angle_deg_offset_for_check:.1f}, Dist={dist_mm:.0f}mm")

                # Check if this point is an obstacle within the stopping threshold
                if 0 < dist_mm < self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM:
                    obstacle_point_found_in_cone = True
                    if self.DEBUG_FRONT_DETECTION:
                        print(
                            f"    !!! OBSTACLE POINT DETECTED (FrontDetection): AngleDegOffset={angle_deg_offset_for_check:.1f}, Dist={dist_mm:.0f}mm !!!")

                    # Update details if this is the closest obstacle point found so far
                    if dist_mm < closest_point_dist_in_front:
                        closest_point_dist_in_front = dist_mm
                        self.any_object_in_front_details = {
                            'avg_dist': dist_mm,  # Storing single closest point's distance
                            'avg_angle': angle_deg_offset_for_check,  # Angle (with offset) of this point
                            'points': 1  # Placeholder, could be expanded for object cluster
                        }

        if self.DEBUG_FRONT_DETECTION:
            if obstacle_point_found_in_cone and self.any_object_in_front_details:
                print(
                    f"  FrontDetection DECISION: Obstacle FOUND. Closest at {self.any_object_in_front_details['avg_dist']:.0f}mm, Angle (offset) {self.any_object_in_front_details['avg_angle']:.1f}deg")
            elif obstacle_point_found_in_cone and not self.any_object_in_front_details:  # Should not happen
                print(f"  FrontDetection DECISION: Obstacle point in cone, but not recorded in details? LOGIC ERROR.")
            else:  # No obstacle points within threshold in the front cone
                print("  FrontDetection DECISION: No obstacle points found in front cone within stop distance.")

        # Update UI label about path status
        if hasattr(self, 'object_info_label'):
            txt, col = "Path Ahead Clear", self.info_color
            if self.any_object_in_front_details:
                txt = f"FRONT OBSTACLE! Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm"
                col = self.warning_color

            if hasattr(self.window, 'after') and self.window.winfo_exists():
                self.window.after(0, self.object_info_label.config, {"text": txt, "fg": col})

    def manage_autonomous_maneuver(self):
        obstacle_present = self.any_object_in_front_details is not None

        if obstacle_present:
            if not self.car_locked:  # If not already locked, lock it
                # print("AUTONOMOUS: Obstacle detected. Stopping and Locking Forward.")
                self._execute_motor_stop_all_movement()  # Send 'S' (Stop)
                send_command('K')  # Send 'K' (Lock Forward)
                self.car_locked = True
                if hasattr(self.window, 'after') and self.window.winfo_exists() and hasattr(self, 'status_label'):
                    self.window.after(0, lambda: self.status_label.config(
                        text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                        fg=self.warning_color))
        else:  # No obstacle in front
            if self.car_locked:  # If locked, but path is now clear, unlock
                # print("AUTONOMOUS: Path clear. Unlocking Forward.")
                send_command('U')  # Send 'U' to unlock FORWARD
                self.car_locked = False
                if hasattr(self.window, 'after') and self.window.winfo_exists() and hasattr(self, 'status_label'):
                    self.window.after(0, lambda: self.status_label.config(
                        text="Path Clear. CAR UNLOCKED.", fg=self.info_color))

    # Motor control command wrappers
    def _execute_motor_rotate_left(self):
        send_command('L')

    def _execute_motor_rotate_right(self):
        send_command('R')

    def _execute_motor_stop_rotation(self):
        send_command('S')  # Assuming 'S' stops rotation too

    def _execute_motor_stop_all_movement(self):
        send_command('S')

    def _execute_motor_move_forward(self):
        send_command('F')

    def _execute_motor_move_backward(self):
        send_command('B')

    def update_lidar_visualization_plot(self):
        # This method uses self.angles (radians, with offset) and self.distances
        def _do_update():
            try:
                if not (hasattr(self, 'ax') and self.ax and hasattr(self, 'canvas') and self.canvas): return

                self.ax.clear()  # Clear previous plot elements
                self.ax.set_facecolor('black')
                self.ax.set_title(  # Dynamic title
                    f'RPLidar ({self.MAX_RANGE_MM / 1000.0:.1f}m Max, Offset:{self.LIDAR_ANGLE_OFFSET_DEG}°)',
                    color=self.text_color, fontsize=8)
                self.ax.grid(color='gray', linestyle=':', alpha=0.6);
                self.ax.set_ylim(0, self.MAX_RANGE_MM)  # Set Y limit (radius)
                self.ax.set_theta_zero_location("N");  # North (0 deg) at top, robot's front due to offset
                self.ax.set_theta_direction('counterclockwise')
                self.ax.tick_params(axis='x', colors=self.text_color, labelsize=7);  # Angle ticks
                self.ax.tick_params(axis='y', colors=self.text_color, labelsize=7)  # Distance ticks

                # Draw front detection cone (yellow, transparent)
                # Cone is centered at 0 radians because data in self.angles is already offset.
                cone_rad_half = np.radians(self.FRONT_ANGLE_CFG_DEG);
                theta_cone = np.linspace(-cone_rad_half, cone_rad_half, 30)  # Angles for cone polygon
                self.ax.fill_between(theta_cone, 0, self.MAX_RANGE_MM, color='yellow', alpha=0.08, zorder=0)

                # Plot Lidar points if available
                if len(self.angles) > 0 and len(self.distances) > 0:
                    valid_dists = np.maximum(self.distances, 1)  # Avoid issues with log(0) if cmap uses log scale
                    if self.angles.shape == valid_dists.shape:
                        # self.angles are in radians and incorporate the LIDAR_ANGLE_OFFSET_DEG
                        self.ax.scatter(self.angles, valid_dists, c=valid_dists, cmap='cool_r', alpha=0.9, s=3,
                                        zorder=1)  # s is point size
                self.canvas.draw_idle()  # Request Tkinter to redraw the canvas
            except Exception as e:
                # Avoid spamming for common errors during shutdown
                if "invalid command name" not in str(e).lower() and \
                        "application has been destroyed" not in str(e).lower():
                    print(f"Error during Lidar visualization update: {e}")

        # Ensure UI updates happen on the main thread
        if hasattr(self, 'window') and self.window.winfo_exists():
            if threading.current_thread() is not threading.main_thread():
                self.window.after(0, _do_update)
            else:
                _do_update()

    def run_demo_mode(self):
        self.update_scan_info_text("Switched to DEMO MODE.")
        if not (hasattr(self, 'demo_thread') and self.demo_thread and self.demo_thread.is_alive()):
            self.is_scanning = True;  # Enable scanning flag for demo loop
            self.is_reconnecting_lidar = False  # Ensure reconnect flag is off
            self.demo_thread = threading.Thread(target=self.demo_mode_scan_loop, name="DemoScanThread", daemon=True);
            self.demo_thread.start()
            self.update_lidar_status("Demo Mode Active", "#007bff")  # Blue color for demo status
        else:
            print("Demo mode already running or thread exists.")

    def demo_mode_scan_loop(self):
        print("Demo mode scan loop started.")
        obstacle_active_in_demo = False;
        toggle_tm = time.time()

        while self.is_scanning:  # Loop while scanning is active
            curr_tm = time.time()
            if curr_tm - toggle_tm > 6:  # Toggle simulated obstacle state every 6 seconds
                obstacle_active_in_demo = not obstacle_active_in_demo
                toggle_tm = curr_tm
                # print(f"Demo mode: Obstacle active: {obstacle_active_in_demo}")

            num_pts_demo = 200;  # Number of points in a simulated 360 scan
            sim_raw_scan_tuples = []  # List of (quality, raw_angle_deg, distance_mm)

            for i in range(num_pts_demo):
                # Simulate raw angle from Lidar (0-360, no offset initially)
                raw_angle_sim_deg = (i / num_pts_demo) * 360.0
                base_dist = self.MAX_RANGE_MM * 0.9  # Default distance

                # For demo obstacle, its position should appear in the robot's front.
                # Calculate what raw Lidar angle would correspond to the robot's front after offset.
                # If LIDAR_ANGLE_OFFSET_DEG = 180, an obstacle at raw_angle_sim_deg = 180
                # would become 0 deg after offset, i.e., robot's front.
                # So, check if raw_angle_sim_deg is near (0 - LIDAR_ANGLE_OFFSET_DEG) % 360

                # Effective angle if offset were applied (for demo logic to place obstacle in robot's front)
                angle_for_demo_obstacle_check = (raw_angle_sim_deg + self.LIDAR_ANGLE_OFFSET_DEG) % 360.0

                is_in_robot_front_cone_for_demo = \
                    (angle_for_demo_obstacle_check <= self.FRONT_ANGLE_CFG_DEG) or \
                    (angle_for_demo_obstacle_check >= (360.0 - self.FRONT_ANGLE_CFG_DEG))

                if obstacle_active_in_demo and is_in_robot_front_cone_for_demo:
                    base_dist = self.ANY_OBSTACLE_IN_FRONT_FOR_STOP_MM * 0.5  # Simulate close obstacle

                # Add some noise to the distance
                dist = np.clip(base_dist + np.random.uniform(-30, 30), 50, self.MAX_RANGE_MM)
                # Simulate: quality=15 (good), raw Lidar angle, distance
                sim_raw_scan_tuples.append((15, raw_angle_sim_deg, dist))

            # Process simulated raw data for internal logic (applies offset, subsamples, etc.)
            self.process_raw_scan_to_points_objects(sim_raw_scan_tuples)
            self.detect_objects_in_front_path()  # Uses processed data
            self.manage_autonomous_maneuver()
            self.update_lidar_visualization_plot()  # Uses processed data

            # Send simulated raw scan data (before offset) via UDP
            if self.UDP_STREAM_ENABLED:
                self._send_lidar_data_udp(sim_raw_scan_tuples)

            time.sleep(self.SCAN_INTERVAL)  # Adhere to scan interval
        print(f"{threading.current_thread().name} (Demo) finished.")

    def update_scan_info_text(self, text):
        if hasattr(self, 'scan_info_text') and self.scan_info_text:
            try:
                def _upd():  # Closure to update text
                    self.scan_info_text.set_text(text);
                    self.canvas.draw_idle()

                if threading.current_thread() is threading.main_thread():
                    _upd()  # Already on main thread
                elif hasattr(self, 'window') and self.window.winfo_exists():
                    self.window.after(0, _upd)  # Schedule update on main thread
            except Exception:
                pass  # Ignore if errors during UI update (e.g. on exit)

    def update_lidar_status(self, status_text, color):
        def _upd():  # Closure for UI update
            try:
                if hasattr(self, 'lidar_status_label') and self.lidar_status_label.winfo_exists():
                    self.lidar_status_label.config(text=f"LIDAR: {status_text}")
                if hasattr(self, 'lidar_indicator') and self.lidar_indicator.winfo_exists():
                    self.lidar_indicator.delete("all");  # Clear previous oval
                    self.lidar_indicator.create_oval(1, 1, 9, 9, fill=color, outline=color)
            except Exception:
                pass  # Ignore if errors during UI update (e.g. on exit)

        if hasattr(self, 'window') and self.window.winfo_exists():
            if threading.current_thread() is threading.main_thread():
                _upd()
            else:
                self.window.after(0, _upd)  # Schedule update on main thread

    def _is_manual_control_allowed(self, action="any"):
        if not (hasattr(self, 'status_label') and self.status_label.winfo_exists()):
            # This check is mostly for robustness, status_label should exist if UI is up.
            return False

        if self.car_locked:  # If car's forward movement is locked by an obstacle
            if action == "forward":
                return False  # Forward is not allowed
            elif action in ["backward", "left", "right", "stop"]:
                return True  # These evasive/stop maneuvers are allowed
            else:  # For "any" or other unknown actions when locked
                return False  # Default to not allowed if locked and action is ambiguous
        return True  # Not locked, all manual actions are allowed by default

    def button_hover(self, btn, is_hovering):
        try:
            if not (hasattr(self, 'status_label') and self.status_label.winfo_exists()): return
            btn.config(bg=self.button_hover_color if is_hovering else self.button_color)

            if is_hovering:
                action_desc = self._get_button_action_description(btn)
                current_action_type = "any"  # Determine action type from button
                if btn == self.btn_forward:
                    current_action_type = "forward"
                elif btn == self.btn_backward:
                    current_action_type = "backward"
                elif btn == self.btn_left:
                    current_action_type = "left"
                elif btn == self.btn_right:
                    current_action_type = "right"
                elif btn == self.btn_stop_manual:
                    current_action_type = "stop"

                can_perform_action = self._is_manual_control_allowed(current_action_type)

                if action_desc:
                    if can_perform_action:
                        self.status_label.config(text=f"Manual: {action_desc}?", fg=self.info_color)
                    elif btn == self.btn_forward and self.car_locked:  # Specific message for forward when locked
                        self.status_label.config(text="FORWARD BLOCKED (Obstacle)", fg=self.warning_color)
                    # If other moves are blocked (which current logic in _is_manual_control_allowed
                    # doesn't do for B/L/R/Stop when locked), status would remain as per autonomous logic.
            else:  # Not hovering, restore status label based on car's current autonomous state
                if self.car_locked and self.any_object_in_front_details:
                    self.status_label.config(
                        text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                        fg=self.warning_color)
                elif self.car_locked:  # Locked but no specific obstacle details (e.g., during brief transitions)
                    self.status_label.config(text="OBSTACLE: FWD LOCKED", fg=self.warning_color)
                else:  # Not locked, path is clear or system is ready
                    self.status_label.config(text="System Ready", fg=self.info_color)
        except tk.TclError:
            pass  # Ignore if widget is destroyed during hover event
        except Exception as e:
            print(f"Error in button_hover: {e}")

    def _get_button_action_description(self, button_widget):
        actions = {
            self.btn_forward: "Move Forward",
            self.btn_backward: "Move Backward",
            self.btn_left: "Turn Left",
            self.btn_right: "Turn Right",
            self.btn_stop_manual: "STOP All Movement"
        }
        return actions.get(button_widget)

    def forward_pressed(self, ev=None):
        if not self._is_manual_control_allowed("forward"):
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():  # Update status if action denied
                self.status_label.config(text="FORWARD BLOCKED (Obstacle)", fg=self.warning_color)
            return
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            self.status_label.config(text="MANUAL: FORWARD", fg=self.info_color)
        self._execute_motor_move_forward()

    def forward_released(self, ev=None):
        # Stop only if the car is NOT locked by an obstacle.
        # If it's locked, the autonomous system is in control of the stop state.
        if not self.car_locked:
            self._execute_motor_stop_all_movement()
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                # If not locked (path clear), status should reflect readiness.
                self.status_label.config(text="System Ready", fg=self.info_color)
        # If car_locked, status label should remain "OBSTACLE: FWD LOCKED..." (or similar)
        # which is handled by manage_autonomous_maneuver or button_hover on mouse_leave.

    def backward_pressed(self, ev=None):
        if not self._is_manual_control_allowed("backward"): return  # Should always be allowed by current logic
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            status_msg = "MANUAL: BACKWARD (FWD Locked)" if self.car_locked else "MANUAL: BACKWARD"
            self.status_label.config(text=status_msg, fg=self.info_color)
        self._execute_motor_move_backward()

    def backward_released(self, ev=None):
        self._execute_motor_stop_all_movement()
        # Restore status label based on car_locked state
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            if self.car_locked and self.any_object_in_front_details:
                self.status_label.config(
                    text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                    fg=self.warning_color)
            elif self.car_locked:
                self.status_label.config(text="OBSTACLE: FWD LOCKED", fg=self.warning_color)
            else:
                self.status_label.config(text="System Ready", fg=self.info_color)

    def left_pressed(self, ev=None):
        if not self._is_manual_control_allowed("left"): return
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            status_msg = "MANUAL: ROTATE LEFT (FWD Locked)" if self.car_locked else "MANUAL: ROTATE LEFT"
            self.status_label.config(text=status_msg, fg=self.info_color)
        self._execute_motor_rotate_left()

    def left_released(self, ev=None):
        self._execute_motor_stop_rotation()
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():  # Restore status
            if self.car_locked and self.any_object_in_front_details:
                self.status_label.config(
                    text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                    fg=self.warning_color)
            elif self.car_locked:
                self.status_label.config(text="OBSTACLE: FWD LOCKED", fg=self.warning_color)
            else:
                self.status_label.config(text="System Ready", fg=self.info_color)

    def right_pressed(self, ev=None):
        if not self._is_manual_control_allowed("right"): return
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            status_msg = "MANUAL: ROTATE RIGHT (FWD Locked)" if self.car_locked else "MANUAL: ROTATE RIGHT"
            self.status_label.config(text=status_msg, fg=self.info_color)
        self._execute_motor_rotate_right()

    def right_released(self, ev=None):
        self._execute_motor_stop_rotation()
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():  # Restore status
            if self.car_locked and self.any_object_in_front_details:
                self.status_label.config(
                    text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                    fg=self.warning_color)
            elif self.car_locked:
                self.status_label.config(text="OBSTACLE: FWD LOCKED", fg=self.warning_color)
            else:
                self.status_label.config(text="System Ready", fg=self.info_color)

    def stop_pressed_released(self, ev=None):  # Acts on press
        self._execute_motor_stop_all_movement()  # Send 'S' (Stop All)

        # Update status label after a brief delay to reflect current lock state post-stop
        def update_stop_status_after_delay():
            if not (hasattr(self, 'status_label') and self.status_label.winfo_exists()): return

            if self.car_locked and self.any_object_in_front_details:
                # If still locked due to persistent obstacle
                self.status_label.config(
                    text=f"OBSTACLE: FWD LOCKED (Dist: {self.any_object_in_front_details['avg_dist']:.0f}mm)",
                    fg=self.warning_color)
            elif self.car_locked:
                # If locked, but maybe details are momentarily unavailable
                self.status_label.config(text="OBSTACLE: FWD LOCKED (Manual Stop)", fg=self.warning_color)
            else:
                # If not locked (path was clear or became clear)
                self.status_label.config(text="MANUAL STOP: System Ready", fg=self.caution_color)

        if hasattr(self, 'window') and self.window.winfo_exists():
            self.window.after(50, update_stop_status_after_delay)  # 50ms delay

    def toggle_fullscreen(self, ev=None):
        self.is_fullscreen = not self.is_fullscreen
        self.window.attributes('-fullscreen', self.is_fullscreen);
        return "break"  # Prevents further processing of the event

    def exit_app(self):
        print("Exiting application...")
        self.is_reconnecting_lidar = False;  # Stop any reconnect attempts
        self.is_scanning = False  # Signal all scanning loops to stop

        # Best effort to stop motors and unlock
        # self.ser is not used for send_command, but check if it were for direct serial
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            print("Sending final STOP ('S') and UNLOCK ('U') via direct serial (if used)...");
            # Direct serial commands would go here if self.ser was used
        else:  # Assuming send_command is primary
            print("Sending final STOP ('S') and UNLOCK ('U') via socket command...");
            send_command('S');
            time.sleep(0.05)  # Brief pause for command processing
            send_command('U');
            time.sleep(0.05)

        # Close UDP socket
        if self.udp_socket:
            print("Closing UDP socket...")
            try:
                self.udp_socket.close()
            except Exception as e:
                print(f"Error closing UDP socket: {e}")
            self.udp_socket = None

        # Join threads
        threads_to_join = []
        for attr_name in ['scan_thread', 'lidar_reconnect_thread', 'demo_thread']:
            thread = getattr(self, attr_name, None)
            if thread and thread.is_alive(): threads_to_join.append(thread)

        for t in threads_to_join:
            print(f"Waiting for {t.name} to finish...");
            t.join(1.5)  # Timeout for join

        # Disconnect Lidar
        if hasattr(self, 'lidar') and self.lidar:
            try:
                print("Stopping Lidar motor and disconnecting Lidar...")
                self.lidar.stop();
                self.lidar.stop_motor();
                self.lidar.disconnect()
            except Exception as e:
                print(f"Error during RPLidar stop/disconnect: {e}")

        # Close serial port if it was opened directly
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            print("Closing direct serial connection (if used)...")
            self.ser.close()

        # Destroy Tkinter window
        if hasattr(self, 'window') and self.window.winfo_exists():
            print("Destroying Tkinter window...")
            self.window.destroy()

        print("Application exited.")


if __name__ == "__main__":
    print("RPLidar Navigator (RPi Optimized - Conditional Manual Control + UDP Stream)")
    print("-----------------------------------------------------------------")
    # These are informational defaults, actual values are set in the class
    DEFAULT_LIDAR_ANGLE_OFFSET_DEG = 180.0
    DEFAULT_MAX_RANGE_MM = 2000

    print(f" - Default Lidar Angle Offset (for internal viz): {DEFAULT_LIDAR_ANGLE_OFFSET_DEG}°.")
    print(f" - Max Lidar range (for internal viz/detection): {DEFAULT_MAX_RANGE_MM / 1000.0:.1f} meters.")
    print(" - Lidar will attempt to reconnect on errors, fallback to Demo mode if fails.")
    print(" - Obstacle detection: Stop ('S') then FWD Lock ('K'). Path clear: Unlock ('U').")
    print(" - When FWD Locked, Backward/Left/Right manual moves are allowed.")
    print(" - Lidar data (raw angles, specific format) can be streamed via UDP.")
    print("Press Ctrl+C in terminal or Esc key in GUI to exit.")
    print("-" * 50)

    root = tk.Tk()
    # Window title updated to reflect UDP capability
    app_instance = RPLidarCarControl(root, "RPLidar Navigator (RPi FwdLock + UDP)")

    print("Application Settings:")
    print(f"  Lidar Angle Offset (internal viz): {app_instance.LIDAR_ANGLE_OFFSET_DEG}°")
    print(f"  Max Range (internal viz/detection): {app_instance.MAX_RANGE_MM}mm")
    print(f"  Scan Processing Interval: {app_instance.SCAN_INTERVAL}s")
    print(f"  Scan Point Subsample Ratio (internal): {app_instance.SCAN_POINT_SUBSAMPLE_RATIO}")
    print(f"  Debug Front Detection: {'ON' if app_instance.DEBUG_FRONT_DETECTION else 'OFF'}")

    print(f"UDP Streaming: {'Enabled' if app_instance.UDP_STREAM_ENABLED else 'Disabled'}")
    if app_instance.UDP_STREAM_ENABLED:
        print(f"  UDP Target: {app_instance.UDP_RECEIVER_IP}:{app_instance.UDP_RECEIVER_PORT}")
        print(f"  UDP Send Rate (target): {app_instance.UDP_SEND_RATE_HZ} Hz")
        print(f"  UDP Max Points per packet: {app_instance.UDP_MAX_POINTS_TO_SEND}")
        print(f"  UDP Min Quality for send: {app_instance.UDP_MIN_QUALITY_FOR_SEND}")
        print(f"  UDP Max Distance for send: {app_instance.UDP_MAX_DISTANCE_FOR_SEND} mm")
    print("-" * 50)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, exiting application...")
        if 'app_instance' in locals() and app_instance: app_instance.exit_app()
    except Exception as e:  # Catch other unexpected errors during mainloop
        print(f"Unhandled exception in main Tkinter loop: {e}")
        import traceback;

        traceback.print_exc()
        if 'app_instance' in locals() and app_instance: app_instance.exit_app()
    finally:
        print("Main loop finished or interrupted.")
        # Ensure cleanup if app_instance exists, especially if WM_DELETE_WINDOW wasn't called
        if 'app_instance' in locals() and app_instance:
            # Check if threads might still be running or window exists
            needs_cleanup = False
            if hasattr(app_instance, 'window') and app_instance.window.winfo_exists():
                needs_cleanup = True  # Window exists, so resources might be active

            # Check threads even if window is gone, as they are daemonic but better to join
            for attr_name in ['scan_thread', 'lidar_reconnect_thread', 'demo_thread']:
                thread = getattr(app_instance, attr_name, None)
                if thread and thread.is_alive():
                    needs_cleanup = True;
                    break

            if needs_cleanup and not (hasattr(app_instance, 'is_scanning') and not app_instance.is_scanning and \
                                      hasattr(app_instance,
                                              'is_reconnecting_lidar') and not app_instance.is_reconnecting_lidar):
                # If scanning/reconnecting flags are still true, or if we can't check them, assume cleanup is needed.
                print("Ensuring application cleanup from finally block...")
                app_instance.exit_app()
    sys.exit(0)