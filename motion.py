import socket
import serial
import serial.tools.list_ports
import time
import threading


class ArduinoCarController:
    def __init__(self):
        self.arduino = None
        self.connected = False
        self.server_socket = None

    def find_arduino_port(self):
        """Automatically find the Arduino port with specific VID:PID detection"""
        print("Scanning for Arduino...")
        ports = serial.tools.list_ports.comports()

        for port in ports:
            print(f"Found port: {port.device} - {port.description}")
            print(f"VID:PID = {port.vid:04X}:{port.pid:04X}" if port.vid and port.pid else "VID:PID = Unknown")

            # Check for Genuine Arduino Uno R3 (VID:PID 2341:0043)
            if port.vid == 0x2341 and port.pid == 0x0043:
                print(f"✓ Genuine Arduino Uno R3 found at: {port.device}")
                return port.device

            # Skip RPLidar (CP2102 devices on USB ports)
            if port.vid == 0x10C4 and port.pid == 0xEA60:  # CP2102 VID:PID
                print(f"⚠ Skipping RPLidar (CP2102) at: {port.device}")
                continue

        # Fallback: Look for ACM devices (typical for Arduino on Linux)
        print("\nNo Arduino with VID:PID 2341:0043 found. Checking ACM devices...")
        for port in ports:
            if 'ttyACM' in port.device:
                # Double-check it's not a CP2102 device
                if port.vid == 0x10C4:  # Skip CP2102 devices
                    print(f"⚠ Skipping CP2102 device at: {port.device}")
                    continue
                print(f"Using Arduino port: {port.device} (ACM device)")
                return port.device

        # Last resort: Check for other Arduino-like devices, but skip CP2102
        print("\nNo ACM devices found. Checking other Arduino identifiers...")
        arduino_keywords = ['arduino', 'ch340', 'ch341', 'ft232']  # Removed cp210 to avoid CP2102

        for port in ports:
            # Skip CP2102 devices (RPLidar)
            if port.vid == 0x10C4 and port.pid == 0xEA60:
                continue

            if any(keyword in port.description.lower() for keyword in arduino_keywords):
                print(f"Arduino-like device found at: {port.device}")
                return port.device

        print("✗ No suitable Arduino port found!")
        print("Expected: /dev/ttyACM0 with VID:PID 2341:0043 (Arduino Uno R3)")
        print("RPLidar on /dev/ttyUSB0 with CP2102 will be skipped")
        return None

    def connect_arduino(self, port=None, baudrate=9600):
        """Connect to Arduino via serial"""
        if not port:
            port = self.find_arduino_port()

        if not port:
            return False

        try:
            print(f"Attempting to connect to Arduino on {port}...")
            self.arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset

            # Test connection by sending a stop command
            self.arduino.write(b'S')
            self.connected = True
            print(f"✓ Successfully connected to Arduino on {port}")
            return True

        except serial.SerialException as e:
            print(f"✗ Failed to connect to {port}: {e}")
            self.connected = False
            return False

    def send_command(self, command):
        """Send command to Arduino"""
        if not self.connected or not self.arduino:
            print("Arduino not connected!")
            return False

        try:
            # Validate command
            valid_commands = ['F', 'B', 'L', 'R', 'S', 'K', 'U']
            if command.upper() not in valid_commands:
                print(f"Invalid command: {command}")
                return False

            # Send command
            self.arduino.write(command.upper().encode())
            print(f"Sent command to Arduino: {command.upper()}")

            # Map commands to readable actions
            actions = {
                'F': 'Moving Forward',
                'B': 'Moving Backward',
                'L': 'Turning Left',
                'R': 'Turning Right',
                'S': 'Stopping',
                'K': 'Force Stop (Locked)',
                'U': 'Unlocking'
            }
            print(f"Action: {actions.get(command.upper(), 'Unknown')}")
            return True

        except Exception as e:
            print(f"Error sending command: {e}")
            return False

    def start_server(self, host='0.0.0.0', port=9000):
        """Start TCP server to receive commands"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((host, port))
            self.server_socket.listen(5)

            print(f"✓ Server started on {host}:{port}")
            print("Waiting for commands...")
            print("Valid commands: F(Forward), B(Backward), L(Left), R(Right), S(Stop), K(Force Stop), U(Unlock)")

            while True:
                try:
                    conn, addr = self.server_socket.accept()
                    with conn:
                        data = conn.recv(10).decode().strip().upper()
                        if data:
                            print(f"\nReceived from {addr}: {data}")
                            self.send_command(data)

                except socket.error as e:
                    if self.server_socket:  # Only print if socket still exists
                        print(f"Socket error: {e}")
                    break

        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up connections"""
        if self.arduino and self.arduino.is_open:
            self.send_command('S')  # Stop motors before disconnecting
            self.arduino.close()
            print("Arduino connection closed")

        if self.server_socket:
            self.server_socket.close()
            print("Server socket closed")


def main():
    controller = ArduinoCarController()

    try:
        # Connect to Arduino
        if not controller.connect_arduino():
            print("Failed to connect to Arduino. Exiting...")
            return

        # Start server mode
        print("\nStarting server mode...")
        controller.start_server(host='0.0.0.0')

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()