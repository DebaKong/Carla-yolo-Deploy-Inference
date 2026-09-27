import socket
import struct

class TcpClient:
    def __init__(self, server_ip: str, server_port: int, timeout: float = 5.0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock.settimeout(self.timeout)

        self.sock.connect((self.server_ip, self.server_port))
        print(
            f"[TCP] Connected to {self.server_ip}:{self.server_port}"
        )

    def send_message(self, data: bytes):
        if self.sock is None:
            raise ConnectionError("TCP client is not connected.")

        header = struct.pack("!I", len(data))
        self.sock.sendall(header + data)

    def receive_exact(self, size: int) -> bytes:
        if self.sock is None:
            raise ConnectionError("TCP client is not connected.")

        data = bytearray()

        while len(data) < size:
            chunk = self.sock.recv(size - len(data))

            if not chunk:
                raise ConnectionError("Server closed connection.")

            data.extend(chunk)

        return bytes(data)

    def receive_message(self) -> bytes:
        if self.sock is None:
            raise ConnectionError("TCP client is not connected.")

        header = self.receive_exact(4)

        payload_size = struct.unpack("!I", header)[0]

        return self.receive_exact(payload_size)

    def receive_tracks(self):
        data = self.receive_message()

        if len(data) < 4:
            raise ValueError("Invalid detection data received.")

        offset = 0
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        tracks = []
        track_size = 28  # Each detection consists of 6 floats (4 bytes each)

        expected_size = 4 + count * track_size

        if len(data) != expected_size:
            raise ValueError(
                f"Expected {expected_size} bytes, but received {len(data)} bytes."
            )

        for _ in range(count):
            class_id = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            track_id = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            confidence = struct.unpack_from("<f", data, offset)[0]
            offset += 4

            x = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            y = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            width = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            height = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            tracks.append({
                "class_id": class_id,
                "track_id": track_id,
                "confidence": confidence,
                "x": x,
                "y": y,
                "width": width,
                "height": height
            })

        return tracks

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            