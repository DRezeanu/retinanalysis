import socket
import os

LISTEN_PORT = 9100
SAVE_DIR = '/home/vyomr/Desktop/data/raw/realtime_test/acquisition'
OUTPUT_FILE = os.path.join(SAVE_DIR, 'received_data.bin')

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(('', LISTEN_PORT))
        server_sock.listen(1)
        print(f"Listening for data on port {LISTEN_PORT}...")
        conn, addr = server_sock.accept()
        print(f"Connection from {addr}")
        with conn, open(OUTPUT_FILE, 'wb') as f:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                f.write(data)
        print(f"Data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()