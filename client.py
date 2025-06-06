import socket
import struct
import pygame
import sys
import threading
import queue
import time

# Constantes para tipos de eventos
EVENT_KEYDOWN = 1
EVENT_KEYUP = 2
EVENT_MOUSEMOTION = 3
EVENT_MOUSEBUTTONDOWN = 4
EVENT_MOUSEBUTTONUP = 5
EVENT_MOUSEWHEEL = 6

class EventSender(threading.Thread):
    def __init__(self, event_socket):
        super().__init__()
        self.event_socket = event_socket
        self.event_queue = queue.Queue(maxsize=1000)
        self.running = True
        self.daemon = True  # El thread se cerrará cuando el programa principal termine

    def run(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)  # Timeout para poder verificar self.running
                self.event_socket.sendall(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error enviando evento: {e}")
                break

    def send_event(self, event_type, data):
        try:
            # Empaquetar el evento
            event = struct.pack('B', event_type) + data
            # Intentar agregar al buffer sin bloquear
            self.event_queue.put_nowait(event)
        except queue.Full:
            # Si el buffer está lleno, descartar el evento más antiguo
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event)
            except:
                pass

    def stop(self):
        self.running = False
        self.join()

class ScreenShareViewer:
    def __init__(self, frame_socket, event_sender):
        self.frame_socket = frame_socket
        self.event_sender = event_sender
        self.running = True
        self.frame_buffer = []
        self.frame_buffer_size = 60
        self.last_frame_time = 0
        self.target_fps = 60
        self.frame_interval = 1.0 / self.target_fps

        # Recibir dimensiones de la pantalla
        width = struct.unpack('<I', self.frame_socket.recv(4))[0]
        height = struct.unpack('<I', self.frame_socket.recv(4))[0]

        # Inicializar Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Screen Share Viewer")

        # Pre-asignar buffer para la conversión RGB
        self.rgb_buffer = bytearray(width * height * 3)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                key_code = event.key
                self.event_sender.send_event(EVENT_KEYDOWN, struct.pack('<I', key_code))
            elif event.type == pygame.KEYUP:
                key_code = event.key
                self.event_sender.send_event(EVENT_KEYUP, struct.pack('<I', key_code))
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                self.event_sender.send_event(EVENT_MOUSEMOTION, struct.pack('<II', x, y))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                button = event.button
                self.event_sender.send_event(EVENT_MOUSEBUTTONDOWN, struct.pack('B', button))
            elif event.type == pygame.MOUSEBUTTONUP:
                button = event.button
                self.event_sender.send_event(EVENT_MOUSEBUTTONUP, struct.pack('B', button))
            elif event.type == pygame.MOUSEWHEEL:
                scroll = event.y
                self.event_sender.send_event(EVENT_MOUSEWHEEL, struct.pack('<i', scroll))

    def run(self):
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            # Manejar eventos de Pygame
            self.handle_events()

            # Recibir y mostrar frame solo si ha pasado suficiente tiempo
            if elapsed >= self.frame_interval:
                try:
                    # Recibir tamaño del frame
                    frame_size = struct.unpack('<I', self.frame_socket.recv(4))[0]
                    
                    # Recibir frame
                    frame_data = b''
                    while len(frame_data) < frame_size:
                        chunk = self.frame_socket.recv(min(4096, frame_size - len(frame_data)))
                        if not chunk:
                            raise ConnectionError("Conexión cerrada")
                        frame_data += chunk

                    # Convertir a imagen Pygame
                    image = pygame.image.load(pygame.io.BytesIO(frame_data))
                    
                    # Mostrar frame
                    self.screen.blit(image, (0, 0))
                    pygame.display.flip()
                    
                    self.last_frame_time = current_time
                except Exception as e:
                    print(f"Error recibiendo frame: {e}")
                    self.running = False

            # Pequeña pausa para no saturar la CPU
            time.sleep(0.001)

        pygame.quit()

def main():
    if len(sys.argv) != 2:
        print("Uso: python client.py <IP_DEL_SERVIDOR>")
        sys.exit(1)

    server_ip = sys.argv[1]

    # Conectar socket para frames
    frame_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    frame_socket.connect((server_ip, 8080))

    # Conectar socket para eventos
    event_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    event_socket.connect((server_ip, 8081))

    # Crear y iniciar el thread de envío de eventos
    event_sender = EventSender(event_socket)
    event_sender.start()

    try:
        # Iniciar el visor
        viewer = ScreenShareViewer(frame_socket, event_sender)
        viewer.run()
    finally:
        # Asegurar que el thread de eventos se detenga
        event_sender.stop()
        frame_socket.close()
        event_socket.close()

if __name__ == "__main__":
    main()
