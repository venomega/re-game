import socket
import struct
import numpy as np
import sdl2
import sdl2.ext
import time
import cv2
import sys
import threading
from collections import deque
import queue

# Constantes para tipos de eventos
EVENT_KEYDOWN = 1
EVENT_KEYUP = 2
EVENT_MOUSEMOTION = 3
EVENT_MOUSEBUTTONDOWN = 4
EVENT_MOUSEBUTTONUP = 5
EVENT_MOUSEWHEEL = 6

class ScreenShareViewer:
    def __init__(self, width, height, event_socket):
        # Inicializar SDL
        sdl2.ext.init()
        
        # Crear ventana
        self.window = sdl2.ext.Window("Screen Share", size=(width, height))
        self.window.show()
        
        # Crear renderer
        self.renderer = sdl2.ext.Renderer(self.window)
        self.renderer.clear(sdl2.ext.Color(0, 0, 0))
        
        # Crear textura con formato RGB24
        self.texture = sdl2.SDL_CreateTexture(
            self.renderer.sdlrenderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            width,
            height
        )
        
        # Variables para FPS
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Tamaño de la ventana
        self.window_width = width
        self.window_height = height
        
        # Buffer de frames circular
        self.frame_buffer = deque(maxlen=60)  # Buffer de 60 frames
        self.running = True
        self.buffer_lock = threading.Lock()
        
        # Pre-asignar buffer para RGB
        self.rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Socket para eventos
        self.event_socket = event_socket
        
        # Buffer de eventos y thread
        self.event_queue = queue.Queue(maxsize=1000)  # Buffer de eventos
        self.event_thread = threading.Thread(target=self._event_sender)
        self.event_thread.daemon = True
        self.event_thread.start()

    def _event_sender(self):
        """Thread dedicado para enviar eventos al servidor"""
        while self.running:
            try:
                event_data = self.event_queue.get(timeout=0.001)  # Timeout corto para no bloquear
                self.event_socket.sendall(event_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error enviando evento: {e}")
                break

    def send_event(self, event_type, data):
        """Agrega un evento al buffer de eventos"""
        try:
            # Crear el paquete de evento
            event_data = struct.pack('<B', event_type) + data
            # Intentar agregar al buffer sin bloquear
            self.event_queue.put_nowait(event_data)
        except queue.Full:
            # Si el buffer está lleno, descartar el evento más antiguo
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event_data)
            except:
                pass

    def handle_events(self):
        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                return False
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    return False
                # Enviar evento de tecla presionada
                data = struct.pack('<I', event.key.keysym.sym)
                self.send_event(EVENT_KEYDOWN, data)
            elif event.type == sdl2.SDL_KEYUP:
                # Enviar evento de tecla liberada
                data = struct.pack('<I', event.key.keysym.sym)
                self.send_event(EVENT_KEYUP, data)
            elif event.type == sdl2.SDL_MOUSEMOTION:
                # Enviar evento de movimiento del ratón
                data = struct.pack('<ii', event.motion.x, event.motion.y)
                self.send_event(EVENT_MOUSEMOTION, data)
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                # Enviar evento de botón del ratón presionado
                data = struct.pack('<B', event.button.button)
                self.send_event(EVENT_MOUSEBUTTONDOWN, data)
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                # Enviar evento de botón del ratón liberado
                data = struct.pack('<B', event.button.button)
                self.send_event(EVENT_MOUSEBUTTONUP, data)
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                # Enviar evento de rueda del ratón
                data = struct.pack('<i', event.wheel.y)
                self.send_event(EVENT_MOUSEWHEEL, data)
        return True

    def update_frame(self, frame_data):
        try:
            # Decodificar JPEG usando OpenCV
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                return
            
            # Obtener el tamaño actual de la ventana
            window_size = self.window.size
            if window_size != (self.window_width, self.window_height):
                # Redimensionar la imagen al tamaño de la ventana
                frame = cv2.resize(frame, window_size, interpolation=cv2.INTER_LINEAR)
                self.window_width, self.window_height = window_size
            
            # Convertir BGR a RGB usando buffer pre-asignado
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self.rgb_buffer)
            
            # Actualizar textura
            sdl2.SDL_UpdateTexture(
                self.texture,
                None,
                self.rgb_buffer.tobytes(),
                self.rgb_buffer.shape[1] * 3
            )
            
            # Limpiar renderer
            self.renderer.clear()
            
            # Dibujar textura
            sdl2.SDL_RenderCopy(
                self.renderer.sdlrenderer,
                self.texture,
                None,
                None
            )
            sdl2.SDL_RenderPresent(self.renderer.sdlrenderer)
            
            # Calcular FPS y latencia
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            latency = (current_time - self.last_frame_time) * 1000  # ms
            
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                print(f"FPS: {self.fps:.1f}, Latencia: {latency:.1f}ms")
                self.frame_count = 0
                self.start_time = current_time
            
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"Error procesando frame: {e}")

    def cleanup(self):
        self.running = False
        sdl2.SDL_DestroyTexture(self.texture)
        sdl2.ext.quit()

def receive_frames(client, viewer):
    # Pre-asignar buffer para recepción
    receive_buffer = bytearray(1024 * 1024)  # 1MB buffer
    
    while viewer.running:
        try:
            # Recibir tamaño del frame comprimido
            frame_size = struct.unpack('<I', client.recv(4))[0]
            
            # Recibir frame comprimido usando buffer pre-asignado
            bytes_received = 0
            while bytes_received < frame_size:
                chunk = client.recv(min(frame_size - bytes_received, 65536))
                if not chunk:
                    raise ConnectionError("Conexión cerrada por el servidor")
                receive_buffer[bytes_received:bytes_received+len(chunk)] = chunk
                bytes_received += len(chunk)

            # Agregar frame al buffer circular
            with viewer.buffer_lock:
                viewer.frame_buffer.append(bytes(receive_buffer[:frame_size]))

        except Exception as e:
            print(f"Error recibiendo frames: {e}")
            break

def main():
    # Conectar al servidor para frames
    frame_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Conectar al servidor para eventos
    event_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_ip = ""
    try:
        server_ip = sys.argv[1]
    except:
        print("Uso: python client.py <IP_DEL_SERVIDOR>")
        sys.exit(1)
    
    frame_socket.connect((server_ip, 8080))
    event_socket.connect((server_ip, 8081))  # Puerto diferente para eventos

    # Recibir dimensiones de la pantalla
    width = struct.unpack('<I', frame_socket.recv(4))[0]
    height = struct.unpack('<I', frame_socket.recv(4))[0]
    print(f"Dimensiones de la pantalla: {width}x{height}")

    # Crear viewer
    viewer = ScreenShareViewer(width, height, event_socket)

    # Iniciar thread de recepción de frames
    receive_thread = threading.Thread(target=receive_frames, args=(frame_socket, viewer))
    receive_thread.daemon = True
    receive_thread.start()

    try:
        running = True
        while running:
            # Procesar eventos SDL
            running = viewer.handle_events()

            # Procesar frames del buffer circular
            with viewer.buffer_lock:
                if viewer.frame_buffer:
                    frame_data = viewer.frame_buffer.popleft()
                    viewer.update_frame(frame_data)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        viewer.cleanup()
        frame_socket.close()
        event_socket.close()

if __name__ == "__main__":
    main()