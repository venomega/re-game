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
import os

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

        # Guardar dimensiones originales del servidor
        self.server_width = width
        self.server_height = height

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
        self.event_queue = queue.Queue(maxsize=9)
        self.event_thread = threading.Thread(target=self._event_sender)
        self.event_thread.daemon = True
        self.event_thread.start()

        # Iniciar thread de eventos
        self.event_processor_thread = threading.Thread(target=self._process_events)
        self.event_processor_thread.daemon = True
        self.event_processor_thread.start()

        # Estado de modo relativo
        self.relative_mouse_mode = True
        sdl2.SDL_SetRelativeMouseMode(True)

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


    def _process_events(self):
        """Thread dedicado para procesar eventos SDL"""
        while self.running:
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    self.running = False
                    break
                elif event.type == sdl2.SDL_KEYDOWN:
                    # Alternar Ctrl+g para capturar/liberar el mouse
                    if (event.key.keysym.sym == ord('g') or event.key.keysym.sym == ord('G')) and (event.key.keysym.mod & (sdl2.KMOD_LCTRL | sdl2.KMOD_RCTRL)):
                        self.relative_mouse_mode = not self.relative_mouse_mode
                        sdl2.SDL_SetRelativeMouseMode(self.relative_mouse_mode)
                        print(f"[INFO] Mouse {'capturado' if self.relative_mouse_mode else 'liberado'} (relative mode {'ON' if self.relative_mouse_mode else 'OFF'})")
                        # Si se libera el mouse, sincronizar la posición real con la última posición virtual
                        if not self.relative_mouse_mode:
                            try:
                                # Warp el mouse a la última posición conocida
                                sdl2.SDL_WarpMouseInWindow(self.window.window, event.motion.x, event.motion.y)
                            except Exception:
                                pass
                    key_data = struct.pack('<I', event.key.keysym.sym)
                    self.event_socket.sendall(struct.pack('<B', EVENT_KEYDOWN) + key_data)
                elif event.type == sdl2.SDL_KEYUP:
                    key_data = struct.pack('<I', event.key.keysym.sym)
                    self.event_socket.sendall(struct.pack('<B', EVENT_KEYUP) + key_data)
                elif event.type == sdl2.SDL_MOUSEMOTION:
                    # Enviar evento de movimiento de mouse
                    x = event.motion.x
                    y = event.motion.y
                    rel_x = event.motion.xrel
                    rel_y = event.motion.yrel
                    data = struct.pack('<Bhhhh', EVENT_MOUSEMOTION, x, y, rel_x, rel_y)
                    self.event_socket.sendall(data)
                elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                    # Enviar evento de botón presionado
                    button = event.button.button
                    x = event.button.x
                    y = event.button.y
                    data = struct.pack('<BBhh', EVENT_MOUSEBUTTONDOWN, button, x, y)
                    self.event_socket.sendall(data)
                elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                    # Enviar evento de botón liberado
                    button = event.button.button
                    x = event.button.x
                    y = event.button.y
                    data = struct.pack('<BBhh', EVENT_MOUSEBUTTONUP, button, x, y)
                    self.event_socket.sendall(data)
                elif event.type == sdl2.SDL_MOUSEWHEEL:
                    # Enviar evento de rueda del mouse
                    x = event.wheel.x
                    y = event.wheel.y
                    data = struct.pack('<Bhh', EVENT_MOUSEWHEEL, x, y)
                    self.event_socket.sendall(data)
            time.sleep(0.0001)  # Pequeña pausa para no saturar la CPU

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

def receive_audio():
    os.system("ffplay -rtsp_flags listen rtsp://192.168.2.192:8888 -nodisp").read()

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
    event_socket.connect((server_ip, 8081))

    # Recibir dimensiones de la pantalla
    width = struct.unpack('<I', frame_socket.recv(4))[0]
    height = struct.unpack('<I', frame_socket.recv(4))[0]
    print(f"Dimensiones de la pantalla: {width}x{height}")

    # Crear viewer
    viewer = ScreenShareViewer(width, height, event_socket)

    # Iniciar thread de recepción de frames
    receive_thread = threading.Thread(target=receive_frames, args=(frame_socket, viewer))
    receive_thread.daemon = False
    receive_thread.start()

    # Iniciar thread de audio
    audio_thread = threading.Thread(target=receive_audio)
    audio_thread.daemon = False
    audio_thread.start()

    try:
        running = True
        while running:
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
