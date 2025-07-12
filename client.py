import socket
import struct
import numpy as np
import sdl2
import time
import sys
import threading
import queue
import os
import av
from PIL import Image
import subprocess

# Constantes para tipos de eventos
EVENT_KEYDOWN = 1
EVENT_KEYUP = 2
EVENT_MOUSEMOTION = 3
EVENT_MOUSEBUTTONDOWN = 4
EVENT_MOUSEBUTTONUP = 5
EVENT_MOUSEWHEEL = 6

def receive_frames_ffmpeg(frame_queue, width, height, udp_port=5000):
    ffmpeg_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-flush_packets", "1",
        "-max_delay", "0",
        "-i", f"udp://0.0.0.0:{udp_port}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=open("/dev/null", "bw"), bufsize=10**8)
    while True:
        raw_frame = proc.stdout.read(width * height * 3)
        if len(raw_frame) < width * height * 3:
            break
        img = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frame_queue.put(img, timeout=0.1)

def send_events(event_socket, running_flag):
    event = sdl2.SDL_Event()
    relative_mouse_mode = True
    sdl2.SDL_SetRelativeMouseMode(True)
    while running_flag['running']:
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT:
                running_flag['running'] = False
                break
            elif event.type == sdl2.SDL_KEYDOWN:
                keysym = event.key.keysym
                # Alternar Ctrl+g para capturar/liberar el mouse
                if (keysym.sym == ord('g') or keysym.sym == ord('G')) and (keysym.mod & (sdl2.KMOD_LCTRL | sdl2.KMOD_RCTRL)):
                    relative_mouse_mode = not relative_mouse_mode
                    sdl2.SDL_SetRelativeMouseMode(relative_mouse_mode)
                    print(f"[INFO] Mouse {'capturado' if relative_mouse_mode else 'liberado'} (relative mode {'ON' if relative_mouse_mode else 'OFF'})")
                key_data = struct.pack('<I', keysym.sym)
                event_socket.sendall(struct.pack('<B', EVENT_KEYDOWN) + key_data)
            elif event.type == sdl2.SDL_KEYUP:
                keysym = event.key.keysym
                key_data = struct.pack('<I', keysym.sym)
                event_socket.sendall(struct.pack('<B', EVENT_KEYUP) + key_data)
            elif event.type == sdl2.SDL_MOUSEMOTION:
                x = event.motion.x
                y = event.motion.y
                rel_x = event.motion.xrel
                rel_y = event.motion.yrel
                data = struct.pack('<Bhhhh', EVENT_MOUSEMOTION, x, y, rel_x, rel_y)
                event_socket.sendall(data)
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                button = event.button.button
                x = event.button.x
                y = event.button.y
                data = struct.pack('<BBhh', EVENT_MOUSEBUTTONDOWN, button, x, y)
                event_socket.sendall(data)
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                button = event.button.button
                x = event.button.x
                y = event.button.y
                data = struct.pack('<BBhh', EVENT_MOUSEBUTTONUP, button, x, y)
                event_socket.sendall(data)
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                x = event.wheel.x
                y = event.wheel.y
                data = struct.pack('<Bhh', EVENT_MOUSEWHEEL, x, y)
                event_socket.sendall(data)
        time.sleep(0.0001)

def get_local_ip():
    """Obtiene la IP local de la interfaz de red principal."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #try:
    #    # No importa si no hay conectividad, solo queremos la IP local
    #    s.connect(('8.8.8.8', 80))
    #    ip = s.getsockname()[0]
    #except Exception:
    #    ip = '127.0.0.1'
    #finally:
    #    s.close()
    #return ip

def receive_audio():
    os.system("ffplay -rtsp_flags listen rtsp://@:8888 -nodisp")

def main():
    event_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = ""
    try:
        server_ip = sys.argv[1]
    except:
        print("Uso: python client.py <IP_DEL_SERVIDOR>")
        sys.exit(1)
    event_socket.connect((server_ip, 8081))
    width, height = 1360, 768
    print(f"Resolución esperada: {width}x{height}")
    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
    window = sdl2.SDL_CreateWindow(
        b"Screen Share",
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        width, height,
        0
    )
    renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.SDL_RENDERER_SOFTWARE)
    texture = sdl2.SDL_CreateTexture(
        renderer,
        sdl2.SDL_PIXELFORMAT_RGB24,
        sdl2.SDL_TEXTUREACCESS_STREAMING,
        width,
        height
    )
    frame_queue = queue.Queue(maxsize=2)
    running_flag = {'running': True}
    audio_thread = threading.Thread(target=receive_audio)
    audio_thread.daemon = False
    audio_thread.start()
    threading.Thread(target=receive_frames_ffmpeg, args=(frame_queue, width, height), daemon=True).start()
    threading.Thread(target=send_events, args=(event_socket, running_flag), daemon=True).start()
    frame_count = 0
    start_time = time.time()
    while running_flag['running']:
        try:
            while True:
                img = frame_queue.get_nowait()
                # Robustez: verifica shape y contigüidad
                if img.shape != (height, width, 3):
                    print(f"[WARN] Frame shape inválido: {img.shape}, esperado: ({height}, {width}, 3). Ignorando frame.")
                    continue
                if not img.flags['C_CONTIGUOUS'] or img.strides[0] != img.shape[1] * 3:
                    img = np.ascontiguousarray(img)
                try:
                    # print("Frame shape:", img.shape, "dtype:", img.dtype)
                    data = img.tobytes()
                    expected_len = height * width * 3
                    if len(data) != expected_len:
                        print(f"[WARN] Frame data size inválido: {len(data)}, esperado: {expected_len}. Ignorando frame.")
                        continue
                    sdl2.SDL_UpdateTexture(texture, None, data, img.shape[1] * 3)
                    sdl2.SDL_RenderClear(renderer)
                    sdl2.SDL_RenderCopy(renderer, texture, None, None)
                    sdl2.SDL_RenderPresent(renderer)
                    frame_count += 1
                except Exception as e:
                    print(f"[ERROR] Excepción al renderizar frame: {e}")
        except queue.Empty:
            pass
        # FPS opcional
        if time.time() - start_time >= 1.0:
            print(f"FPS: {frame_count}")
            frame_count = 0
            start_time = time.time()
        time.sleep(0.001)
    sdl2.SDL_DestroyTexture(texture)
    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
    event_socket.close()

if __name__ == "__main__":
    main()
