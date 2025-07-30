package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"io"
	"log"
	"math"
	"net"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"time"
	"strings"
	"fmt"

	"bufio"

	"github.com/kbinani/screenshot"
	"github.com/go-vgo/robotgo"
	"github.com/ThomasT75/uinput"
	"screenshare/vjoy"
)

const (
	frameRate = 60  // FPS realista para baja latencia
	frameDuration = time.Second / frameRate
	framePort = ":8080"
	eventPort = ":8081"
	audioPort = ":8082"  // Puerto para streaming de audio
	maxBufferSize = 1024 * 1024 // 1MB buffer
	bufferSize = 1 // Buffer de 1 frame para mínima latencia
	jpegQuality = 40 // Mejor balance calidad/velocidad
	eventBufferSize = 1
	audioSampleRate = 48000 // Tasa de muestreo de audio actualizada a 48kHz
	audioChannels = 2       // Audio estéreo
	audioBufferSize = 1024  // Ajustado para mejor calidad
)

var video_process map[string]*os.Process = make(map[string]*os.Process)

// Constantes para tipos de eventos
const (
	EVENT_KEYDOWN = 1
	EVENT_KEYUP = 2
	EVENT_MOUSEMOTION = 3
	EVENT_MOUSEBUTTONDOWN = 4
	EVENT_MOUSEBUTTONUP = 5
	EVENT_MOUSEWHEEL = 6
	EVENT_JOYSTICK = 10
	EVENT_JOYSTICK_CAPS = 11
)

// Estructura para eventos
type Event struct {
	Type byte
	Data []byte
}

type EventBuffer struct {
	events []Event
	mu     sync.Mutex
	cond   *sync.Cond
	head   int
	tail   int
	size   int
}

func NewEventBuffer(size int) *EventBuffer {
	eb := &EventBuffer{
		events: make([]Event, size),
		size:   size,
	}
	eb.cond = sync.NewCond(&eb.mu)
	return eb
}

func (eb *EventBuffer) Add(event Event) {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	// Si el buffer está lleno, avanzar el head
	if (eb.tail+1)%eb.size == eb.head {
		eb.head = (eb.head + 1) % eb.size
	}

	// Agregar el nuevo evento
	eb.events[eb.tail] = event
	eb.tail = (eb.tail + 1) % eb.size
	eb.cond.Signal()
}

func (eb *EventBuffer) Get() Event {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	// Esperar hasta que haya un evento disponible
	for eb.head == eb.tail {
		eb.cond.Wait()
	}

	// Obtener el evento más antiguo
	event := eb.events[eb.head]
	eb.head = (eb.head + 1) % eb.size
	return event
}

func (eb *EventBuffer) Clear() {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.head = 0
	eb.tail = 0
}

type FrameBuffer struct {
	frames []image.Image
	mu     sync.Mutex
	cond   *sync.Cond
	head   int
	tail   int
	size   int
}

func NewFrameBuffer(size int) *FrameBuffer {
	fb := &FrameBuffer{
		frames: make([]image.Image, size),
		size:   size,
	}
	fb.cond = sync.NewCond(&fb.mu)
	return fb
}

func (fb *FrameBuffer) Add(frame image.Image) {
	fb.mu.Lock()
	defer fb.mu.Unlock()

	// Si el buffer está lleno, avanzar el head
	if (fb.tail+1)%fb.size == fb.head {
		fb.head = (fb.head + 1) % fb.size
	}

	// Agregar el nuevo frame
	fb.frames[fb.tail] = frame
	fb.tail = (fb.tail + 1) % fb.size
	fb.cond.Signal()
}

func (fb *FrameBuffer) Get() image.Image {
	fb.mu.Lock()
	defer fb.mu.Unlock()

	// Esperar hasta que haya un frame disponible
	for fb.head == fb.tail {
		fb.cond.Wait()
	}

	// Obtener el frame más antiguo
	frame := fb.frames[fb.head]
	fb.head = (fb.head + 1) % fb.size
	return frame
}

func (fb *FrameBuffer) Clear() {
	fb.mu.Lock()
	defer fb.mu.Unlock()
	fb.head = 0
	fb.tail = 0
}

// AudioBuffer para manejar el streaming de audio
type AudioBuffer struct {
	buffer []float32
	mu     sync.Mutex
	cond   *sync.Cond
	head   int
	tail   int
	size   int
}

func NewAudioBuffer(size int) *AudioBuffer {
	ab := &AudioBuffer{
		buffer: make([]float32, size),
		size:   size,
	}
	ab.cond = sync.NewCond(&ab.mu)
	return ab
}

func (ab *AudioBuffer) Add(samples []float32) {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	for _, sample := range samples {
		// Si el buffer está lleno, avanzar el head
		if (ab.tail+1)%ab.size == ab.head {
			ab.head = (ab.head + 1) % ab.size
		}

		// Agregar la nueva muestra
		ab.buffer[ab.tail] = sample
		ab.tail = (ab.tail + 1) % ab.size
	}
	ab.cond.Signal()
}

func (ab *AudioBuffer) Get(n int) []float32 {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	// Esperar hasta que haya suficientes muestras disponibles
	for (ab.tail-ab.head+ab.size)%ab.size < n {
		ab.cond.Wait()
	}

	// Obtener las muestras más antiguas
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		samples[i] = ab.buffer[ab.head]
		ab.head = (ab.head + 1) % ab.size
	}
	return samples
}

// AudioReader implementa io.Reader para el streaming de audio
type AudioReader struct {
	buffer []float32
	pos    int
	mu     sync.Mutex
	cond   *sync.Cond
}

func NewAudioReader(size int) *AudioReader {
	ar := &AudioReader{
		buffer: make([]float32, size),
	}
	ar.cond = sync.NewCond(&ar.mu)
	return ar
}

func (r *AudioReader) Read(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Esperar hasta que haya datos disponibles
	for r.pos >= len(r.buffer) {
		r.cond.Wait()
	}

	// Convertir el buffer de float32 a bytes
	available := len(r.buffer) - r.pos
	bytesToRead := len(p)
	if bytesToRead > available*4 {
		bytesToRead = available * 4
	}

	// Copiar los datos al buffer de salida
	for i := 0; i < bytesToRead/4; i++ {
		binary.LittleEndian.PutUint32(p[i*4:], math.Float32bits(r.buffer[r.pos+i]))
	}

	r.pos += bytesToRead / 4
	return bytesToRead, nil
}

func (r *AudioReader) Write(samples []float32) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Copiar las nuevas muestras al buffer
	copy(r.buffer, samples)
	r.pos = 0
	r.cond.Signal() // Notificar que hay nuevos datos disponibles
}

func init() {
	// Configurar Go para usar todos los núcleos disponibles
	runtime.GOMAXPROCS(runtime.NumCPU())
}

// Mapeo de códigos SDL a códigos de robotgo
var keyMap = map[uint32]string{
	13:  "enter",     // Return
	8:   "backspace", // Backspace
	9:   "tab",       // Tab
	27:  "esc",       // Escape
	32:  "space",     // Space
	273: "up",        // Up arrow
	274: "down",      // Down arrow
	275: "right",     // Right arrow
	276: "left",      // Left arrow
	277: "insert",    // Insert
	278: "home",      // Home
	279: "end",       // End
	280: "pageup",    // Page Up
	281: "pagedown",  // Page Down
	282: "f1",        // F1
	283: "f2",        // F2
	284: "f3",        // F3
	285: "f4",        // F4
	286: "f5",        // F5
	287: "f6",        // F6
	288: "f7",        // F7
	289: "f8",        // F8
	290: "f9",        // F9
	291: "f10",       // F10
	292: "f11",       // F11
	293: "f12",       // F12
	1073742048: "ctrl",   // Left Ctrl
	1073742052: "ctrl",   // Right Ctrl
	1073742049: "shift",  // Left Shift
	1073742053: "shift",  // Right Shift
	1073742050: "alt",    // Left Alt
	1073742054: "alt",    // Right Alt
	1073742051: "super",  // Left Super/Windows
	1073742055: "super",  // Right Super/Windows
}

// Estado de las teclas modificadoras
var modifierState = struct {
	ctrl  bool
	alt   bool
	super bool
	shift bool
	sync.Mutex
}{}

// Estado del mouse grab
var mouseGrab = struct {
	enabled bool
	windowX, windowY int
	windowWidth, windowHeight int
	sync.Mutex
}{}

var virtualMouse uinput.Mouse

func getKeyName(keyCode uint32) string {
	if name, ok := keyMap[keyCode]; ok {
		return name
	}
	// Para teclas normales, convertir a minúscula
	return string(rune(keyCode))
}


// Mutex global para sincronizar la captura de pantalla
var captureMutex sync.Mutex

func handleConnection(conn net.Conn) {
	defer conn.Close()

	// Obtener las dimensiones de la pantalla
	bounds := screenshot.GetDisplayBounds(0)
	width := bounds.Dx()
	height := bounds.Dy()

	// Enviar dimensiones al cliente
	binary.Write(conn, binary.LittleEndian, uint32(width))
	binary.Write(conn, binary.LittleEndian, uint32(height))

	// Buffer pre-asignado para JPEG
	jpegBuffer := make([]byte, maxBufferSize)

	// Canal para sincronización
	frameChan := make(chan image.Image, 1)

	// Iniciar thread de captura
	//go func(){
	//	lastCapture := time.Now()
	//	for {
	//		// Control de framerate para la captura
	//		elapsed := time.Since(lastCapture)
	//		if elapsed < frameDuration {
	//			time.Sleep(frameDuration - elapsed)
	//		}
	//		lastCapture = time.Now()

	//		// Captura de pantalla con mutex
	//		captureMutex.Lock()
	//		img, err := screenshot.CaptureDisplay(0)
	//		captureMutex.Unlock()

	//		if err != nil {
	//			log.Printf("Error capturando pantalla: %v", err)
	//			continue
	//		}

	//		// Enviar frame al canal
	//		select {
	//		case frameChan <- img:
	//		default:
	//			// Si el canal está lleno, descartamos el frame
	//		}
	//	}
	//}()

	frameCount := 0
	lastFPS := time.Now()
	lastFrameTime := time.Now()

	for {
		startTime := time.Now()

		// Obtener frame del canal
		frame := <-frameChan

		// Comprimir frame a JPEG usando el buffer pre-asignado
		buf := bytes.NewBuffer(jpegBuffer[:0])
		if err := jpeg.Encode(buf, frame, &jpeg.Options{Quality: jpegQuality}); err != nil {
			log.Printf("Error comprimiendo frame: %v", err)
			continue
		}

		// Enviar tamaño del frame comprimido
		frameSize := uint32(buf.Len())
		if err := binary.Write(conn, binary.LittleEndian, frameSize); err != nil {
			log.Printf("Error enviando tamaño del frame: %v", err)
			return
		}

		// Enviar frame comprimido
		if _, err := conn.Write(buf.Bytes()); err != nil {
			log.Printf("Error enviando frame: %v", err)
			return
		}

		// Calcular y mostrar FPS cada segundo
		frameCount++
		if time.Since(lastFPS) >= time.Second {
			latency := time.Since(lastFrameTime).Milliseconds()
			log.Printf("FPS: %d, Latencia: %dms, Tamaño frame: %d bytes", frameCount, latency, frameSize)
			frameCount = 0
			lastFPS = time.Now()
		}
		lastFrameTime = time.Now()

		// Mantener el framerate (pero sin sleep si la red es el cuello de botella)
		elapsed := time.Since(startTime)
		if elapsed < frameDuration {
			// time.Sleep(frameDuration - elapsed) // Puedes comentar esto para máxima velocidad
		}
	}
}

func captureSystemAudio(audioChan chan<- []float32) {
	log.Printf("Starting captureSystemAudio function")

	// Construct ffmpeg command to capture from ALSA
	for {
		cmd := exec.Command("ffmpeg",
			"-re",             // Read input at native frame rate
			"-f", "alsa",      // Use ALSA directly
			"-i", "default",   // Use default ALSA device
			"-f", "f32le",     // Output format: 32-bit float little-endian
			"-ar", "48000",    // Sample rate: 48kHz
			"-ac", "2",        // Channels: stereo
			"-bufsize", "4096", // Buffer size ajustado
			"-loglevel", "error", // Solo errores
			"-")               // Output to stdout

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			log.Printf("Error creating pipe for ffmpeg: %v", err)
			time.Sleep(time.Second)
			continue
		}

		stderr, err := cmd.StderrPipe()
		if err != nil {
			log.Printf("Error creating stderr pipe for ffmpeg: %v", err)
			time.Sleep(time.Second)
			continue
		}

		// Capturar stderr para logging
		go func() {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				log.Printf("ffmpeg: %s", scanner.Text())
			}
		}()

		if err := cmd.Start(); err != nil {
			log.Printf("Error starting ffmpeg: %v", err)
			time.Sleep(time.Second)
			continue
		}

		log.Printf("Starting system audio capture from ALSA default device")

		// Buffer for reading data
		buf := make([]byte, audioBufferSize*4) // 4 bytes per sample (float32)

		// Esperar a que ffmpeg termine
		done := make(chan error, 1)
		go func() {
			done <- cmd.Wait()
		}()

		// Leer datos hasta que ffmpeg termine
		for {
			select {
			case err := <-done:
				log.Printf("ffmpeg process ended: %v", err)
				goto restart_alsa
			default:
				n, err := stdout.Read(buf)
				if err != nil {
					if err != io.EOF {
						log.Printf("Error reading audio: %v", err)
					}
					goto restart_alsa
				}

				if n == 0 {
					time.Sleep(time.Millisecond * 1) // Mínima espera
					continue
				}

				// Convert bytes to float32 samples
				samples := make([]float32, n/4)
				for i := range samples {
					samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4]))
				}

				// Send samples to channel
				select {
				case audioChan <- samples:
				default:
					// Si el canal está lleno, descartamos las muestras
				}
			}
		}

	restart_alsa:
		log.Printf("Restarting ALSA capture...")
		time.Sleep(time.Second)
	}
}

func handleAudioConnection(conn net.Conn) {
	defer conn.Close()
	log.Printf("Starting handleAudioConnection")

	// Enviar byte de inicio de configuración
	if _, err := conn.Write([]byte{0xFF}); err != nil {
		log.Printf("Error sending config start byte: %v", err)
		return
	}

	// Enviar configuración de audio al cliente
	config := struct {
		SampleRate   uint32
		Channels     uint32
		BufferSize   uint32
	}{
		SampleRate:   uint32(audioSampleRate),
		Channels:     uint32(audioChannels),
		BufferSize:   uint32(audioBufferSize),
	}

	log.Printf("Sending audio configuration: SampleRate=%d, Channels=%d, BufferSize=%d",
		config.SampleRate, config.Channels, config.BufferSize)

	// Enviar la configuración completa en un solo write
	configBytes := make([]byte, 12) // 3 uint32 = 12 bytes
	binary.LittleEndian.PutUint32(configBytes[0:], config.SampleRate)
	binary.LittleEndian.PutUint32(configBytes[4:], config.Channels)
	binary.LittleEndian.PutUint32(configBytes[8:], config.BufferSize)

	if _, err := conn.Write(configBytes); err != nil {
		log.Printf("Error sending audio config: %v", err)
		return
	}

	// Esperar confirmación del cliente
	response := make([]byte, 1)
	if _, err := conn.Read(response); err != nil {
		log.Printf("Error reading client confirmation: %v", err)
		return
	}

	if response[0] != 0xAA {
		log.Printf("Invalid client confirmation: %v", response[0])
		return
	}

	log.Printf("Audio configuration confirmed by client")

	// Canal para recibir audio del sistema
	audioChan := make(chan []float32, 20) // Buffer moderado para mejor calidad

	// Iniciar captura de audio del sistema
	log.Printf("Starting system audio capture")
	go captureSystemAudio(audioChan)

	log.Printf("Iniciando streaming de audio...")

	for {
		// Recibir audio del sistema
		samples := <-audioChan

		// Enviar byte de inicio de chunk
		if _, err := conn.Write([]byte{0xFE}); err != nil {
			log.Printf("Error sending chunk start byte: %v", err)
			return
		}

		// Enviar tamaño del chunk
		chunkSize := uint32(len(samples) * 4) // 4 bytes por muestra (float32)
		chunkSizeBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(chunkSizeBytes, chunkSize)
		if _, err := conn.Write(chunkSizeBytes); err != nil {
			log.Printf("Error enviando tamaño de audio: %v", err)
			return
		}

		// Convertir muestras a bytes
		buf := make([]byte, chunkSize)
		for i, sample := range samples {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(sample))
		}

		// Enviar chunk de audio
		if _, err := conn.Write(buf); err != nil {
			log.Printf("Error enviando audio: %v", err)
			return
		}
	}
}

// Lanza ffmpeg para capturar la pantalla y enviar el stream al cliente por UDP
func startFFmpegScreenCapture(clientIP string) error {
	//ffmpeg -re -f x11grab -video_size 1360x768 -i :0.0 -vaapi_device /dev/dri/renderD128 -vcodec h264_vaapi -vf format=nv12|vaapi,hwupload -b:v 5M -minrate 5M -maxrate 5M -bufsize 1M -f mpegts udp://192.168.2.185:5000
	args := []string{
		"-f", "x11grab",   // Usar x11grab para capturar pantalla
		"-video_size", "1360x768", // Tamaño de la pantalla
		"-framerate", "60", // Tasa de fotogramas
		"-i", ":0.0",      // Entrada de pantalla x11grab
		"-vaapi_device", "/dev/dri/renderD128", // Dispositivo vaapi
		"-vcodec", "h264_vaapi", // Usar codificador VAAPI para H.h264_vaapi
		"-vf", "format=nv12|vaapi,hwupload", // Formato y subida a hardware
		"-r", "60", // Tasa de fotogramas de salida
		"-b:v", "8M",       // Tasa de bits de video_size
		"-minrate", "8M",   // Tasa de bits Mínima
		"-maxrate", "8M",   // Tasa de bits máxima
		"-bufsize", "4M",   // Tamaño del bufferSize
		"-f", "mpegts",     // Formato de salida: MPEG-TS
		"udp://" + clientIP + ":5000", // Dirección UDP del Cliente
	}

	//args := []string{
	//	"-re",
	//	"-f", "x11grab",
	//	"-video_size", "1360x768",
	//	"-i", ":0.0",
	//	"-vaapi_device", "/dev/dri/renderD128",
	//	"-vcodec", "h264_vaapi",
	//	"-vf", "format=nv12|vaapi,hwupload",
	//	"-f", "mpegts",
	//	"udp://" + clientIP + ":5000",
	//}
	cmd := exec.Command("ffmpeg", args...)
	cmd.Stdout = nil
	cmd.Stderr = nil
	err := cmd.Start()
	video_process[clientIP] = cmd.Process
	if err == nil {
		log.Printf("ffmpeg lanzado con h264_vaapi para %s", clientIP)
		return nil
	} else {
		log.Printf("Fallo ffmpeg con h264_vaapi: %v", err)
		return err
	}
}

func handleEventConnection(conn net.Conn) {
	defer conn.Close()
	log.Printf("Cliente conectado para eventos desde %s", conn.RemoteAddr())

	// Lanzar ffmpeg al conectar el cliente de eventos
	clientIP, _, _ := net.SplitHostPort(conn.RemoteAddr().String())
	err := startFFmpegScreenCapture(clientIP)
	if err != nil {
		log.Printf("Error lanzando ffmpeg: %v", err)
	} else {
		log.Printf("ffmpeg lanzado para %s", clientIP)
	}

	// Buffer para recibir eventos
	eventBuffer := make([]byte, 70) // 1 tipo + 1 idx + 1 axes + 1 btns + 64 name (para caps)
	var lastX, lastY int

	// Canal y goroutine para joystick
	joystickChan := make(chan [9]byte, 32)
	gamepads := make(map[byte]*vjoy.VJoy)
	jsCaps := make(map[byte]struct{axes, btns byte; name string})
	jsBtnMap := make(map[byte][]int)
	jsAxisMap := make(map[byte][]int)
	go func() {
		for raw := range joystickChan {
			idx := raw[0]
			js_event := raw[1:9]
			if _, ok := gamepads[idx]; !ok {
				caps, hasCaps := jsCaps[idx]
				if hasCaps {
					axmap := jsAxisMap[idx]
					btnmap := jsBtnMap[idx]
					vj, err := vjoy.Create(caps.name, axmap, btnmap)
					if err != nil {
						log.Printf("No se pudo crear joystick virtual para js%d: %v", idx, err)
						continue
					}
					gamepads[idx] = vj
				} else {
					vj, err := vjoy.Create(fmt.Sprintf("screenshare-js%d", idx), []int{0x00, 0x01}, []int{0x100})
					if err != nil {
						log.Printf("No se pudo crear joystick virtual para js%d: %v", idx, err)
						continue
					}
					gamepads[idx] = vj
				}
				log.Printf("Joystick virtual creado para js%d", idx)
			}
			value := int16(binary.LittleEndian.Uint16(js_event[4:6]))
			type_ := js_event[6]
			number := js_event[7]
			vj := gamepads[idx]
			if type_&0x01 != 0 {
				btnMap, ok := jsBtnMap[idx]
				if ok && int(number) < len(btnMap) {
					code := btnMap[int(number)]
					if value != 0 {
						vj.SendButton(code, 1)
					} else {
						vj.SendButton(code, 0)
					}
				} else {
					log.Printf("[JOYSTICK][WARN] Botón %d fuera de rango para js%d", number, idx)
				}
			} else if type_&0x02 != 0 {
				axisMap, ok := jsAxisMap[idx]
				if ok && int(number) < len(axisMap) {
					code := axisMap[int(number)]
					vj.SendAxis(code, int32(value))
				} else {
					log.Printf("[JOYSTICK][WARN] Eje %d fuera de rango para js%d", number, idx)
				}
			}
		}
	}()

	for {
		_, err := io.ReadFull(conn, eventBuffer[:1])
		if err != nil {
			if err != io.EOF {
				log.Printf("Error leyendo tipo de evento: %v", err)
			}
			close(joystickChan)
			return
		}
		eventType := eventBuffer[0]

		if eventType == EVENT_JOYSTICK_CAPS {
			// Leer idx, num_axes, num_btns, nombre (64 bytes)
			_, err := io.ReadFull(conn, eventBuffer[1:68])
			if err != nil {
				log.Printf("Error leyendo caps de joystick: %v", err)
				continue
			}
			idx := eventBuffer[1]
			axes := eventBuffer[2]
			btns := eventBuffer[3]
			name := string(eventBuffer[4:68])
			name = strings.TrimRight(name, "\x00")
			// Leer arrays de códigos de ejes y botones
			axmap := make([]int, axes)
			btnmap := make([]int, btns)
			if axes > 0 {
				axbuf := make([]byte, int(axes))
				_, err := io.ReadFull(conn, axbuf)
				if err != nil {
					log.Printf("Error leyendo axmap: %v", err)
					continue
				}
				for i := 0; i < int(axes); i++ {
					axmap[i] = int(axbuf[i])
				}
				log.Printf("axmap (Go): %v", axmap)
			}
			if btns > 0 {
				btnbuf := make([]byte, 2*int(btns))
				_, err := io.ReadFull(conn, btnbuf)
				if err != nil {
					log.Printf("Error leyendo btnmap: %v", err)
					continue
				}
				for i := 0; i < int(btns); i++ {
					btnmap[i] = int(binary.LittleEndian.Uint16(btnbuf[i*2:(i+1)*2]))
				}
				log.Printf("btnmap (Go): %v", btnmap)
			}
			jsCaps[idx] = struct{axes, btns byte; name string}{axes, btns, name}
			jsAxisMap[idx] = axmap
			jsBtnMap[idx] = btnmap
			log.Printf("[JOYSTICK_CAPS] idx=%d axes=%d btns=%d name=%s axmap=%v btnmap=%v", idx, axes, btns, name, axmap, btnmap)
			continue
		}

		if eventType == EVENT_JOYSTICK {
			_, err := io.ReadFull(conn, eventBuffer[1:10])
			if err != nil {
				log.Printf("Error leyendo evento joystick: %v", err)
				close(joystickChan)
				return
			}
			var raw [9]byte
			raw[0] = eventBuffer[1] // idx
			copy(raw[1:], eventBuffer[2:10]) // js_event
			joystickChan <- raw
			continue
		}

		switch eventType {
		case EVENT_KEYDOWN, EVENT_KEYUP:
			// Leer 4 bytes de datos
			_, err := io.ReadFull(conn, eventBuffer[1:5])
			if err != nil {
				log.Printf("Error leyendo datos de evento de teclado: %v", err)
				return
			}
			keyCode := binary.LittleEndian.Uint32(eventBuffer[1:5])
			keyName := getKeyName(keyCode)
			if eventType == EVENT_KEYDOWN {
				log.Printf("Tecla presionada: %s", keyName)
				robotgo.KeyDown(keyName)
			} else {
				log.Printf("Tecla liberada: %s", keyName)
				robotgo.KeyUp(keyName)
			}
		case EVENT_MOUSEMOTION:
			// Leer 8 bytes: x, y, rel_x, rel_y (int16)
			_, err := io.ReadFull(conn, eventBuffer[1:9])
			if err != nil {
				log.Printf("Error leyendo datos de mouse motion: %v", err)
				return
			}
			x := int(int16(binary.LittleEndian.Uint16(eventBuffer[1:3])))
			y := int(int16(binary.LittleEndian.Uint16(eventBuffer[3:5])))
			rel_x := int(int16(binary.LittleEndian.Uint16(eventBuffer[5:7])))
			rel_y := int(int16(binary.LittleEndian.Uint16(eventBuffer[7:9])))
			// Usar movimiento relativo
			if rel_x != 0 || rel_y != 0 {
				virtualMouse.Move(int32(rel_x), int32(rel_y))
			}
			lastX, lastY = x, y
		case EVENT_MOUSEBUTTONDOWN, EVENT_MOUSEBUTTONUP:
			// Leer 5 bytes: button (uint8), x, y (int16)
			_, err := io.ReadFull(conn, eventBuffer[1:6])
			if err != nil {
				log.Printf("Error leyendo datos de mouse button: %v", err)
				return
			}
			button := eventBuffer[1]
			x := int(int16(binary.LittleEndian.Uint16(eventBuffer[2:4])))
			y := int(int16(binary.LittleEndian.Uint16(eventBuffer[4:6])))
			// Mover mouse relativo si es necesario
			dx := x - lastX
			dy := y - lastY
			if dx != 0 || dy != 0 {
				virtualMouse.Move(int32(dx), int32(dy))
				lastX, lastY = x, y
			}
			if button == 1 {
				if eventType == EVENT_MOUSEBUTTONDOWN {
					virtualMouse.LeftPress()
				} else {
					virtualMouse.LeftRelease()
				}
			} else if button == 3 {
				if eventType == EVENT_MOUSEBUTTONDOWN {
					virtualMouse.RightPress()
				} else {
					virtualMouse.RightRelease()
				}
			} else if button == 2 {
				if eventType == EVENT_MOUSEBUTTONDOWN {
					virtualMouse.MiddlePress()
				} else {
					virtualMouse.MiddleRelease()
				}
			}
		case EVENT_MOUSEWHEEL:
			// Leer 4 bytes: x, y (int16)
			_, err := io.ReadFull(conn, eventBuffer[1:5])
			if err != nil {
				log.Printf("Error leyendo datos de mouse wheel: %v", err)
				return
			}
			x := int(int16(binary.LittleEndian.Uint16(eventBuffer[1:3])))
			y := int(int16(binary.LittleEndian.Uint16(eventBuffer[3:5])))
			if y != 0 {
				virtualMouse.Wheel(false, int32(y))
			}
			if x != 0 {
				virtualMouse.Wheel(true, int32(x))
			}
		}
	}
}

func CheckErr(err error, text string){
	if err != nil{
		println("ERROR:", text, err.Error())
		os.Exit(1)
	}
}

func GetDefaultSink() string{
	cmd := exec.Command("pactl", "info")
	stdout, err := cmd.Output()
	CheckErr(err, "Error getting default sink")
	sink := []byte("")
	for _, v := range bytes.Split(stdout, []byte("\n")){
		if bytes.Contains(v, []byte("Default Sink:")) {
			sink = bytes.TrimSpace(bytes.Split(v, []byte(":"))[1])
		}
	}
	return string(sink) + ".monitor"
}

func main() {
	// Obtener la IP local
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatal(err)
	}
	var localIP string
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				localIP = ipnet.IP.String()
				break
			}
		}
	}

	if localIP == "" {
		localIP = "127.0.0.1"
	}

	// Iniciar el servidor TCP para frames
	log.Printf("Starting frame server on %s%s", localIP, framePort)
	frameListener, err := net.Listen("tcp", framePort)
	if err != nil {
		log.Fatal(err)
	}
	defer frameListener.Close()

	// Iniciar el servidor TCP para eventos
	log.Printf("Starting event server on %s%s", localIP, eventPort)
	eventListener, err := net.Listen("tcp", eventPort)
	if err != nil {
		log.Fatal(err)
	}
	defer eventListener.Close()

	// Iniciar el servidor TCP para audio
	//log.Printf("Starting audio server on %s%s", localIP, audioPort)
	//audioListener, err := net.Listen("tcp", audioPort)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//defer audioListener.Close()

	log.Printf("Servidor iniciado en %s%s (frames), %s%s (eventos) y %s%s (audio)",
		localIP, framePort, localIP, eventPort, localIP, audioPort)
	log.Println("Esperando conexión del cliente...")

	// WaitGroup para mantener el programa en ejecución
	var wg sync.WaitGroup
	wg.Add(3) // Para los tres servidores
	chan_addr := make(chan string, 1)

	// Crear mouse virtual global
	virtualMouse, err = uinput.CreateMouse("/dev/uinput", []byte("screenshare-mouse"))
	if err != nil {
		log.Fatalf("No se pudo crear mouse virtual: %v", err)
	}
	defer virtualMouse.Close()

	// Aceptar conexiones de frames
	go func() {
		defer wg.Done()
		for {
			conn, err := frameListener.Accept()
			if err != nil {
				log.Printf("Error al aceptar conexión de frames: %v", err)
				continue
			}
			log.Printf("Cliente conectado para frames desde %s", conn.RemoteAddr())
			go handleConnection(conn)
		}
	}()


	go func() {
		defer wg.Done()
		for {
			client_addr := strings.Split(<-chan_addr, ":")[0]
			// ffmpeg -re -f pulse -i alsa_output.pci-0000_08_00.1.hdmi-stereo-extra3.monitor -f rtsp rtsp://192.168.2.192:8888
			cmd := exec.Command(
				"ffmpeg",
				"-re",             // Leer entrada a la tasa de fotogramas nativa
				"-xerror",
				"-f", "pulse",     // Usar PulseAudio
				"-i", GetDefaultSink(), // Usar el monitor de la salida de audio por defecto
				"-f", "rtsp",       // Formato de salida: rtp
				"rtsp://"+client_addr+":8888", // Dirección RTSP del cliente
			)
			time.Sleep(1e9 * 4)
			go func (cmd *exec.Cmd, client_addr string) {
				err = cmd.Start()
				if err != nil {
					log.Printf("Error al iniciar ffmpeg para RTSP: %v", err)
				} else {
					log.Printf("ffmpeg lanzado para RTSP en %s", client_addr)
				}
				cmd.Wait()
				video_process[client_addr].Kill()

			} (cmd, client_addr)

		}
	}()

	// Aceptar conexiones de audio
	//go func() {
	//	defer wg.Done()
	//	for {
	//		conn, err := audioListener.Accept()
	//		if err != nil {
	//			log.Printf("Error al aceptar conexión de audio: %v", err)
	//			continue
	//		}
	//		log.Printf("Cliente conectado para audio desde %s", conn.RemoteAddr())
	//		go handleAudioConnection(conn)
	//	}
	//}()

	// Aceptar conexiones de eventos
	go func() {
		defer wg.Done()
		for {
			conn, err := eventListener.Accept()
			chan_addr <- conn.RemoteAddr().String()
			if err != nil {
				log.Printf("Error al aceptar conexión de eventos: %v", err)
				continue
			}
			log.Printf("Cliente conectado para eventos desde %s", conn.RemoteAddr())
			go handleEventConnection(conn)
		}
	}()

	// Esperar indefinidamente
	wg.Wait()
}
