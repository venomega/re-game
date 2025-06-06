package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"log"
	"net"
	"runtime"
	"sync"
	"time"

	"github.com/kbinani/screenshot"
	"github.com/go-vgo/robotgo"
)

const (
	frameRate = 240  // Aumentamos el framerate objetivo
	frameDuration = time.Second / frameRate
	framePort = ":8080"
	eventPort = ":8081"
	maxBufferSize = 1024 * 1024 // 1MB buffer
	bufferSize = 60 // Buffer de 60 frames
	jpegQuality = 60 // Reducimos calidad JPEG para mayor velocidad
	eventBufferSize = 1000 // Tamaño del buffer de eventos
)

// Constantes para tipos de eventos
const (
	EVENT_KEYDOWN = 1
	EVENT_KEYUP = 2
	EVENT_MOUSEMOTION = 3
	EVENT_MOUSEBUTTONDOWN = 4
	EVENT_MOUSEBUTTONUP = 5
	EVENT_MOUSEWHEEL = 6
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

func init() {
	// Configurar Go para usar todos los núcleos disponibles
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func handleEvents(conn net.Conn) {
	defer conn.Close()

	// Crear buffer de eventos
	eventBuffer := NewEventBuffer(eventBufferSize)

	// Iniciar thread de procesamiento de eventos
	go func() {
		for {
			event := eventBuffer.Get()
			switch event.Type {
			case EVENT_KEYDOWN:
				keyCode := binary.LittleEndian.Uint32(event.Data)
				robotgo.KeyDown(string(keyCode))
			case EVENT_KEYUP:
				keyCode := binary.LittleEndian.Uint32(event.Data)
				robotgo.KeyUp(string(keyCode))
			case EVENT_MOUSEMOTION:
				x := int(binary.LittleEndian.Uint32(event.Data[:4]))
				y := int(binary.LittleEndian.Uint32(event.Data[4:8]))
				robotgo.MoveMouse(x, y)
			case EVENT_MOUSEBUTTONDOWN:
				button := event.Data[0]
				switch button {
				case 1: // Izquierdo
					robotgo.MouseDown("left")
				case 2: // Medio
					robotgo.MouseDown("center")
				case 3: // Derecho
					robotgo.MouseDown("right")
				}
			case EVENT_MOUSEBUTTONUP:
				button := event.Data[0]
				switch button {
				case 1: // Izquierdo
					robotgo.MouseUp("left")
				case 2: // Medio
					robotgo.MouseUp("center")
				case 3: // Derecho
					robotgo.MouseUp("right")
				}
			case EVENT_MOUSEWHEEL:
				scroll := int(binary.LittleEndian.Uint32(event.Data))
				if scroll > 0 {
					robotgo.Scroll(scroll, 0)  // 0 para arriba
				} else {
					robotgo.Scroll(-scroll, 1)  // 1 para abajo
				}
			}
		}
	}()

	// Buffer para recibir eventos
	buf := make([]byte, 1024)

	for {
		// Recibir tipo de evento
		_, err := conn.Read(buf[:1])
		if err != nil {
			log.Printf("Error recibiendo tipo de evento: %v", err)
			return
		}

		eventType := buf[0]
		var eventData []byte

		// Recibir datos del evento según el tipo
		switch eventType {
		case EVENT_KEYDOWN, EVENT_KEYUP:
			_, err := conn.Read(buf[:4])
			if err != nil {
				log.Printf("Error recibiendo código de tecla: %v", err)
				return
			}
			eventData = make([]byte, 4)
			copy(eventData, buf[:4])
		case EVENT_MOUSEMOTION:
			_, err := conn.Read(buf[:8])
			if err != nil {
				log.Printf("Error recibiendo coordenadas del ratón: %v", err)
				return
			}
			eventData = make([]byte, 8)
			copy(eventData, buf[:8])
		case EVENT_MOUSEBUTTONDOWN, EVENT_MOUSEBUTTONUP:
			_, err := conn.Read(buf[:1])
			if err != nil {
				log.Printf("Error recibiendo botón del ratón: %v", err)
				return
			}
			eventData = make([]byte, 1)
			copy(eventData, buf[:1])
		case EVENT_MOUSEWHEEL:
			_, err := conn.Read(buf[:4])
			if err != nil {
				log.Printf("Error recibiendo dirección de la rueda: %v", err)
				return
			}
			eventData = make([]byte, 4)
			copy(eventData, buf[:4])
		}

		// Agregar evento al buffer
		eventBuffer.Add(Event{Type: eventType, Data: eventData})
	}
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
	frameListener, err := net.Listen("tcp", framePort)
	if err != nil {
		log.Fatal(err)
	}
	defer frameListener.Close()

	// Iniciar el servidor TCP para eventos
	eventListener, err := net.Listen("tcp", eventPort)
	if err != nil {
		log.Fatal(err)
	}
	defer eventListener.Close()

	log.Printf("Servidor iniciado en %s%s (frames) y %s%s (eventos)", localIP, framePort, localIP, eventPort)
	log.Println("Esperando conexión del cliente...")

	// Aceptar conexiones de eventos
	go func() {
		for {
			conn, err := eventListener.Accept()
			if err != nil {
				log.Printf("Error al aceptar conexión de eventos: %v", err)
				continue
			}
			log.Printf("Cliente conectado para eventos desde %s", conn.RemoteAddr())
			go handleEvents(conn)
		}
	}()

	// Aceptar conexiones de frames
	for {
		conn, err := frameListener.Accept()
		if err != nil {
			log.Printf("Error al aceptar conexión de frames: %v", err)
			continue
		}

		log.Printf("Cliente conectado para frames desde %s", conn.RemoteAddr())
		go handleConnection(conn)
	}
}

func handleConnection(conn net.Conn) {
	defer conn.Close()

	// Obtener las dimensiones de la pantalla
	bounds := screenshot.GetDisplayBounds(0)
	width := bounds.Dx()
	height := bounds.Dy()

	// Enviar dimensiones al cliente
	binary.Write(conn, binary.LittleEndian, uint32(width))
	binary.Write(conn, binary.LittleEndian, uint32(height))

	// Crear buffer de frames
	frameBuffer := NewFrameBuffer(bufferSize)

	// Mutex para sincronizar la captura de pantalla
	var captureMutex sync.Mutex

	// Iniciar thread de captura
	go func() {
		for {
			// Sincronizar acceso a la captura de pantalla
			captureMutex.Lock()
			img, err := screenshot.CaptureDisplay(0)
			captureMutex.Unlock()

			if err != nil {
				log.Printf("Error capturando pantalla: %v", err)
				continue
			}

			frameBuffer.Add(img)
			time.Sleep(frameDuration)
		}
	}()

	frameCount := 0
	lastFPS := time.Now()
	lastFrameTime := time.Now()

	// Buffer pre-asignado para JPEG
	jpegBuffer := make([]byte, maxBufferSize)

	for {
		startTime := time.Now()

		// Obtener frame del buffer
		frame := frameBuffer.Get()

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

		// Mantener el framerate
		elapsed := time.Since(startTime)
		if elapsed < frameDuration {
			time.Sleep(frameDuration - elapsed)
		}
	}
}
