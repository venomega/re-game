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
)

const (
	frameRate = 120  // Aumentamos el framerate objetivo
	frameDuration = time.Second / frameRate
	port = ":8080"
	maxBufferSize = 1024 * 1024 // 1MB buffer
	bufferSize = 30 // Buffer de 30 frames
	jpegQuality = 80 // Calidad JPEG
)

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

	// Iniciar el servidor TCP
	listener, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatal(err)
	}
	defer listener.Close()

	log.Printf("Servidor iniciado en %s%s", localIP, port)
	log.Println("Esperando conexión del cliente...")

	// Aceptar conexiones
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Printf("Error al aceptar conexión: %v", err)
			continue
		}

		log.Printf("Cliente conectado desde %s", conn.RemoteAddr())
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

	// Iniciar captura de pantalla en un thread separado
	go func() {
		for {
			img, err := screenshot.CaptureDisplay(0)
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

	for {
		startTime := time.Now()

		// Obtener frame del buffer
		frame := frameBuffer.Get()

		// Comprimir frame a JPEG
		buf := new(bytes.Buffer)
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
