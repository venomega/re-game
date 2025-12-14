package main

import (
	"io"
	"net"
)


func AudioSender(fd  io.ReadCloser, client_addr string){
			println("AudioSender reached")
			stdout_buffer := make([]byte, 44100)
			udp_addr, err := net.ResolveUDPAddr("udp", client_addr + ":8889")
			CheckErr(err, "cannot resolve udp address:" + client_addr + ":8889")

			conn, err := net.DialUDP("udp", nil, udp_addr)
			CheckErr(err, "cannot connect to udp address:" + client_addr + ":8889")
			println("AudioSender ready")
			for{
				n, err := fd.Read(stdout_buffer)
				if err != nil{
					println("AudioSender exit", err.Error())
					break
				}
				n, err = conn.Write(stdout_buffer[:n])
				if err != nil{
					println("AudioSender exit", err.Error())
					break
				}
			}
			err = video_process[client_addr].Kill()
				if err != nil{
					println("kill video_proces", client_addr, err.Error())
				}

}
