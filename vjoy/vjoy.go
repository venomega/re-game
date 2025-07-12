package vjoy

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -lrt
#include "vjoy.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
	"errors"
)

type VJoy struct {
	fd C.int
}

func Create(name string, axes []int, buttons []int) (*VJoy, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var caxes *C.int
	var cbuttons *C.int

	if len(axes) > 0 {
		caxesArr := make([]C.int, len(axes))
		for i, v := range axes {
			caxesArr[i] = C.int(v)
		}
		caxes = &caxesArr[0]
	}
	if len(buttons) > 0 {
		cbuttonsArr := make([]C.int, len(buttons))
		for i, v := range buttons {
			cbuttonsArr[i] = C.int(v)
		}
		cbuttons = &cbuttonsArr[0]
	}

	fd := C.vjoy_create(cname, caxes, C.int(len(axes)), cbuttons, C.int(len(buttons)))
	if fd < 0 {
		return nil, ErrCreateFailed
	}
	return &VJoy{fd: fd}, nil
}

func (v *VJoy) SendAxis(axisCode int, value int32) error {
	ret := C.vjoy_send_axis(v.fd, C.int(axisCode), C.int32_t(value))
	if ret != 0 {
		return ErrSendFailed
	}
	return nil
}

func (v *VJoy) SendButton(buttonCode int, value int) error {
	ret := C.vjoy_send_button(v.fd, C.int(buttonCode), C.int(value))
	if ret != 0 {
		return ErrSendFailed
	}
	return nil
}

func (v *VJoy) Close() {
	C.vjoy_close(v.fd)
}


var (
	ErrCreateFailed = errors.New("vjoy: failed to create virtual joystick")
	ErrSendFailed   = errors.New("vjoy: failed to send event")
)
