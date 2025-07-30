package vjoy

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -lrt
#include "uhid.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

type UHID struct {
	fd C.int
}

func CreateDualSense(name string) (*UHID, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	fd := C.uhid_create_dualsense(cname)
	if fd < 0 {
		return nil, ErrCreateFailed
	}
	return &UHID{fd: fd}, nil
}

func (u *UHID) SendReport(report []byte) error {
	if len(report) == 0 {
		return nil
	}
	ret := C.uhid_send_report(u.fd, (*C.uint8_t)(unsafe.Pointer(&report[0])), C.int(len(report)))
	if ret != 0 {
		return ErrSendFailed
	}
	return nil
}

func (u *UHID) Close() {
	C.uhid_close(u.fd)
} 