#ifndef UHID_H
#define UHID_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Crea un dispositivo HID virtual DualSense. Devuelve el fd de uhid, o -1 en error.
int uhid_create_dualsense(const char* name);

// Envía un input report HID (de tamaño len) al dispositivo virtual
int uhid_send_report(int fd, const uint8_t* report, int len);

// Cierra el dispositivo
void uhid_close(int fd);

#ifdef __cplusplus
}
#endif

#endif // UHID_H 