#ifndef VJOY_H
#define VJOY_H

#include <linux/uinput.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Crea un joystick virtual. axes y buttons son arrays de códigos ABS_*/BTN_*
// Devuelve el fd del dispositivo, o -1 en error.
int vjoy_create(const char* name, const int* axes, int num_axes, const int* buttons, int num_buttons);

// Envía un evento de eje (axis_code: ABS_*, value: -32768..32767)
int vjoy_send_axis(int fd, int axis_code, int32_t value);

// Envía un evento de botón (button_code: BTN_*, value: 0/1)
int vjoy_send_button(int fd, int button_code, int value);

// Libera el dispositivo
void vjoy_close(int fd);

#ifdef __cplusplus
}
#endif

#endif // VJOY_H 