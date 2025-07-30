#include "vjoy.h"
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <errno.h>

int vjoy_create(const char* name, const int* axes, int num_axes, const int* buttons, int num_buttons) {
    struct uinput_user_dev uidev;
    int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) { perror("open /dev/uinput"); return -1; }

    // Habilitar ejes
    for (int i = 0; i < num_axes; ++i) {
        if (axes[i] < 0 || axes[i] >= ABS_CNT) {
            fprintf(stderr, "[vjoy] Eje fuera de rango: %d (0x%x)\n", axes[i], axes[i]);
            continue;
        }
        if (ioctl(fd, UI_SET_ABSBIT, axes[i]) < 0) {
            fprintf(stderr, "[vjoy] ioctl UI_SET_ABSBIT fallo para eje %d (code 0x%x): %s\n", i, axes[i], strerror(errno));
            goto error;
        }
        fprintf(stderr, "[vjoy] Eje habilitado: %d (0x%x)\n", axes[i], axes[i]);
    }
    // Habilitar botones
    for (int i = 0; i < num_buttons; ++i) {
        if (buttons[i] < 0 || buttons[i] >= KEY_CNT) {
            fprintf(stderr, "[vjoy] Botón fuera de rango: %d (0x%x)\n", buttons[i], buttons[i]);
            continue;
        }
        if (ioctl(fd, UI_SET_KEYBIT, buttons[i]) < 0) {
            fprintf(stderr, "[vjoy] ioctl UI_SET_KEYBIT fallo para botón %d (code 0x%x): %s\n", i, buttons[i], strerror(errno));
            goto error;
        }
        fprintf(stderr, "[vjoy] Botón habilitado: %d (0x%x)\n", buttons[i], buttons[i]);
    }
    if (ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0) { perror("ioctl EV_KEY"); goto error; }
    if (ioctl(fd, UI_SET_EVBIT, EV_ABS) < 0) { perror("ioctl EV_ABS"); goto error; }

    memset(&uidev, 0, sizeof(uidev));
    snprintf(uidev.name, UINPUT_MAX_NAME_SIZE, "%s", name);
    uidev.id.bustype = BUS_USB;
    uidev.id.vendor  = 0x1234;
    uidev.id.product = 0x5678;
    uidev.id.version = 1;
    // Configurar rangos de ejes
    for (int i = 0; i < num_axes; ++i) {
        if (axes[i] >= 0 && axes[i] < ABS_CNT) {
            uidev.absmin[axes[i]] = -32768;
            uidev.absmax[axes[i]] = 32767;
            uidev.absflat[axes[i]] = 128;
        }
    }
    if (write(fd, &uidev, sizeof(uidev)) < 0) { perror("write uidev"); goto error; }
    if (ioctl(fd, UI_DEV_CREATE) < 0) { perror("ioctl UI_DEV_CREATE"); goto error; }
    usleep(100000); // Esperar a que el dispositivo esté listo
    return fd;
error:
    close(fd);
    return -1;
}

int vjoy_send_axis(int fd, int axis_code, int32_t value) {
    struct input_event ev = {0};
    ev.type = EV_ABS;
    ev.code = axis_code;
    ev.value = value;
    if (write(fd, &ev, sizeof(ev)) < 0) return -1;
    // Sync
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    if (write(fd, &ev, sizeof(ev)) < 0) return -1;
    return 0;
}

int vjoy_send_button(int fd, int button_code, int value) {
    struct input_event ev = {0};
    ev.type = EV_KEY;
    ev.code = button_code;
    ev.value = value;
    if (write(fd, &ev, sizeof(ev)) < 0) return -1;
    // Sync
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    if (write(fd, &ev, sizeof(ev)) < 0) return -1;
    return 0;
}

void vjoy_close(int fd) {
    if (fd >= 0) {
        ioctl(fd, UI_DEV_DESTROY);
        close(fd);
    }
} 