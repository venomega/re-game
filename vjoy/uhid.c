#include "uhid.h"
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <linux/uhid.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdint.h>

// Descriptor HID del DualSense (recortado para ejemplo, debe ser el real para compatibilidad total)
static const uint8_t dualsense_hid_report_descriptor[] = {
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x05, // Usage (Game Pad)
    0xA1, 0x01, // Collection (Application)
    0x15, 0x00, // Logical Minimum (0)
    0x25, 0x01, // Logical Maximum (1)
    0x35, 0x00, // Physical Minimum (0)
    0x45, 0x01, // Physical Maximum (1)
    0x75, 0x01, // Report Size (1)
    0x95, 0x10, // Report Count (16 buttons)
    0x05, 0x09, // Usage Page (Button)
    0x19, 0x01, // Usage Minimum (Button 1)
    0x29, 0x10, // Usage Maximum (Button 16)
    0x81, 0x02, // Input (Data,Var,Abs)
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x25, 0x07, // Logical Maximum (7)
    0x46, 0x3B, 0x01, // Physical Maximum (315)
    0x75, 0x04, // Report Size (4)
    0x95, 0x01, // Report Count (1)
    0x65, 0x14, // Unit (Eng Rot:Angular Pos)
    0x09, 0x39, // Usage (Hat switch)
    0x81, 0x42, // Input (Data,Var,Abs,Null)
    0x65, 0x00, // Unit (None)
    0x95, 0x01, // Report Count (1)
    0x75, 0x04, // Report Size (4)
    0x81, 0x03, // Input (Const,Var,Abs)
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x30, // Usage (X)
    0x09, 0x31, // Usage (Y)
    0x09, 0x32, // Usage (Z)
    0x09, 0x35, // Usage (Rz)
    0x15, 0x00, // Logical Minimum (0)
    0x26, 0xFF, 0x00, // Logical Maximum (255)
    0x75, 0x08, // Report Size (8)
    0x95, 0x04, // Report Count (4)
    0x81, 0x02, // Input (Data,Var,Abs)
    0xC0 // End Collection
};

int uhid_create_dualsense(const char* name) {
    int fd = open("/dev/uhid", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    struct uhid_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = UHID_CREATE2;
    snprintf((char*)ev.u.create2.name, sizeof(ev.u.create2.name), "%s", name);
    snprintf((char*)ev.u.create2.uniq, sizeof(ev.u.create2.uniq), "remotedualsense");
    ev.u.create2.rd_size = sizeof(dualsense_hid_report_descriptor);
    memcpy(ev.u.create2.rd_data, dualsense_hid_report_descriptor, sizeof(dualsense_hid_report_descriptor));
    ev.u.create2.bus = BUS_USB;
    ev.u.create2.vendor = 0x054C; // Sony
    ev.u.create2.product = 0x0CE6; // DualSense
    ev.u.create2.version = 0x0100;
    ev.u.create2.country = 0;
    if (write(fd, &ev, sizeof(ev)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int uhid_send_report(int fd, const uint8_t* report, int len) {
    struct uhid_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = UHID_INPUT2;
    if (len > sizeof(ev.u.input2.data)) len = sizeof(ev.u.input2.data);
    ev.u.input2.size = len;
    memcpy(ev.u.input2.data, report, len);
    if (write(fd, &ev, sizeof(ev)) < 0) return -1;
    return 0;
}

void uhid_close(int fd) {
    if (fd >= 0) {
        struct uhid_event ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = UHID_DESTROY;
        write(fd, &ev, sizeof(ev));
        close(fd);
    }
} 