# ESP32 cue receiver

The ESP32 side of the top-edge cue (`TopEdgeCue` in `experiment/render_logic.py`). The experiment PC will send
one short message over Bluetooth LE each time the cue fires, that is, when the finger crosses a bar's top edge,
entering or leaving. **For now the ESP32 only shows that a message arrived.** There is no motor yet, and nothing on
the PC sends these messages yet.

The board is an **ESP32-C3 Super Mini**, and the PC connects to it directly over Bluetooth. The Arduino tools below
are only used to put this program on the board; there is no Arduino board in between.

On every valid message the ESP32:

- raises **GPIO 4** (`CUE_PIN`) for 2 ms, for the oscilloscope;
- lights the blue LED (GPIO 8) for 0.1 s;
- sends the message back on the ACK characteristic, with its own receive time added;
- prints a line on the serial monitor (115200 baud), for example `cue 1 entering left at 5123456 us`.

## Setup and upload on the lab PC (Windows, PowerShell)

Once per computer: install the Arduino command line tool. Then **open a new PowerShell window**, so that the
`arduino-cli` command is found.

```powershell
winget install --id ArduinoSA.CLI -e
```

Once per computer: install the ESP32 board package (a few hundred MB).

```powershell
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
```

Plug the ESP32 in by USB and find its port (`COM3`, `COM5`, ...):

```powershell
arduino-cli board list
```

Each time: in the repository folder, set the port, then compile, upload and open the serial monitor. Change
`COM5` to the port shown above. `nologo_esp32c3_super_mini` is the ESP32 core's own entry for this board. With it,
the serial monitor works over the board's USB port with no other setting.

```powershell
$PORT = "COM5"
$FQBN = "esp32:esp32:nologo_esp32c3_super_mini"
arduino-cli compile --fqbn $FQBN firmware/esp32_cue
arduino-cli upload -p $PORT --fqbn $FQBN firmware/esp32_cue
arduino-cli monitor -p $PORT --config baudrate=115200
```

The monitor should print `EA-CUE advertising; cue pin GPIO 4`. Close it with Ctrl+C.

- **No port appears, or the upload cannot connect:** hold the **BOOT** button while plugging the USB cable in,
  release it, and upload again. Then press **RST** (or plug the cable in again) to start the program. The board
  has no separate USB-serial chip, so no driver is needed.
- **GPIO 4** (marked SCK on the board) is free as long as nothing else is wired to it. To use another pin, change
  `CUE_PIN` at the top of the sketch. Avoid GPIO 2, 8 and 9: they decide how the chip starts, GPIO 8 drives the
  LED and GPIO 9 is the BOOT button.

The Arduino IDE works as well: install **esp32 by Espressif Systems** from its Boards Manager, open
`esp32_cue.ino`, choose the board **Nologo ESP32C3 Super Mini** and its port, upload, and open the Serial Monitor
at 115200 baud.

## Bluetooth LE layout

| | UUID | |
| --- | --- | --- |
| Service | `cfdd2bb4-1857-421a-9450-e9c8a92ea3f9` | advertised, device name `EA-CUE` |
| CUE | `03c34c2a-b29f-4040-b812-5778ed2507cd` | PC → ESP32, write with or without response |
| ACK | `34c5a9d1-fb6d-445f-b9d9-e6fe0eac7136` | ESP32 → PC, notify |

## Message format

The messages travel over Bluetooth only. The USB cable is used just to upload the sketch and to read the text
lines on the serial monitor; no cue message goes over it.

One message per write, exactly 4 bytes. Anything of another length is ignored and reported on the serial
monitor.

| Byte | Meaning |
| --- | --- |
| 0–1 | sequence number, unsigned 16-bit, little-endian (low byte first) |
| 2 | kind: `0` test, `1` entering, `2` leaving |
| 3 | bar: `0` left, `1` right |

The sequence number counts the messages, so a lost one shows up as a gap. It goes back to 0 after 65535.

The ACK notification is the same 4 bytes followed by the ESP32's `micros()` at receipt, unsigned 32-bit,
little-endian (8 bytes in all). Test messages are for the latency check. They will not drive the motor once
there is one.

Worked example: message number 300, the finger entering the right bar.

| | Bytes (hex) | |
| --- | --- | --- |
| PC → ESP32 | `2C 01` `01` `01` | 300 = 0x012C, low byte first; entering; right |
| ESP32 → PC | `2C 01 01 01` `80 2D 4E 00` | the same 4 bytes; received at 5 123 456 µs = 0x004E2D80, low byte first |

## Check with a phone, before any PC code

With nRF Connect (Android or iOS):

1. Scan, find `EA-CUE`, connect.
2. On the ACK characteristic (`34c5a9d1-…`), enable notifications.
3. On the CUE characteristic (`03c34c2a-…`), write the bytes (hex) `01 00 01 00`.
4. Expected: the LED lights briefly, the serial monitor prints `cue 1 entering left at … us`, and a notification
   `01-00-01-00-…` with 4 more bytes arrives.
5. Write `02 00 02 01`. Expected: `cue 2 leaving right`.

## Not done yet

- The motor and its pulse pattern.
- The PC sender (`experiment/cue_link.py`) and its latency check.
- A request for a short connection interval. The PC (Windows) picks the interval, and the latency check will show
  whether asking for a shorter one is worth it.
