# ESP32 cue receiver

The ESP32 side of the top-edge cue (`TopEdgeCue` in `experiment/render_logic.py`). Each time the cue fires, that
is, each time the finger crosses a bar's top edge (entering or leaving), the experiment PC will send one short text
message over Bluetooth LE. The ESP32 then plays a pulse on the **Haptuator Mark II** through a **MAX98357A**
amplifier. On the PC, `experiment/cue_link.py` sends them during a session with `cue_enabled` set, and
`python -m experiment.cue_link` sends test messages (`experiment/README.md`, *Top-edge cue (Haptuator)*). A phone
can send them too (see the check at the end).

The board is an **ESP32-C3 Super Mini**, and the PC connects to it directly over Bluetooth. The Arduino tools below
are only used to put this program on the board; there is no Arduino board in between.

On every valid message the ESP32:

- plays the pulse pattern on the Haptuator;
- raises **GPIO 4** (`MARK_PIN`) for 2 ms, for the oscilloscope;
- lights the blue LED (GPIO 8) for 0.1 s;
- prints the message on the serial monitor (115200 baud), for example `EXIT_LEFT at 5123456 us`.

## Message format

The messages travel over Bluetooth only. The USB cable is used just to upload the sketch and to read the serial
monitor; no cue message goes over it.

One message per write: a line of text ending in a newline (`\n`). The newline may be left out.

| Message | Sent when the finger crosses the top edge of | ESP32 |
| --- | --- | --- |
| `ENTER_LEFT` | the left bar, going down into the bar | plays the pulse |
| `EXIT_LEFT` | the left bar, going up out of the bar | plays the pulse |
| `ENTER_RIGHT` | the right bar, going down into the bar | plays the pulse |
| `EXIT_RIGHT` | the right bar, going up out of the bar | plays the pulse |
| no message | the finger stays on one side of the top edge, or leaves a bar through its side | nothing |

The PC sends at most one message per crossing. `TopEdgeCue` decides when: a bar is not re-armed until the finger
is 3 mm away from its top edge, two messages are at least 100 ms apart, and a jump longer than 20 mm is taken as a
lift, not a crossing. Anything else that arrives is ignored and reported on the serial monitor.

There is no sequence number and no reply, so the PC cannot tell whether a message was lost. Latency is measured
with the oscilloscope: GPIO 4 or the Haptuator signal against the EA signal.

## Haptuator and amplifier

The Haptuator Mark II (TL002-09-A) is a voice coil: 5.5 Ω, at most **3.0 V** and 0.5 A, rated 90–1000 Hz. It is
driven like a small loudspeaker. **Never connect it to an ESP32 pin directly.** The pin cannot supply the current
and would be damaged. The ESP32-C3 has no analog audio output, so the sketch sends the signal digitally (I2S) to
the MAX98357A.

| ESP32-C3 Super Mini | MAX98357A |
| --- | --- |
| GPIO 5 | BCLK |
| GPIO 6 | LRC |
| GPIO 7 | DIN |
| 5V (or the battery) | Vin |
| GND | GND |
| | `+` and `−` outputs → the Haptuator's two pins |

Leave the amplifier's GAIN and SD pins unconnected. To use other pins, change `I2S_BCLK_PIN`, `I2S_LRC_PIN` and
`I2S_DIN_PIN` at the top of the sketch.

The pattern is set at the top of the sketch. **These values are placeholders. Nothing is decided yet.**

| Setting | Value | |
| --- | --- | --- |
| `PULSE_COUNT` | 2 | a double pulse, as the professor suggested |
| `PULSE_MS` | 25 ms | each pulse |
| `GAP_MS` | 30 ms | between the pulses |
| `FREQUENCY_HZ` | 200 Hz | the datasheet's response peaks near 110 Hz; above 500 Hz it becomes audible |
| `RAMP_MS` | 5 ms | smooth start and end of each pulse, so it does not click |
| `AMPLITUDE` | 0.2 | fraction of the amplifier's full scale |

The whole pattern lasts 80 ms. This is under the 100 ms that `TopEdgeCue` keeps between two messages, so one
pattern ends before the next can start.

**Before putting the Haptuator on a finger:** play a pulse and measure the voltage across the Haptuator with the
oscilloscope. Raise `AMPLITUDE` only while it stays under 3.0 V.

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

The monitor should print `EA-CUE advertising; Haptuator 2 x 25 ms at 200 Hz`. If it shows nothing, the line was
probably printed before the monitor opened: go on with the phone check. Close the monitor with Ctrl+C.

- **No port appears, or the upload cannot connect:** hold the **BOOT** button while plugging the USB cable in,
  release it, and upload again. Then press **RST** (or plug the cable in again) to start the program. The board
  has no separate USB-serial chip, so no driver is needed.
- **Pins:** GPIO 4 (marked SCK on the board) is the oscilloscope mark, and GPIO 5, 6 and 7 go to the amplifier.
  Avoid GPIO 2, 8 and 9: they decide how the chip starts, GPIO 8 drives the LED and GPIO 9 is the BOOT button.

The Arduino IDE works as well: install **esp32 by Espressif Systems** from its Boards Manager, open
`esp32_cue.ino`, choose the board **Nologo ESP32C3 Super Mini** and its port, upload, and open the Serial Monitor
at 115200 baud.

## Bluetooth LE layout

| | UUID | |
| --- | --- | --- |
| Service | `cfdd2bb4-1857-421a-9450-e9c8a92ea3f9` | advertised, device name `EA-CUE` |
| CUE | `03c34c2a-b29f-4040-b812-5778ed2507cd` | PC → ESP32, write with or without response |

## Check with a phone, before any PC code

With nRF Connect (Android or iOS):

1. Scan, find `EA-CUE`, connect.
2. On the CUE characteristic (`03c34c2a-…`), write the text (UTF-8) `EXIT_LEFT`.
3. Expected: the Haptuator gives a double pulse, the LED lights briefly, and the serial monitor prints
   `EXIT_LEFT at … us`.
4. Write `ENTER_RIGHT`. Expected: the same, with `ENTER_RIGHT at … us`.
5. Write `HELLO`. Expected: `ignored: "HELLO"`, and the Haptuator stays still.

## Not done yet

- Choosing the pulse pattern and the amplitude.
- Trying the PC side with the ESP32. It is written, but so far tested only against a fake Bluetooth link.
- A request for a short connection interval. The PC (Windows) picks the interval, and the latency measurement
  will show whether asking for a shorter one is worth it.
