// ESP32 side of the top-edge cue: receives cue messages from the experiment PC and plays a pulse on the Haptuator
// through a MAX98357A amplifier. The messages come over the USB cable or over Bluetooth LE; both are always open,
// and the PC's settings decide which one it uses.
//
// The PC sends one text line each time the finger crosses a bar's top edge: ENTER_LEFT, EXIT_LEFT, ENTER_RIGHT or
// EXIT_RIGHT. On each one the ESP32 sends the pulse pattern to the amplifier over I2S, raises MARK_PIN (for the
// oscilloscope), lights the board LED, and prints the message on the serial monitor. The pattern values are
// placeholders until the lab chooses them. The wiring is in README.md.
//
// Written for the ESP32-C3 Super Mini and the ESP32 Arduino core 3.x. On another board, check the pins and
// LED_ACTIVE_LOW.

#include <BLEDevice.h>
#include <BLEServer.h>
#include <ESP_I2S.h>
#include <math.h>

static const char *DEVICE_NAME = "EA-CUE";
static const char *SERVICE_UUID = "cfdd2bb4-1857-421a-9450-e9c8a92ea3f9";
static const char *CUE_UUID = "03c34c2a-b29f-4040-b812-5778ed2507cd";  // PC -> ESP32: one message per write

static const char *MESSAGES[] = {"ENTER_LEFT", "EXIT_LEFT", "ENTER_RIGHT", "EXIT_RIGHT"};
static const size_t MESSAGE_COUNT = sizeof(MESSAGES) / sizeof(MESSAGES[0]);

static const int MARK_PIN = 4;  // high for MARK_US from each valid message's arrival, for the oscilloscope
static const uint32_t MARK_US = 2000;
static const uint32_t LED_MS = 100;
static const bool LED_ACTIVE_LOW = true;  // the Super Mini's blue LED (GPIO 8) lights when its pin is low

// Haptuator Mark II (TL002-09-A): 5.5 ohm, at most 3.0 V and 0.5 A, rated 90-1000 Hz. It is driven like a
// loudspeaker, by the amplifier. A GPIO pin cannot drive it.
static const int I2S_BCLK_PIN = 5;  // MAX98357A BCLK
static const int I2S_LRC_PIN = 6;   // MAX98357A LRC
static const int I2S_DIN_PIN = 7;   // MAX98357A DIN
static constexpr uint32_t SAMPLE_RATE_HZ = 48000;

// The pattern: PULSE_COUNT sine bursts of PULSE_MS each, GAP_MS apart, each starting and ending with a smooth
// RAMP_MS ramp. Placeholders: a double pulse as the professor suggested, at 200 Hz. Nothing is decided yet.
static constexpr float FREQUENCY_HZ = 200.0f;
static constexpr int PULSE_COUNT = 2;
static constexpr uint32_t PULSE_MS = 25;
static constexpr uint32_t GAP_MS = 30;
static constexpr uint32_t RAMP_MS = 5;
// Fraction of the amplifier's full scale. Start low: measure the voltage across the Haptuator with the
// oscilloscope and keep it under 3.0 V before raising this.
static constexpr float AMPLITUDE = 0.2f;

static_assert(PULSE_COUNT >= 1, "at least one pulse");
static_assert(2 * RAMP_MS <= PULSE_MS, "the two ramps must fit in a pulse");
static constexpr size_t PULSE_FRAMES = SAMPLE_RATE_HZ * PULSE_MS / 1000;
static constexpr size_t GAP_FRAMES = SAMPLE_RATE_HZ * GAP_MS / 1000;
static constexpr size_t PATTERN_FRAMES = PULSE_COUNT * PULSE_FRAMES + (PULSE_COUNT - 1) * GAP_FRAMES;
static int16_t pattern[2 * PATTERN_FRAMES];  // stereo frames, the same sample on both channels; gaps stay zero

static I2SClass i2s;
static QueueHandle_t pulses = nullptr;  // one item per pattern to play
static char usbLine[32];                // the USB line being received, up to its newline
static size_t usbLength = 0;
static bool usbOverflow = false;
static volatile bool markHigh = false;
static volatile uint32_t markSinceUs = 0;
static volatile bool ledOn = false;
static volatile uint32_t ledSinceMs = 0;

static void setLed(bool on) {
#ifdef LED_BUILTIN
  digitalWrite(LED_BUILTIN, on != LED_ACTIVE_LOW ? HIGH : LOW);
#endif
}

// Index into MESSAGES, or -1 if the text is none of them.
static int messageIndex(const char *text, size_t length) {
  for (size_t i = 0; i < MESSAGE_COUNT; i++) {
    if (strlen(MESSAGES[i]) == length && memcmp(MESSAGES[i], text, length) == 0) {
      return (int)i;
    }
  }
  return -1;
}

static void buildPattern() {
  const size_t rampFrames = SAMPLE_RATE_HZ * RAMP_MS / 1000;
  size_t frame = 0;
  for (int pulse = 0; pulse < PULSE_COUNT; pulse++) {
    for (size_t n = 0; n < PULSE_FRAMES; n++, frame++) {
      float envelope = 1.0f;
      if (n < rampFrames) {
        envelope = 0.5f * (1.0f - cosf((float)PI * n / rampFrames));
      } else if (n >= PULSE_FRAMES - rampFrames) {
        envelope = 0.5f * (1.0f - cosf((float)PI * (PULSE_FRAMES - n) / rampFrames));
      }
      const float value = AMPLITUDE * envelope * sinf(2.0f * (float)PI * FREQUENCY_HZ * n / SAMPLE_RATE_HZ);
      const int16_t sample = (int16_t)lroundf(value * 32767.0f);
      pattern[2 * frame] = sample;
      pattern[2 * frame + 1] = sample;
    }
    frame += GAP_FRAMES;  // left at zero
  }
}

// Plays one pattern per queued message. Its own task, so that the BLE callback never waits for the amplifier.
static void playerTask(void *) {
  uint8_t index;
  for (;;) {
    if (xQueueReceive(pulses, &index, portMAX_DELAY) == pdTRUE) {
      i2s.write(pattern, sizeof(pattern));
    }
  }
}

// One message, from either link. Plays the pulse for a known message; anything else is reported and ignored.
static void handleMessage(const char *text, size_t length, uint32_t receivedUs, const char *source) {
  // The newline may be left out, so that a phone app can send the bare word.
  while (length > 0 && (text[length - 1] == '\n' || text[length - 1] == '\r')) {
    length--;
  }
  const int index = messageIndex(text, length);
  if (index < 0) {
    Serial.printf("ignored (%s): \"%.*s\"\n", source, (int)(length < 40 ? length : 40), text);
    return;
  }
  // The mark first: its rising edge is the arrival time the oscilloscope compares with the EA signal.
  digitalWrite(MARK_PIN, HIGH);
  markSinceUs = receivedUs;
  markHigh = true;
  const uint8_t item = (uint8_t)index;
  const bool dropped = xQueueSend(pulses, &item, 0) != pdTRUE;
  setLed(true);
  ledSinceMs = millis();
  ledOn = true;
  Serial.printf("%s at %lu us via %s%s\n", MESSAGES[index], (unsigned long)receivedUs, source,
                dropped ? " (pulse dropped: two already waiting)" : "");
}

// The same messages over the USB cable: one line each, ending in a newline. A line too long to be one is ignored.
static void readUsb() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    if (c == '\n') {
      if (usbOverflow) {
        Serial.println("ignored (usb): line too long");
      } else {
        handleMessage(usbLine, usbLength, micros(), "usb");
      }
      usbLength = 0;
      usbOverflow = false;
    } else if (usbLength < sizeof(usbLine)) {
      usbLine[usbLength++] = (char)c;
    } else {
      usbOverflow = true;
    }
  }
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    Serial.println("connected");
  }

  void onDisconnect(BLEServer *server) override {
    Serial.println("disconnected; advertising again");
    BLEDevice::startAdvertising();
  }
};

class CueCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    // One message per write.
    const uint32_t receivedUs = micros();
    handleMessage((const char *)characteristic->getData(), characteristic->getLength(), receivedUs, "bluetooth");
  }
};

void setup() {
  Serial.begin(115200);
#if ARDUINO_USB_MODE && ARDUINO_USB_CDC_ON_BOOT
  // Serial is the USB cable. A PC that sends cues over it need not read these lines: never wait for it to.
  Serial.setTxTimeoutMs(0);
#endif
  pinMode(MARK_PIN, OUTPUT);
  digitalWrite(MARK_PIN, LOW);
#ifdef LED_BUILTIN
  pinMode(LED_BUILTIN, OUTPUT);
  setLed(false);
#endif

  buildPattern();
  i2s.setPins(I2S_BCLK_PIN, I2S_LRC_PIN, I2S_DIN_PIN);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE_HZ, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S did not start: the Haptuator will stay still");
  }
  pulses = xQueueCreate(2, sizeof(uint8_t));
  xTaskCreate(playerTask, "haptuator", 4096, nullptr, 5, nullptr);

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  BLEService *service = server->createService(SERVICE_UUID);
  BLECharacteristic *cue = service->createCharacteristic(
      CUE_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  cue->setCallbacks(new CueCallbacks());
  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.printf("%s ready on USB and Bluetooth; Haptuator %d x %lu ms at %.0f Hz\n", DEVICE_NAME, PULSE_COUNT,
                (unsigned long)PULSE_MS, FREQUENCY_HZ);
}

void loop() {
  readUsb();
  if (markHigh && micros() - markSinceUs >= MARK_US) {
    digitalWrite(MARK_PIN, LOW);
    markHigh = false;
  }
  if (ledOn && millis() - ledSinceMs >= LED_MS) {
    setLed(false);
    ledOn = false;
  }
  delay(1);
}
