// ESP32 side of the top-edge cue: receives cue messages from the experiment PC over Bluetooth LE.
//
// For now it only shows that a message arrived. On every valid message it raises CUE_PIN for PIN_PULSE_US (for
// the oscilloscope), lights the board LED for LED_MS, sends the message back with its own receive time, and
// prints a line on the serial monitor. There is no motor yet. The message format is in README.md next to this
// file.
//
// Written for the ESP32-C3 Super Mini. Uses only the BLE library that comes with the ESP32 Arduino core, so it
// builds on any ESP32 that has BLE; on another board, check CUE_PIN and LED_ACTIVE_LOW.

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>

static const char *DEVICE_NAME = "EA-CUE";
static const char *SERVICE_UUID = "cfdd2bb4-1857-421a-9450-e9c8a92ea3f9";
static const char *CUE_UUID = "03c34c2a-b29f-4040-b812-5778ed2507cd";  // PC -> ESP32: one message per write
static const char *ACK_UUID = "34c5a9d1-fb6d-445f-b9d9-e6fe0eac7136";  // ESP32 -> PC: the message back

static const int CUE_PIN = 4;  // goes high on every valid message; check that the board leaves GPIO 4 free
static const uint32_t PIN_PULSE_US = 2000;
static const uint32_t LED_MS = 100;
static const bool LED_ACTIVE_LOW = true;  // the Super Mini's blue LED (GPIO 8) lights when its pin is low

// seq (uint16, little-endian), kind, bar
static const size_t MESSAGE_BYTES = 4;
static const uint8_t KIND_TEST = 0;
static const uint8_t KIND_ENTERING = 1;
static const uint8_t KIND_LEAVING = 2;

static BLECharacteristic *ack = nullptr;
static volatile bool pinHigh = false;
static volatile uint32_t pinSinceUs = 0;
static volatile bool ledOn = false;
static volatile uint32_t ledSinceMs = 0;

static void setLed(bool on) {
#ifdef LED_BUILTIN
  digitalWrite(LED_BUILTIN, on != LED_ACTIVE_LOW ? HIGH : LOW);
#endif
}

static const char *kindName(uint8_t kind) {
  switch (kind) {
    case KIND_TEST: return "test";
    case KIND_ENTERING: return "entering";
    case KIND_LEAVING: return "leaving";
    default: return "unknown-kind";
  }
}

static const char *barName(uint8_t bar) {
  switch (bar) {
    case 0: return "left";
    case 1: return "right";
    default: return "unknown-bar";
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
    const uint32_t receivedUs = micros();
    const size_t length = characteristic->getLength();
    if (length != MESSAGE_BYTES) {
      Serial.printf("ignored: %u bytes, expected %u\n", (unsigned)length, (unsigned)MESSAGE_BYTES);
      return;
    }
    // The pin first: its rising edge is the moment the oscilloscope compares with the generator output.
    digitalWrite(CUE_PIN, HIGH);
    pinSinceUs = receivedUs;
    pinHigh = true;
    setLed(true);
    ledSinceMs = millis();
    ledOn = true;

    uint8_t reply[MESSAGE_BYTES + 4];
    memcpy(reply, characteristic->getData(), MESSAGE_BYTES);
    for (int i = 0; i < 4; i++) {
      reply[MESSAGE_BYTES + i] = (receivedUs >> (8 * i)) & 0xFF;
    }
    ack->setValue(reply, sizeof(reply));
    ack->notify();

    const unsigned seq = reply[0] | (reply[1] << 8);
    Serial.printf("cue %u %s %s at %lu us\n", seq, kindName(reply[2]), barName(reply[3]), (unsigned long)receivedUs);
  }
};

void setup() {
  Serial.begin(115200);
  pinMode(CUE_PIN, OUTPUT);
  digitalWrite(CUE_PIN, LOW);
#ifdef LED_BUILTIN
  pinMode(LED_BUILTIN, OUTPUT);
  setLed(false);
#endif

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  BLEService *service = server->createService(SERVICE_UUID);

  BLECharacteristic *cue = service->createCharacteristic(
      CUE_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  cue->setCallbacks(new CueCallbacks());
  ack = service->createCharacteristic(ACK_UUID, BLECharacteristic::PROPERTY_NOTIFY);
#if defined(CONFIG_BLUEDROID_ENABLED)
  // Lets the PC switch notifications on. NimBLE adds this descriptor by itself and rejects a manual one.
  ack->addDescriptor(new BLE2902());
#endif
  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.printf("%s advertising; cue pin GPIO %d\n", DEVICE_NAME, CUE_PIN);
}

void loop() {
  if (pinHigh && micros() - pinSinceUs >= PIN_PULSE_US) {
    digitalWrite(CUE_PIN, LOW);
    pinHigh = false;
  }
  if (ledOn && millis() - ledSinceMs >= LED_MS) {
    setLed(false);
    ledOn = false;
  }
  delay(1);
}
