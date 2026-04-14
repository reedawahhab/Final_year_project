/*
  ADS131A04 -> ESP32 (Arduino Nano ESP32) -> BLE Notify streaming (BATCHED: 2 samples/notification)

  BLE:
    SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

  Packet (40 bytes, little-endian) = 2 samples back-to-back
  Each sample (20 bytes):
    uint32 t_us
    int32  ch1
    int32  ch2
    int32  ch3
    int32  ch4
*/

#include <Arduino.h>
#include <SPI.h>

// -------- BLE (ESP32 BLE Arduino) --------
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// -------------------- Pins --------------------
static const int PIN_CS    = 10;
static const int PIN_DRDY  = 2;
static const int PIN_RESET = 4;

// -------------------- SPI --------------------
static SPISettings ADS_SPI(2000000, MSBFIRST, SPI_MODE1); // start safe at 2 MHz

// -------------------- ADS131A0x Commands --------------------
static const uint16_t CMD_NULL    = 0x0000;
static const uint16_t CMD_RESET   = 0x0011;
static const uint16_t CMD_STANDBY = 0x0022;
static const uint16_t CMD_WAKEUP  = 0x0033;
static const uint16_t CMD_LOCK    = 0x0555;
static const uint16_t CMD_UNLOCK  = 0x0655;

static inline uint8_t CMD_RREG(uint8_t addr) { return (uint8_t)(0b00100000 | (addr & 0x1F)); }
static inline uint8_t CMD_WREG(uint8_t addr) { return (uint8_t)(0b01000000 | (addr & 0x1F)); }

// -------------------- Registers --------------------
static const uint8_t REG_A_SYS_CFG = 0x0B;
static const uint8_t REG_D_SYS_CFG = 0x0C;
static const uint8_t REG_CLK1      = 0x0D;
static const uint8_t REG_CLK2      = 0x0E;
static const uint8_t REG_ADC_ENA   = 0x0F;
static const uint8_t REG_ADC1      = 0x11;
static const uint8_t REG_ADC2      = 0x12;
static const uint8_t REG_ADC3      = 0x13;
static const uint8_t REG_ADC4      = 0x14;

// -------------------- DRDY flag --------------------
volatile bool g_drdy = false;
void IRAM_ATTR onDrdyFalling() { g_drdy = true; }

// -------------------- BLE globals --------------------
BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;
volatile bool g_connected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override { g_connected = true; }
  void onDisconnect(BLEServer* server) override {
    g_connected = false;
    BLEDevice::startAdvertising();
  }
};

// -------------------- Batching (2 samples per notify) --------------------
struct Sample {
  uint32_t t_us;
  int32_t c1, c2, c3, c4;
};

static Sample s_buf[2];
static int s_count = 0;

// -------------------- Helpers --------------------
static inline void csLow()  { digitalWrite(PIN_CS, LOW); }
static inline void csHigh() { digitalWrite(PIN_CS, HIGH); }

static uint16_t transfer24_cmdReadStatus(uint16_t cmd16)
{
  uint8_t tx[3] = { (uint8_t)(cmd16 >> 8), (uint8_t)(cmd16 & 0xFF), 0x00 };
  uint8_t rx[3] = { 0, 0, 0 };

  SPI.beginTransaction(ADS_SPI);
  csLow();
  SPI.transferBytes(tx, rx, 3);
  csHigh();
  SPI.endTransaction();

  return (uint16_t(rx[0]) << 8) | uint16_t(rx[1]);
}

static uint8_t readReg(uint8_t addr)
{
  uint16_t cmd16 = (uint16_t(CMD_RREG(addr)) << 8) | 0x00;
  (void)transfer24_cmdReadStatus(cmd16);              // issue
  uint16_t resp = transfer24_cmdReadStatus(CMD_NULL); // readback
  return (uint8_t)(resp & 0xFF);
}

static void writeReg(uint8_t addr, uint8_t data)
{
  uint16_t cmd16 = (uint16_t(CMD_WREG(addr)) << 8) | data;
  (void)transfer24_cmdReadStatus(cmd16);
}

static int32_t signExtend24(uint32_t x)
{
  if (x & 0x800000) return (int32_t)(x | 0xFF000000);
  return (int32_t)x;
}

static bool readDataFrame_4ch(int32_t &c1, int32_t &c2, int32_t &c3, int32_t &c4, uint16_t &status16)
{
  // 4ch, CRC off => 5 words * 3 bytes = 15 bytes
  const int BYTES = 5 * 3;
  uint8_t tx[BYTES];
  uint8_t rx[BYTES];
  memset(tx, 0, sizeof(tx));
  memset(rx, 0, sizeof(rx));

  SPI.beginTransaction(ADS_SPI);
  csLow();
  SPI.transferBytes(tx, rx, BYTES);
  csHigh();
  SPI.endTransaction();

  status16 = (uint16_t(rx[0]) << 8) | uint16_t(rx[1]);

  auto word24 = [&](int w) -> uint32_t {
    int i = w * 3;
    return (uint32_t(rx[i]) << 16) | (uint32_t(rx[i + 1]) << 8) | uint32_t(rx[i + 2]);
  };

  c1 = signExtend24(word24(1));
  c2 = signExtend24(word24(2));
  c3 = signExtend24(word24(3));
  c4 = signExtend24(word24(4));
  return true;
}

static void hwResetPulse()
{
  digitalWrite(PIN_RESET, LOW);
  delay(2);
  digitalWrite(PIN_RESET, HIGH);
  delay(5);
}

static bool waitReadyWord()
{
  // READY word for ADS131A04 is 0xFF04
  const uint16_t expected = 0xFF04;
  uint32_t t0 = millis();
  while (millis() - t0 < 2000) {
    uint16_t s = transfer24_cmdReadStatus(CMD_NULL);
    if (s == expected) return true;
    delay(2);
  }
  return false;
}

static void adsInit()
{
  hwResetPulse();
  (void)transfer24_cmdReadStatus(CMD_RESET);
  delay(5);

  if (!waitReadyWord()) {
    Serial.println("ERR: no READY 0xFF04. Check SPI/clock/mode pins.");
    return;
  }

  (void)transfer24_cmdReadStatus(CMD_UNLOCK);
  delay(2);

  (void)transfer24_cmdReadStatus(CMD_STANDBY);
  delay(2);

  // Bring-up config:
  // - Internal ref enable (INT_REFEN=1)
  // - CRC off
  // - 16.384MHz crystal, choose ~500 SPS: OSR=512 with divider chain -> CLK2=0x85
  writeReg(REG_A_SYS_CFG, 0x68);  // internal ref enable
  writeReg(REG_D_SYS_CFG, 0x3C);  // defaults, CRC off
  writeReg(REG_CLK1,      0x08);  // default divider chain (good start)
  writeReg(REG_CLK2,      0x85);  // ICLK_DIV=4, OSR=512 -> ~500 SPS
  writeReg(REG_ADC1,      0x00);
  writeReg(REG_ADC2,      0x00);
  writeReg(REG_ADC3,      0x00);
  writeReg(REG_ADC4,      0x00);
  writeReg(REG_ADC_ENA,   0x0F);  // enable all 4 channels

  (void)transfer24_cmdReadStatus(CMD_WAKEUP);
  delay(2);

  Serial.print("CLK2=0x"); Serial.print(readReg(REG_CLK2), HEX);
  Serial.print(" ADC_ENA=0x"); Serial.println(readReg(REG_ADC_ENA), HEX);
}

// -------------------- BLE setup --------------------
static void bleInit()
{
  BLEDevice::init("EMG_Sensor");
  BLEDevice::setMTU(185); // ok even if macOS ignores

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  pCharacteristic->setValue((uint8_t*)"", 0);

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("BLE advertising started");
}

// -------------------- Arduino --------------------
void setup()
{
  Serial.begin(115200);

  // Helps Nano ESP32 / macOS serial monitor show output reliably
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 1500)) { delay(10); }
  Serial.println("Serial OK");

  pinMode(PIN_CS, OUTPUT);
  csHigh();

  pinMode(PIN_RESET, OUTPUT);
  digitalWrite(PIN_RESET, HIGH);

  pinMode(PIN_DRDY, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdyFalling, FALLING);

  SPI.begin(); // D11/D12/D13 on Nano ESP32

  bleInit();
  adsInit();
}

void loop()
{
  if (!g_drdy) return;
  g_drdy = false;

  int32_t c1, c2, c3, c4;
  uint16_t status16;
  readDataFrame_4ch(c1, c2, c3, c4, status16);

  Sample s;
  s.t_us = micros();
  s.c1 = c1; s.c2 = c2; s.c3 = c3; s.c4 = c4;

  s_buf[s_count++] = s;

  // Once we have 2 samples, send them together (40 bytes)
  if (s_count >= 2) {
    s_count = 0;

    uint8_t pkt[40];

    // sample 0
    memcpy(pkt + 0,  &s_buf[0].t_us, 4);
    memcpy(pkt + 4,  &s_buf[0].c1,   4);
    memcpy(pkt + 8,  &s_buf[0].c2,   4);
    memcpy(pkt + 12, &s_buf[0].c3,   4);
    memcpy(pkt + 16, &s_buf[0].c4,   4);

    // sample 1
    memcpy(pkt + 20, &s_buf[1].t_us, 4);
    memcpy(pkt + 24, &s_buf[1].c1,   4);
    memcpy(pkt + 28, &s_buf[1].c2,   4);
    memcpy(pkt + 32, &s_buf[1].c3,   4);
    memcpy(pkt + 36, &s_buf[1].c4,   4);

    if (g_connected && pCharacteristic) {
      pCharacteristic->setValue(pkt, sizeof(pkt));
      pCharacteristic->notify();
    }
  }
}