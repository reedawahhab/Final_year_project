// ------------------------------------------------------------
// DFRobot SEN0203 Heart Rate Sensor (ANALOG MODE)
// Streams raw PPG samples to Python over Serial
// Output format: t_ms,raw
// ------------------------------------------------------------

const uint8_t PPG_PIN = A1;
const uint32_t BAUDRATE = 115200;
const float FS = 100.0f;                     // 100 Hz sample rate
const uint32_t SAMPLE_PERIOD_US = (uint32_t)(1000000.0f / FS);

void setup() {
  Serial.begin(BAUDRATE);

  // For boards that support it (ESP32, etc.)
  #if defined(ARDUINO_ARCH_ESP32)
    analogReadResolution(12);
  #endif

  delay(1000);
}

void loop() {
  static uint32_t nextSampleUs = 0;

  uint32_t nowUs = micros();
  if ((int32_t)(nowUs - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_PERIOD_US;

    uint32_t t_ms = millis();
    uint16_t raw = analogRead(PPG_PIN);

    Serial.print(t_ms);
    Serial.print(",");
    Serial.println(raw);
  }
}