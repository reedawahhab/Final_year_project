# Final Year Project

This repository contains code for physiological signal acquisition, processing, feature extraction, and embedded firmware for microcontroller-based data capture.

## Structure

### `acquistion_signal_proccessing/`
Python scripts for acquiring and visualizing raw physiological signals from ADC or BLE sources.

- `eeg_ecg.py`
  - BLE-based EEG/ECG acquisition and live display
  - Uses an `EMG_Sensor` BLE device and a batched 2-sample notify packet format
  - Includes filtering, scaling, and post-processing settings for EEG/ECG

- `gsr_howland.py`
  - Serial live plotter for a GSR Howland current source setup
  - Performs RMS, envelope smoothing, and baseline tracking for GSR recordings

- `gsr_transimpedance.py`
  - Serial live conductance plotter with IQ demodulation for a GSR transimpedance front-end
  - Reads voltage/current ADC channels over serial, computes lock-in amplifier conductance, and plots live

- `ppg.py`
  - Serial live debugger for PPG data
  - Reads a three-value CSV-style output from a microcontroller and plots raw and difference waveforms

### `features_extraction/`
Python scripts for extracting features from recorded physiological data.

- `eeg_ecg_features.py`
  - EEG/ECG signal processing and feature extraction pipeline
  - Uses filtering, windowed feature computation, Welch spectral analysis, and peak detection
  - Configured for 4-channel acquisition (3 EEG + 1 ECG)

- `gsr_features.py`
  - Feature extraction for GSR recordings from a Howland circuit
  - Computes signal statistics, positive AUC, and peak counts with configurable detection settings

- `ppg_features.py`
  - PPG CSV loader and pulse feature extraction
  - Includes filtering, peak detection, IBI constraints, and physiology-specific processing settings

### `microcontrollers_C/`
Embedded firmware sources for Arduino/ESP32 and STM32 microcontrollers.

- `arduino_eeg_ecg.c`
  - ADS131A04 to ESP32 BLE firmware
  - Streams 4-channel EEG/ECG data over BLE using a custom service/characteristic

- `main_ppg.c`
  - STM32 PPG firmware for 20 Hz UART output
  - Uses timed ADC averaging and UART packet streaming for a Python viewer

- `ppg_2.c`
  - Arduino sketch for DFRobot SEN0203 heart rate sensor in analog mode
  - Outputs raw PPG samples over serial as `t_ms,raw`

- `stm32_gsr.c`
  - STM32 GSR firmware with sine excitation table and UART output
  - Appears to produce a fixed reference sine waveform and read ADC data

- `stm32_gsr_2.c`
  - Second STM32 GSR firmware version with dual ADC channel capture
  - Sends `adc_ch0,adc_ch1` values over UART when new samples are ready

## Dependencies

The Python scripts rely on packages commonly used for signal processing and hardware interfacing, including:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `bleak`
- `pyserial`

## Notes

- There was no existing `README` in the repository before this file was added.
- The repository is organized around data acquisition and feature extraction for biosignals, with both PC-side Python tools and embedded C firmware.
- Some scripts use hard-coded paths and serial/COM port settings, so update them before running in your environment.
