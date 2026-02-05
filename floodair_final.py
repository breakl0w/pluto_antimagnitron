#!/usr/bin/env python3

import argparse
import time
import random
import numpy as np
import yaml
from gnuradio import gr, blocks, analog, digital
import osmosdr
import sys

try:
    from prettyprinter import cpprint
except:
    from pprint import pprint

# PyQtGraph для визуализации
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import threading


class CallbackBlock:
    def __init__(self, callback, sample_rate=50e6):
        self.callback = callback
        self.sample_rate = sample_rate
        self.buffer = []
        self.buffer_size = int(sample_rate * 0.01)  # 10ms

    def work(self, input_items, output_items):
        in0 = input_items[0]
        self.buffer.extend(in0)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        self.callback(self.buffer.copy())
        return len(in0)


class Visualizer:
    def __init__(self, sample_rate=50e6, fft_size=1024, update_interval=100):
        self.app = QApplication([])
        self.win = QMainWindow()
        self.win.setWindowTitle("RF Signal Visualizer")
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.win.setCentralWidget(self.central_widget)

        # Спектрограмма
        self.plot_spectrum = pg.PlotItem()
        self.plot_spectrum.setTitle("Spectrum")
        self.plot_spectrum.setLabel('bottom', 'Frequency (Hz)')
        self.plot_spectrum.setLabel('left', 'Magnitude (dB)')
        self.plot_spectrum.setXRange(0, sample_rate // 2)
        self.plot_spectrum.setYRange(-100, 0)
        self.spectrum_curve = self.plot_spectrum.plot(pen='b', name="Spectrum")

        # График амплитуды
        self.plot_time = pg.PlotItem()
        self.plot_time.setTitle("Time Domain")
        self.plot_time.setLabel('bottom', 'Time (s)')
        self.plot_time.setLabel('left', 'Amplitude')
        self.plot_time.setXRange(0, 0.01)  # 10ms
        self.plot_time.setYRange(-1, 1)
        self.time_curve = self.plot_time.plot(pen='r', name="Signal")

        # Глазная диаграмма
        self.plot_eye = pg.PlotItem()
        self.plot_eye.setTitle("Eye Diagram")
        self.plot_eye.setLabel('bottom', 'Time (symbol)')
        self.plot_eye.setLabel('left', 'Amplitude')
        self.eye_curve = self.plot_eye.plot(pen='g', name="Eye")

        # Добавляем графики
        self.layout.addWidget(pg.GraphicsLayoutWidget())
        self.layout.addWidget(pg.GraphicsLayoutWidget())
        self.layout.addWidget(pg.GraphicsLayoutWidget())

        # Начальные данные
        self.fft_data = np.zeros(fft_size)
        self.time_data = np.zeros(int(sample_rate * 0.01))  # 10ms
        self.eye_data = []

        # Таймер
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

        self.running = False
        self.sample_rate = sample_rate
        self.fft_size = fft_size

    def update_data(self, data):
        if not self.running or len(data) == 0:
            return

        # Обновляем временной график
        self.time_data = data[-len(self.time_data):]

        # Спектр
        fft_data = np.abs(np.fft.rfft(self.time_data))
        fft_data = 20 * np.log10(fft_data + 1e-10)  # dB

        # Глазная диаграмма
        if len(self.eye_data) > 1000:
            self.eye_data.pop(0)
        for i in range(len(data) - 1):
            self.eye_data.append((i, np.real(data[i])))
            if len(self.eye_data) > 1000:
                break

        # Обновляем графики
        self.time_curve.setData(self.time_data)
        self.spectrum_curve.setData(
            np.fft.rfftfreq(len(self.time_data), d=1/self.sample_rate)[:len(fft_data)],
            fft_data
        )
        self.eye_curve.setData(*zip(*self.eye_data))

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        self.app.exec_()


class VisualizerThread:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.visualizer.run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.thread.join()


class FloodAir:
    def __init__(self, options):
        super(FloodAir, self).__init__()
        self.options = options

        self.signal_type = options.get("signal_type")
        self.signal_power = options.get("signal_power")
        self.visualize = options.get("visualize", False)
        self.record_data = options.get("record_data", False)
        self.output_file = options.get("output_file", "rf_output.dat")

        self.hopper_entropy = False
        self.hopper_delay_static = options.get("hopper_delay_static")
        self.hopper_delay_min = options.get("hopper_delay_min")
        self.hopper_delay_max = options.get("hopper_delay_max")

        self.RF_gain = 60
        self.IF_gain = 60
        self.sink = None
        self.tb = None
        self.source = None
        self.sample_rate = 50e6
        self.bandwidth = 50e6
        self.setup_once = False
        self.hopper_mode = options.get("hopper_mode", 3.0)
        self.visualizer = None
        self.visualizer_thread = None

        if self.visualize:
            self.visualizer = Visualizer(sample_rate=self.sample_rate, update_interval=100)

    def set_gains(self):
        if -40 <= self.signal_power <= 5:
            self.RF_gain = 0
            if self.signal_power < -5:
                self.IF_gain = self.signal_power + 40
            elif -5 <= self.signal_power <= 2:
                self.IF_gain = self.signal_power + 41
            elif 2 < self.signal_power <= 5:
                self.IF_gain = self.signal_power + 42
        elif self.signal_power > 5:
            self.RF_gain = 14
            self.IF_gain = self.signal_power + 34
        return self.RF_gain, self.IF_gain

    def set_freq(self, freq):
        try:
            self.sink.set_center_freq(freq, 0)
        except Exception as e:
            print(f"Error setting frequency: {e}")

    def get_freq(self):
        try:
            f = self.sink.get_center_freq()
        except Exception as e:
            print(f"Error getting frequency: {e}")
            f = self.options.get("frequency_start") * 10e5
        return f

    def print_freq(self):
        print(f"\r\t\t\t\t\t\tLet it eat: {self.get_freq() / 10e5}MHz\t", end="")

    def _hop_wait(self):
        if self.hopper_mode > 3.2:
            return
        _wait = self.hopper_delay_static
        if self.hopper_entropy:
            _wait = random.uniform(self.hopper_delay_min, self.hopper_delay_max)

        print("\n\r", _wait, flush=True, end="")
        print("s...", flush=True, end="\r")
        time.sleep(_wait)

    def _waveform(self):
        if self.source:
            return

        throttle = blocks.throttle(gr.sizeof_gr_complex * 1, self.sample_rate, True)

        match self.signal_type:
            case 1:
                print("signal_type:\tsine")
                self.source = analog.sig_source_c(
                    self.sample_rate, analog.GR_SAW_WAVE, 1000, 1, 0, 0
                )
                self.tb.connect(self.source, throttle)
                self.tb.connect(throttle, self.sink)

            case 2:
                print("signal_type:\tQPSK")
                constellation = digital.constellation_rect(
                    [-1 - 1j, -1 + 1j, 1 + 1j, 1 - 1j], [0, 1, 3, 2], 4, 2, 2, 1, 1
                ).base()
                mod = digital.generic_mod(
                    constellation=constellation,
                    differential=True,
                    samples_per_symbol=4,
                    pre_diff_code=True,
                    excess_bw=0.035,
                    verbose=True,
                )
                data = np.random.randint(0, 255, 1000).tolist()
                self.source = blocks.vector_source_b(data, True)
                self.tb.connect(self.source, mod)
                self.tb.connect(mod, throttle, self.sink)

            case 3:
                print("signal_type:\tnoise")
                self.source = analog.noise_source_c(
                    analog.analog.GR_UNIFORM, 1.0, random.randint(11111, 55555)
                )
                self.tb.connect(self.source, throttle)
                self.tb.connect(throttle, self.sink)

            case 4:
                print("signal_type:\tBPSK")
                constellation = digital.constellation_rect([-1 + 0j, 1 + 0j], [0, 1], 2, 1, 1, 1, 1).base()
                mod = digital.generic_mod(
                    constellation=constellation,
                    differential=False,
                    samples_per_symbol=2,
                    pre_diff_code=False,
                    excess_bw=0.35,
                    verbose=True,
                )
                data = np.random.randint(0, 2, 1000).tolist()
                self.source = blocks.vector_source_b(data, True)
                self.tb.connect(self.source, mod)
                self.tb.connect(mod, throttle, self.sink)

            case 5:
                print("signal_type:\t8PSK")
                constellation = digital.constellation_rect([
                    np.exp(1j * np.pi * k / 4) for k in range(8)
                ], list(range(8)), 8, 3, 3, 1, 1).base()
                mod = digital.generic_mod(
                    constellation=constellation,
                    differential=False,
                    samples_per_symbol=4,
                    pre_diff_code=False,
                    excess_bw=0.35,
                    verbose=True,
                )
                data = np.random.randint(0, 8, 1000).tolist()
                self.source = blocks.vector_source_b(data, True)
                self.tb.connect(self.source, mod)
                self.tb.connect(mod, throttle, self.sink)

            case 6:
                print("signal_type:\tOOK")
                data = np.random.randint(0, 2, 1000).tolist()
                self.source = blocks.vector_source_b(data, True)
                self.tb.connect(self.source, throttle, self.sink)

            case 7:
                print("signal_type:\tFSK")
                try:
                    # Попробуем использовать fsk_mod_fc (новый)
                    fsk_mod = digital.fsk_mod_fc(
                        samples_per_symbol=4,
                        bits_per_symbol=1,
                        freq_deviation=1000,
                        sample_rate=self.sample_rate,
                        output_rate=self.sample_rate,
                        verbose=True,
                    )
                except AttributeError:
                    try:
                        # Если нет — попробуем старый fsk_mod
                        fsk_mod = digital.fsk_mod(
                            samples_per_symbol=4,
                            bits_per_symbol=1,
                            freq_deviation=1000,
                            sample_rate=self.sample_rate,
                            output_rate=self.sample_rate,
                            verbose=True,
                        )
                    except AttributeError:
                        # Если оба отсутствуют — используем простой FSK
                        freq1 = 1000
                        freq2 = 2000
                        data = np.random.randint(0, 2, 1000).tolist()
                        self.source = blocks.vector_source_b(data, True)
                        
                        sin1 = analog.sig_source_c(self.sample_rate, analog.GR_SIN_WAVE, freq1, 1, 0, 0)
                        sin2 = analog.sig_source_c(self.sample_rate, analog.GR_SIN_WAVE, freq2, 1, 0, 0)
                        
                        switch = blocks.select(
                            gr.sizeof_gr_complex * 1,
                            [sin1, sin2],
                            0
                        )
                        
                        self.tb.connect(self.source, switch)
                        self.tb.connect(switch, throttle, self.sink)
                        return
                
                data = np.random.randint(0, 2, 1000).tolist()
                self.source = blocks.vector_source_b(data, True)
                self.tb.connect(self.source, fsk_mod)
                self.tb.connect(fsk_mod, throttle, self.sink)

            case _:
                print("Unknown signal type:", self.signal_type)
                exit(1)

        # Запись в файл, если включена
        if self.record_data:
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, self.output_file, False)
            self.tb.connect(throttle, self.file_sink)

        # --- ДОБАВЛЯЕМ CALLBACK BLOCK ---
        if self.visualizer:
            callback_block = CallbackBlock(lambda data: self.visualizer.update_data(data), self.sample_rate)
            self.tb.connect(throttle, callback_block)
            self.tb.connect(callback_block, self.sink)

    def _sink(self, freq):
        try:
            soapy_string = self.options.get("device_soapy_str")
        except:
            soapy_string = "hackrf=0,bias_tx=0,if_gain=47,multiply_const=6"

        print(f"soapy:\t{soapy_string}")

        if not self.sink:
            try:
                self.sink = osmosdr.sink(args=soapy_string)
            except Exception as e:
                print(f"Error initializing sink: {e}")
                exit(1)

        if self.sink is None:
            return

        self.sink.set_time_unknown_pps(osmosdr.time_spec_t())
        self.sink.set_sample_rate(self.sample_rate)
        self.sink.set_center_freq(freq, 0)
        self.sink.set_freq_corr(0, 0)
        self.sink.set_gain(self.RF_gain, 0)
        self.sink.set_if_gain(self.IF_gain, 0)
        self.sink.set_bb_gain(20, 0)
        self.sink.set_antenna("", 0)
        self.sink.set_bandwidth(self.bandwidth, 0)

    def flood_setup(self, freq):
        self.set_freq(freq)

        if self.setup_once is True:
            self._waveform()
            return

        self.setup_once = True
        self.RF_gain, self.IF_gain = self.set_gains()

        self.tb = gr.top_block()
        self._sink(freq)
        self._waveform()

        # Запуск визуализации
        if self.visualizer:
            self.visualizer_thread = VisualizerThread(self.visualizer)
            self.visualizer_thread.start()

    def flood_run(self):
        self.tb.start()
        if self.options.get("hopper_mode") == 1:
            input("Enter to stop\n\n")
        else:
            self._hop_wait()
        try:
            self.tb.stop()
        except Exception as e:
            raise e
        finally:
            self.tb.wait()

    def flood(self, freq):
        try:
            self.flood_setup(freq)
        except Exception as e:
            print(e)
            return e

        self.print_freq()
        try:
            self.flood_run()
        except Exception as e:
            print(e)
            return e

    def set_frequency(self, init_freq, channel):
        if channel == 1:
            freq = init_freq
        else:
            freq = init_freq + (channel - 1) * (
                self.options.get("frequency_delta") * 10e5
            )

        return freq

    def constant(self):
        try:
            self.flood(self.options.get("frequency_start"))
        except Exception as e:
            print(e)
            exit(1)

    def sweeping(self, init_freq, lst_freq):
        channel = 1
        n_channels = (lst_freq - init_freq) // (
            self.options.get("frequency_delta") * 10e5
        )

        while True:
            if channel > n_channels:
                channel = 1
            freq = self.set_frequency(init_freq, channel)

            try:
                self.flood(freq)
                channel += 1
            except Exception as e:
                print(e)
                self.setup_once = False
                time.sleep(0.001)

    def hopper(self, init_freq, lst_freq):
        freq_range = (round(lst_freq) - round(init_freq)) // (
            self.options.get("frequency_delta") * 10e5
        )
        channel = 1

        while True:
            freq = self.set_frequency(init_freq, channel)
            try:
                self.flood(freq)
            except Exception as e:
                print(e)
                self.setup_once = False
                time.sleep(0.001)

            channel = int(random.randint(1, round(freq_range + 1)))

    def rangin(self, iterable):
        started = False
        while True:
            for freq in iterable:
                freq = freq * 10e5
                try:
                    if started:
                        self.tb.stop()
                        self.tb.wait()

                    self.flood_setup(freq)
                    self.print_freq()
                    self.tb.start()
                    started = True
                except Exception as e:
                    print(e)
                    self.setup_once = False

            self.tb.stop()
            self.tb.wait()
            started = False


def_opts = {
    "device_soapy_str": "hackrf=0,bias_tx=0,if_gain=47,multiply_const=6",
    "signal_power": 47,
    "signal_type": 2,
    "frequency_delta": 1,
    "frequency_start": 2400,
    "frequency_end": 2500,
    "hopper_mode": 3,
    "hopper_delay_static": 0.01,
    "hopper_delay_min": 0.001,
    "hopper_delay_max": 20,
    "ranger_str": "1600,2300,r:2400-2500_1",
    "visualize": True,
    "record_data": False,
    "output_file": "rf_output.dat",
}


def prompt_freqs(options):
    while True:
        _f = input("enter minimum center frequency in MHz: ")
        try:
            options["frequency_start"] = float(_f)
            break
        except:
            pass

    while True:
        _f = input("enter end center frequency in MHz: ")
        try:
            options["frequency_end"] = float(_f)
            break
        except:
            pass

    try:
        cpprint(options)
    except:
        pprint(options)

    print("\nusing default values", end="", flush=True)
    for i in range(1, 4):
        time.sleep(1)
        end = ""
        if i == 4:
            end = "\n\n"
        print(".", end=end, flush=True)

    return options


def arg_parser():
    ap = argparse.ArgumentParser(description="Flood the airwaves with RF noise")
    ap.add_argument(
        "-c",
        "--config",
        help="config file to load options from",
        default="config.yaml",
    )
    ap.add_argument(
        "-d",
        "--device_soapy_str",
        help="soapysdr device string",
        type=str,
        default=def_opts.get("device_soapy_str"),
    )
    ap.add_argument(
        "-f",
        "--frequency_start",
        help="min center frequency in MHz",
        type=float,
        default=def_opts.get("frequency_start"),
    )
    ap.add_argument(
        "-m",
        "--frequency-end",
        help="max center frequency in MHz",
        type=float,
        default=def_opts.get("frequency_end"),
    )
    ap.add_argument(
        "-p",
        "--signal_power",
        help="RF signal_power in dB",
        type=int,
        default=def_opts.get("signal_power"),
    )
    ap.add_argument(
        "-w",
        "--signal_type",
        help="source signal_type",
        type=int,
        default=def_opts.get("signal_type"),
        choices=[1, 2, 3, 4, 5, 6, 7],
    )
    ap.add_argument(
        "-o",
        "--hopper_mode",
        help="channel hopping mechanism",
        type=float,
        default=def_opts.get("hopper_mode"),
        choices=[1, 2, 3, 3.1, 4, 4.1],
    )
    ap.add_argument(
        "-t",
        "--hopper_delay_static",
        help="time to stay on each frequency hopped to",
        type=float,
        default=def_opts.get("hopper_delay_static"),
    )
    ap.add_argument(
        "-l",
        "--hopper_delay_min",
        help="minimum value for random hopper_delay_static",
        type=float,
        default=def_opts.get("hopper_delay_min"),
    )
    ap.add_argument(
        "-u",
        "--hopper_delay_max",
        help="maximum value for random hopper_delay_static",
        type=float,
        default=def_opts.get("hopper_delay_max"),
    )
    ap.add_argument(
        "-r",
        "--ranger_str",
        help="comma separated range values for ranger function",
        type=str,
        default=def_opts.get("ranger_str"),
    )
    ap.add_argument(
        "-e",
        "--frequency_delta",
        help="difference between frequencies during frequency hopping",
        type=float,
        default=def_opts.get("frequency_delta"),
    )
    ap.add_argument(
        "-v",
        "--visualize",
        help="Enable real-time visualization",
        action="store_true",
        default=def_opts.get("visualize"),
    )
    ap.add_argument(
        "-R",
        "--record_data",
        help="Record data to file",
        action="store_true",
        default=def_opts.get("record_data"),
    )
    ap.add_argument(
        "-O",
        "--output_file",
        help="Output file for recorded data",
        type=str,
        default=def_opts.get("output_file"),
    )

    return ap.parse_args()


def merge_options(options, arg_vars):
    for key, value in arg_vars.items():
        if key == "config":
            continue
        if value is not None and value != def_opts[key]:
            options[key] = value
    return options


def load_config():
    args = arg_parser()

    try:
        config_file = open(args.config, "r")
        options = yaml.safe_load(config_file)
        config_file.close()
    except Exception as e:
        print(f"failed to load config ({args.config})\n{e}\n")
        options = def_opts
        options = prompt_freqs(options)

    options = merge_options(options, vars(args))

    try:
        cpprint(options)
    except:
        pprint(options)

    return options


def main():
    options = load_config()

    wavy = FloodAir(options)

    freq = options.get("frequency_start") * 10e5
    freq_max = options.get("frequency_end") * 10e5

    if freq_max < freq:
        print("frequency_end must be greater than frequency_start")
        exit(1)

    options["frequency_start"] = freq

    wavy.hopper_mode = options.get("hopper_mode")

    match wavy.hopper_mode:
        case 1:
            wavy.hopper_entropy = False
            wavy.constant()
        case 2:
            wavy.hopper_entropy = False
            wavy.sweeping(freq, freq_max)
        case 3:
            wavy.hopper_entropy = False
            wavy.hopper(freq, freq_max)
        case 3.1:
            wavy.hopper_entropy = True
            wavy.hopper(freq, freq_max)
        case 4:
            wavy.rangin(
                ranger.Ranger(
                    options.get("ranger_str"),
                    sleep_secs=options.get("hopper_delay_static"),
                )
            )
        case 4.1:
            wavy.hopper_entropy = True
            wavy.rangin(
                ranger.Ranger(
                    options.get("ranger_str"),
                    sleep_secs=options.get("hopper_delay_static"),
                    entropy=True,
                )
            )
        case _:
            print(
                "unknown 'hopper_mode'. options:\n",
                "1\tconstant\n",
                "2\tsweeping\n",
                "3\thopper\n",
                "3.1\thopper with entropy\n",
            )
            exit(1)


if __name__ == "__main__":
    main()
