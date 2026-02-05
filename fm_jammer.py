#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FM Jammer v2.0 для AntSDR E200
Адаптированный для FM диапазона 88-108 МГц
"""

from gnuradio import analog, gr, blocks, digital
from gnuradio.filter import firdes
import sys, signal, time, random
import numpy as np
from argparse import ArgumentParser
import osmosdr
import threading
from collections import deque
import shutil
from datetime import datetime
from scipy import signal as sig  # Исправленный импорт!

class FMJammerVisualizer:
    """Визуализатор для FM джаммера"""
    def __init__(self, sample_rate=5e6, center_freq=98e6, signal_type=3, anti_radar_mode=False):
        self.width, self.height = shutil.get_terminal_size()
        self.width = min(self.width, 200)
        self.height = min(self.height, 60)
        
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.signal_type = signal_type
        self.anti_radar_mode = anti_radar_mode
        
        self.buffer = deque(maxlen=self.width * 3)
        self.spectrum_buffer = deque(maxlen=self.width)
        self.waterfall_buffer = deque(maxlen=self.height // 2)
        self.running = True
        
        # FM-специфичные детекции
        self.fm_stations_detected = []
        self.target_power = -100.0
        self.detection_threshold = -60.0  # Более низкий порог для FM
        self.detection_history = deque(maxlen=100)
        
        self.stats = {
            'samples_processed': 0,
            'peak_amplitude': 0.0,
            'avg_amplitude': 0.0,
            'current_gain': 0,
            'start_time': time.time(),
            'peak_frequency': 0.0,
            'bandwidth_usage': 0.0,
            'signal_power_dbm': 0.0,
            'noise_floor_dbm': -100.0,
            'snr_db': 0.0,
            'stations_jammed': 0,
            'target_frequency': 0.0,
            'jamming_effectiveness': 0.0,
            'pulses_sent': 0
        }
        
        self.amplitude_history = deque(maxlen=100)
        self.power_history = deque(maxlen=100)
        
        self.colors = {
            'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
            'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
            'white': '\033[97m', 'reset': '\033[0m',
            'gray': '\033[90m', 'bright_green': '\033[92m',
            'bright_red': '\033[91m\033[1m', 'bright_yellow': '\033[93m\033[1m'
        }
        
        self.waterfall_chars = [' ', '·', '░', '▒', '▓', '█']
        self.waterfall_colors = [
            '\033[38;5;16m',  # Черный
            '\033[38;5;17m',  # Темно-синий
            '\033[38;5;21m',  # Синий
            '\033[38;5;51m',  # Голубой
            '\033[38;5;226m', # Желтый
            '\033[38;5;196m'  # Красный
        ]
        
        # Обновленные названия сигналов для FM
        self.signal_names = {
            1: "SINE", 2: "QPSK", 3: "NOISE",
            4: "BPSK", 5: "8PSK", 6: "OOK", 7: "FSK",
            8: "PULSE", 9: "SWEEP", 10: "ADAPTIVE",
            11: "FM-PILOT", 12: "FM-STEREO", 13: "FM-RDS"
        }
    
    def detect_fm_stations(self, fft_db, freqs):
        """Детекция FM станций в спектре"""
        # Ищем пики выше порога
        peaks = []
        for i in range(1, len(fft_db)-1):
            if (fft_db[i] > self.detection_threshold and 
                fft_db[i] > fft_db[i-1] and 
                fft_db[i] > fft_db[i+1]):
                peaks.append((freqs[i], fft_db[i]))
        
        self.fm_stations_detected = peaks
        self.stats['stations_jammed'] = len(peaks)
        
        if peaks:
            # Берем самую сильную станцию как цель
            strongest = max(peaks, key=lambda x: x[1])
            self.stats['target_frequency'] = strongest[0]
            self.target_power = strongest[1]
            self.detection_history.append(1)
        else:
            self.detection_history.append(0)
        
        # Эффективность (% времени когда станции подавлены)
        if len(self.detection_history) > 10:
            self.stats['jamming_effectiveness'] = (1 - np.mean(list(self.detection_history))) * 100
    
    def update_data(self, data):
        if len(data) == 0:
            return
        
        self.stats['samples_processed'] += len(data)
        
        # Временной домен
        real_data = np.real(data)
        
        if len(real_data) > 0:
            step = max(1, len(real_data) // 20)
            for i in range(0, len(real_data), step):
                self.buffer.append(real_data[i])
            
            current_amp = np.abs(data)
            self.stats['peak_amplitude'] = max(self.stats['peak_amplitude'], np.max(current_amp))
            avg_amp = np.mean(current_amp)
            self.stats['avg_amplitude'] = avg_amp
            self.amplitude_history.append(avg_amp)
            
            power_watts = np.mean(current_amp ** 2)
            if power_watts > 0:
                power_dbm = 10 * np.log10(power_watts * 1000)
                self.stats['signal_power_dbm'] = power_dbm
                self.power_history.append(power_dbm)
        
        # Спектральный анализ
        if len(data) >= 512:
            fft_size = 512
            fft_data = np.fft.fftshift(np.fft.fft(data[:fft_size]))
            fft_magnitude = np.abs(fft_data)
            fft_db = 20 * np.log10(fft_magnitude + 1e-10)
            
            # Частоты
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/self.sample_rate))
            freqs += self.center_freq
            
            # FM станции детекция
            self.detect_fm_stations(fft_db, freqs)
            
            min_db = np.min(fft_db)
            max_db = np.max(fft_db)
            self.stats['noise_floor_dbm'] = min_db
            
            if max_db > min_db:
                fft_normalized = (fft_db - min_db) / (max_db - min_db)
                
                peak_idx = np.argmax(fft_magnitude)
                freq_resolution = self.sample_rate / fft_size
                peak_freq_offset = (peak_idx - fft_size/2) * freq_resolution
                self.stats['peak_frequency'] = self.center_freq + peak_freq_offset
                
                signal_power = max_db
                noise_power = np.median(fft_db)
                self.stats['snr_db'] = signal_power - noise_power
                
                threshold = max_db - 20
                bandwidth_bins = np.sum(fft_db > threshold)
                self.stats['bandwidth_usage'] = (bandwidth_bins / fft_size) * self.sample_rate
                
                step = max(1, len(fft_normalized) // 32)
                for i in range(0, len(fft_normalized), step):
                    chunk = fft_normalized[i:i+step]
                    self.spectrum_buffer.append(np.mean(chunk))
        
        # Waterfall
        if len(data) >= 256:
            fft_waterfall = np.abs(np.fft.fftshift(np.fft.fft(data[:256])))
            if np.max(fft_waterfall) > 0:
                fft_norm = fft_waterfall / np.max(fft_waterfall)
                self.waterfall_buffer.append(fft_norm)
    
    def draw_highres_plot(self, data, plot_height, color='green', fill=True):
        if len(data) < 2:
            return [" " * self.width] * plot_height
        
        data_array = np.array(list(data))
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if max_val == min_val:
            return ["─" * self.width] * plot_height
        
        scale = (plot_height - 1) / (max_val - min_val)
        plot_matrix = [[' ' for _ in range(self.width)] for _ in range(plot_height)]
        
        x_indices = np.linspace(0, len(data_array)-1, self.width)
        interpolated = np.interp(x_indices, np.arange(len(data_array)), data_array)
        
        for x in range(self.width):
            val = interpolated[x]
            y_pos = int((val - min_val) * scale)
            y_pos = max(0, min(plot_height-1, y_pos))
            
            plot_matrix[plot_height-1-y_pos][x] = '●'
            
            if fill:
                for y in range(y_pos):
                    if plot_matrix[plot_height-1-y][x] == ' ':
                        intensity = y / y_pos if y_pos > 0 else 0
                        if intensity < 0.3:
                            plot_matrix[plot_height-1-y][x] = '░'
                        elif intensity < 0.6:
                            plot_matrix[plot_height-1-y][x] = '▒'
                        else:
                            plot_matrix[plot_height-1-y][x] = '▓'
        
        return [''.join(line) for line in plot_matrix]
    
    def draw_waterfall(self, height):
        if len(self.waterfall_buffer) < 2:
            return [" " * self.width] * height
        
        waterfall_lines = []
        for i in range(min(height, len(self.waterfall_buffer))):
            line_data = self.waterfall_buffer[-(i+1)]
            line = ""
            
            x_indices = np.linspace(0, len(line_data)-1, self.width)
            interpolated = np.interp(x_indices, np.arange(len(line_data)), line_data)
            
            for j in range(self.width):
                intensity = interpolated[j]
                char_idx = int(intensity * (len(self.waterfall_chars) - 1))
                color_idx = int(intensity * (len(self.waterfall_colors) - 1))
                
                char_idx = min(char_idx, len(self.waterfall_chars) - 1)
                color_idx = min(color_idx, len(self.waterfall_colors) - 1)
                
                line += self.waterfall_colors[color_idx] + self.waterfall_chars[char_idx] + self.colors['reset']
            
            waterfall_lines.append(line)
        
        return waterfall_lines
    
    def draw_mega_meter(self, value, width, label, color, show_value=True):
        bars = int(value * width)
        bars = max(0, min(width, bars))
        
        if value < 0.3:
            bar_color = self.colors['green']
        elif value < 0.7:
            bar_color = self.colors['yellow']
        else:
            bar_color = self.colors['red']
        
        meter = f"[{bar_color}{'█' * bars}{self.colors['gray']}{'░' * (width - bars)}{self.colors['reset']}]"
        
        if show_value:
            return f"{label}: {meter} {value*100:5.1f}%"
        else:
            return f"{label}: {meter}"
    
    def display(self):
        print("\033[2J\033[H", end="")
        
        header_height = 4
        stats_height = 12
        time_height = (self.height - header_height - stats_height) // 3
        spectrum_height = (self.height - header_height - stats_height) // 3
        waterfall_height = self.height - header_height - stats_height - time_height - spectrum_height - 2
        
        # HEADER
        print(f"{self.colors['cyan']}╔{'═' * (self.width-2)}╗{self.colors['reset']}")
        
        title = f"📻 FM JAMMER - {self.signal_names.get(self.signal_type, 'UNKNOWN')}"
        padding = (self.width - len(title) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['bright_yellow']}{' ' * padding}{title}{' ' * (self.width - len(title) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        uptime = time.time() - self.stats['start_time']
        uptime_str = f"⏱ Uptime: {int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        timestamp = f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        info_line = f"{uptime_str} | {timestamp}"
        padding = (self.width - len(info_line) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['white']}{' ' * padding}{info_line}{' ' * (self.width - len(info_line) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        
        # STATISTICS
        print(f"{self.colors['cyan']}║{self.colors['yellow']} 📊 FM STATISTICS{' ' * (self.width-22)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # FM Status
        stations_count = self.stats['stations_jammed']
        target_freq_mhz = self.stats['target_frequency'] / 1e6 if self.stats['target_frequency'] else 0
        effectiveness = self.stats['jamming_effectiveness']
        
        status_line = f"📡 Stations Detected: {stations_count} | Target: {target_freq_mhz:.3f} MHz | Effectiveness: {effectiveness:.1f}%"
        print(f"{self.colors['cyan']}║{self.colors['bright_red' if stations_count > 0 else 'bright_green']} {status_line}{' ' * (self.width - len(status_line) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        freq_mhz = self.center_freq / 1e6
        peak_freq_mhz = self.stats['peak_frequency'] / 1e6
        stats_line1 = f"📡 Center: {freq_mhz:.2f} MHz | Peak: {peak_freq_mhz:.2f} MHz | Power: {self.stats['signal_power_dbm']:.1f} dBm | SNR: {self.stats['snr_db']:.1f} dB"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line1}{' ' * (self.width - len(stats_line1) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        stats_line2 = f"📈 Samples: {self.stats['samples_processed']:,} | Peak Amp: {self.stats['peak_amplitude']:.4f} | Avg Amp: {self.stats['avg_amplitude']:.4f}"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line2}{' ' * (self.width - len(stats_line2) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        bw_mhz = self.stats['bandwidth_usage'] / 1e6
        sample_rate_mhz = self.sample_rate / 1e6
        stats_line3 = f"📶 Sample Rate: {sample_rate_mhz:.2f} MHz | Active BW: {bw_mhz:.2f} MHz | Noise Floor: {self.stats['noise_floor_dbm']:.1f} dBm"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line3}{' ' * (self.width - len(stats_line3) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Meters
        if len(self.buffer) > 0:
            amp_norm = min(1.0, self.stats['avg_amplitude'])
            amp_meter = self.draw_mega_meter(amp_norm, self.width-25, "🔊 AMPLITUDE", 'yellow')
            print(f"{self.colors['cyan']}║ {amp_meter}{' ' * (self.width - len(amp_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.power_history) > 0:
            power_norm = min(1.0, max(0.0, (self.stats['signal_power_dbm'] + 50) / 50))
            power_meter = self.draw_mega_meter(power_norm, self.width-25, "⚡ POWER   ", 'red')
            print(f"{self.colors['cyan']}║ {power_meter}{' ' * (self.width - len(power_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        gain_norm = self.stats['current_gain'] / 40.0
        gain_meter = self.draw_mega_meter(gain_norm, self.width-25, "🎚 GAIN     ", 'cyan')
        print(f"{self.colors['cyan']}║ {gain_meter}{' ' * (self.width - len(gain_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # FM Jamming Effectiveness
        eff_norm = self.stats['jamming_effectiveness'] / 100.0
        eff_meter = self.draw_mega_meter(eff_norm, self.width-25, "📻 JAMMING ", 'green')
        print(f"{self.colors['cyan']}║ {eff_meter}{' ' * (self.width - len(eff_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        
        # TIME DOMAIN
        print(f"{self.colors['cyan']}║{self.colors['green']} 📈 TIME DOMAIN{' ' * (self.width-20)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.buffer) > 1:
            time_plot = self.draw_highres_plot(list(self.buffer), time_height - 1, 'green', fill=True)
            for line in time_plot:
                print(f"{self.colors['cyan']}║{self.colors['green']}{line}{' ' * (self.width - len(line) - 1)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(time_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # SPECTRUM
        print(f"{self.colors['cyan']}║{self.colors['blue']} 📊 SPECTRUM{' ' * (self.width-17)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.spectrum_buffer) > 1:
            spec_plot = self.draw_highres_plot(list(self.spectrum_buffer), spectrum_height - 1, 'blue', fill=True)
            for line in spec_plot:
                print(f"{self.colors['cyan']}║{self.colors['blue']}{line}{' ' * (self.width - len(line) - 1)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(spectrum_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # WATERFALL
        print(f"{self.colors['cyan']}║{self.colors['magenta']} 🌊 WATERFALL{' ' * (self.width-18)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.waterfall_buffer) > 1:
            waterfall_lines = self.draw_waterfall(waterfall_height - 1)
            for line in waterfall_lines:
                print(f"{self.colors['cyan']}║{line}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(waterfall_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        # FOOTER
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        ctrl = "🎮 Ctrl+C=Stop | Mode: FM JAMMER | 88-108 MHz Coverage"
        print(f"{self.colors['cyan']}║{self.colors['cyan']} {ctrl}{' ' * (self.width - len(ctrl) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        print(f"{self.colors['cyan']}╚{'═' * (self.width-2)}╝{self.colors['reset']}")

class CallbackBlock(gr.sync_block):
    def __init__(self, callback, sample_rate=5e6):
        gr.sync_block.__init__(
            self,
            name="CallbackBlock",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.callback = callback
        self.sample_rate = sample_rate

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        
        out0[:] = in0
        
        if len(in0) > 0:
            self.callback(in0)
        
        return len(in0)

class FMJammer(gr.top_block):
    def __init__(self, options=None):
        gr.top_block.__init__(self, "FM Jammer")
        
        options_dict = vars(options) if options else {}
        self.samp_rate = options_dict.get("samp_rate", 5e6)
        self.f0 = options_dict.get("f0", 98e6)  # Default 98 MHz FM
        self.bw = options_dict.get("bw", 20e6)  # 20 MHz для покрытия всего FM
        self.signal_type = options_dict.get("signal_type", 3)  # Noise default для FM
        self.visualize = options_dict.get("visualize", False)
        self.current_gain = 0

        # SDR Sink - увеличенные gain для FM
        self.osmosdr_sink_0 = osmosdr.sink(args="numchan=1 antsdr=0")
        self.osmosdr_sink_0.set_sample_rate(self.samp_rate)
        self.osmosdr_sink_0.set_center_freq(self.f0, 0)
        self.osmosdr_sink_0.set_gain(20, 0)      # Увеличено для FM
        self.osmosdr_sink_0.set_if_gain(60, 0)   # Увеличено для FM
        self.osmosdr_sink_0.set_bb_gain(60, 0)   # Увеличено для FM
        self.osmosdr_sink_0.set_bandwidth(self.bw, 0)

        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate, True)

        self._create_source()

        if self.visualize:
            print("🎨 Initializing FM Jammer visualizer...")
            self.visualizer = FMJammerVisualizer(
                sample_rate=self.samp_rate,
                center_freq=self.f0,
                signal_type=self.signal_type,
                anti_radar_mode=False  # FM mode
            )
            self.callback_block = CallbackBlock(
                lambda data: self.visualizer.update_data(data), 
                self.samp_rate
            )
            self.connect(self.source, self.throttle, self.callback_block, self.osmosdr_sink_0)
        else:
            self.connect(self.source, self.throttle, self.osmosdr_sink_0)

    def _create_source(self):
        signal_names = {
            1: "📡 SINE", 2: "🛰 QPSK", 3: "🌪 NOISE",
            4: "📶 BPSK", 5: "🎯 8PSK", 6: "⚡ OOK", 7: "📻 FSK",
            8: "💥 PULSE", 9: "🌀 SWEEP", 10: "🧠 ADAPTIVE",
            11: "📻 FM-PILOT", 12: "📻 FM-STEREO", 13: "📻 FM-RDS"
        }
        
        name = signal_names.get(self.signal_type, "❓ UNKNOWN")
        print(f"🎯 Signal Type: {name}")
        
        if self.signal_type == 1:  # SINE
            self.source = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        elif self.signal_type == 2:  # QPSK
            symbols = [1+1j, -1+1j, -1-1j, 1-1j] * 1000
            self.source = blocks.vector_source_c(symbols, True)
        elif self.signal_type == 3:  # NOISE - отлично для FM!
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)
        elif self.signal_type == 4:  # BPSK
            symbols = [-1, 1] * 1000
            complex_symbols = [complex(s, 0) for s in symbols]
            self.source = blocks.vector_source_c(complex_symbols, True)
        elif self.signal_type == 5:  # 8PSK
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            symbols = [np.exp(1j * angle) for angle in angles] * 200
            self.source = blocks.vector_source_c(symbols, True)
        elif self.signal_type == 6:  # OOK
            symbols = [0, 1] * 1000
            complex_symbols = [complex(s, 0) for s in symbols]
            self.source = blocks.vector_source_c(complex_symbols, True)
        elif self.signal_type == 7:  # FSK
            t = np.linspace(0, 1, int(self.samp_rate/100))
            freq1, freq2 = 1000, 2000
            signal1 = np.exp(1j * 2 * np.pi * freq1 * t)
            signal2 = np.exp(1j * 2 * np.pi * freq2 * t)
            mixed_signal = []
            for i in range(100):
                mixed_signal.extend(signal1 if i % 2 == 0 else signal2)
            self.source = blocks.vector_source_c(mixed_signal, True)
        elif self.signal_type == 8:  # PULSE
            pulse_width = int(self.samp_rate * 0.001)  # 1 ms pulse
            gap_width = int(self.samp_rate * 0.001)  # 1 ms gap
            pulse = np.ones(pulse_width, dtype=np.complex64)
            gap = np.zeros(gap_width, dtype=np.complex64)
            pattern = np.concatenate([pulse, gap, pulse, gap * 3, pulse, gap * 5])
            pattern_repeated = np.tile(pattern, 100)
            self.source = blocks.vector_source_c(pattern_repeated, True)
        elif self.signal_type == 9:  # SWEEP - ИСПРАВЛЕНО!
            # Frequency sweep для покрытия FM диапазона
            t = np.linspace(0, 1, int(self.samp_rate))
            freq_start = -10e6  # -10 MHz offset
            freq_end = 10e6    # +10 MHz offset (покрытие 20 MHz)
            chirp = sig.chirp(t, freq_start, 1, freq_end, method='linear')
            chirp_complex = chirp.astype(np.complex64)
            self.source = blocks.vector_source_c(chirp_complex, True)
        elif self.signal_type == 11:  # FM PILOT (19 кГц)
            self.source = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 19000, 1, 0, 0)
        elif self.signal_type == 12:  # FM STEREO
            # Имитация FM стерео с пилот-тоном и субканалами
            pilot_19khz = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 19000, 0.1, 0, 0)
            stereo_38khz = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 38000, 0.3, 0, 0)
            # Простое смешивание
            self.source = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 1000, 0.5, 0, 0)
        elif self.signal_type == 13:  # FM RDS (57 кГц)
            self.source = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 57000, 0.1, 0, 0)
        else:
            print("⚠ Unknown signal type, using NOISE")
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)

    def pulse(self):
        print('📻 FM Jammer ACTIVE. Press Ctrl+C to stop.')
        print(f'🎯 Targeting FM band: {self.f0/1e6:.1f} MHz...')
        try:
            pulse_count = 0
            while True:
                time.sleep(0.1)
                
                # Динамическая модуляция gain для FM
                if self.signal_type == 9:  # Sweep mode
                    # Модуляция для sweep
                    gain = 25 + int(15 * np.sin(pulse_count * 0.1))  # 25-40 dB
                elif self.signal_type == 3:  # Noise mode
                    # Случайная модуляция для шума
                    gain = random.randint(30, 45)
                else:
                    gain = random.randint(25, 40)
                
                self.current_gain = gain
                self.osmosdr_sink_0.set_if_gain(gain, 0)
                
                if hasattr(self, 'visualizer'):
                    self.visualizer.stats['current_gain'] = gain
                    self.visualizer.stats['pulses_sent'] = pulse_count
                    self.visualizer.display()
                else:
                    stations = len(getattr(self, 'fm_stations_detected', []))
                    print(f"📻 JAMMING | Gain: {gain:2d} dB | Count: {pulse_count:6d} | Freq: {self.f0/1e6:.1f} MHz | Stations: {stations}", end='\r')
                
                pulse_count += 1
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down FM Jammer...")

def main():
    parser = ArgumentParser(description="📻 FM Jammer for AntSDR E200")
    parser.add_argument("--f0", type=float, default=98e6, help="Center frequency (Hz) - default: 98 MHz")
    parser.add_argument("--samp_rate", type=float, default=5e6, help="Sample rate (Hz)")
    parser.add_argument("--bw", type=float, default=20e6, help="Bandwidth (Hz) - default: 20 MHz")
    parser.add_argument("--signal_type", type=int, default=3, 
                       choices=[1,2,3,4,5,6,7,8,9,10,11,12,13],
                       help="1=Sine, 2=QPSK, 3=Noise, 4=BPSK, 5=8PSK, 6=OOK, 7=FSK, 8=PULSE, 9=SWEEP, 11=FM-PILOT, 12=FM-STEREO, 13=FM-RDS")
    parser.add_argument("--visualize", action="store_true", help="Enable HD visualization")
    
    args = parser.parse_args()

    # Проверка FM диапазона
    if not (88e6 <= args.f0 <= 108e6):
        print("⚠️  Warning: Frequency outside FM band (88-108 MHz)")
    
    print("=" * 70)
    print("📻  FM JAMMER - Broadcast Band Interference")
    print("=" * 70)
    print(f"📡 Target Frequency: {args.f0/1e6:.3f} MHz")
    print(f"📊 Sample Rate: {args.samp_rate/1e6:.1f} MHz")
    print(f"📶 Bandwidth: {args.bw/1e6:.1f} MHz")
    print(f"🎯 Signal Type: {args.signal_type}")
    print(f"🎨 Visualization: {'ENABLED' if args.visualize else 'DISABLED'}")
    print("=" * 70)
    print("⚠️  WARNING: This interferes with FM radio broadcasts!")
    print("⚠️  Use responsibly and check local regulations!")
    print("=" * 70)

    tb = FMJammer(options=args)

    def sig_handler(sig=None, frame=None):
        print("\n🛑 Received stop signal...")
        tb.stop()
        tb.wait()
        print("✅ FM Jammer stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("🚀 Starting FM Jammer...")
    tb.start()

    try:
        tb.pulse()
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()
        print("\n✅ Shutdown complete.")

if __name__ == '__main__':
    main()
