#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

class MegaVisualizer:
    def __init__(self, sample_rate=2e6, center_freq=2437e6, signal_type=3):
        # Получаем размер терминала
        self.width, self.height = shutil.get_terminal_size()
        self.width = min(self.width, 200)
        self.height = min(self.height, 60)
        
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.signal_type = signal_type
        
        # Буферы с адаптивным размером
        self.buffer = deque(maxlen=self.width * 3)
        self.spectrum_buffer = deque(maxlen=self.width)
        self.waterfall_buffer = deque(maxlen=self.height // 2)
        self.running = True
        
        # Статистика
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
            'snr_db': 0.0
        }
        
        # История для графиков
        self.amplitude_history = deque(maxlen=100)
        self.power_history = deque(maxlen=100)
        
        # Цвета
        self.colors = {
            'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
            'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
            'white': '\033[97m', 'reset': '\033[0m',
            'gray': '\033[90m', 'bright_green': '\033[92m',
            'bright_red': '\033[91m\033[1m', 'bright_yellow': '\033[93m\033[1m'
        }
        
        # Градиенты
        self.waterfall_chars = [' ', '·', '░', '▒', '▓', '█']
        self.waterfall_colors = [
            '\033[38;5;16m',  # Черный
            '\033[38;5;17m',  # Темно-синий
            '\033[38;5;21m',  # Синий
            '\033[38;5;51m',  # Голубой
            '\033[38;5;226m', # Желтый
            '\033[38;5;196m'  # Красный
        ]
        
        self.signal_names = {
            1: "SINE WAVE", 2: "QPSK", 3: "NOISE",
            4: "BPSK", 5: "8PSK", 6: "OOK", 7: "FSK"
        }
    
    def update_data(self, data):
        if len(data) == 0:
            return
        
        self.stats['samples_processed'] += len(data)
        
        # Временной домен
        real_data = np.real(data)
        imag_data = np.imag(data)
        
        if len(real_data) > 0:
            # Добавляем данные
            step = max(1, len(real_data) // 20)
            for i in range(0, len(real_data), step):
                self.buffer.append(real_data[i])
            
            # Статистика амплитуды
            current_amp = np.abs(data)
            self.stats['peak_amplitude'] = max(self.stats['peak_amplitude'], np.max(current_amp))
            avg_amp = np.mean(current_amp)
            self.stats['avg_amplitude'] = avg_amp
            self.amplitude_history.append(avg_amp)
            
            # Мощность сигнала в dBm
            power_watts = np.mean(current_amp ** 2)
            if power_watts > 0:
                power_dbm = 10 * np.log10(power_watts * 1000)  # Переводим в dBm
                self.stats['signal_power_dbm'] = power_dbm
                self.power_history.append(power_dbm)
        
        # Спектральный анализ
        if len(data) >= 512:
            fft_size = 512
            fft_data = np.fft.fftshift(np.fft.fft(data[:fft_size]))
            fft_magnitude = np.abs(fft_data)
            fft_db = 20 * np.log10(fft_magnitude + 1e-10)
            
            # Нормализация
            min_db = np.min(fft_db)
            max_db = np.max(fft_db)
            self.stats['noise_floor_dbm'] = min_db
            
            if max_db > min_db:
                fft_normalized = (fft_db - min_db) / (max_db - min_db)
                
                # Пиковая частота
                peak_idx = np.argmax(fft_magnitude)
                freq_resolution = self.sample_rate / fft_size
                peak_freq_offset = (peak_idx - fft_size/2) * freq_resolution
                self.stats['peak_frequency'] = self.center_freq + peak_freq_offset
                
                # SNR
                signal_power = max_db
                noise_power = np.median(fft_db)
                self.stats['snr_db'] = signal_power - noise_power
                
                # Использование полосы (энергия > -20dB от пика)
                threshold = max_db - 20
                bandwidth_bins = np.sum(fft_db > threshold)
                self.stats['bandwidth_usage'] = (bandwidth_bins / fft_size) * self.sample_rate
                
                # Добавляем в буфер спектра
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
        """Улучшенный график с заполнением"""
        if len(data) < 2:
            return [" " * self.width] * plot_height
        
        data_array = np.array(list(data))
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if max_val == min_val:
            return ["─" * self.width] * plot_height
        
        scale = (plot_height - 1) / (max_val - min_val)
        plot_matrix = [[' ' for _ in range(self.width)] for _ in range(plot_height)]
        
        # Интерполяция для плавности
        x_indices = np.linspace(0, len(data_array)-1, self.width)
        interpolated = np.interp(x_indices, np.arange(len(data_array)), data_array)
        
        for x in range(self.width):
            val = interpolated[x]
            y_pos = int((val - min_val) * scale)
            y_pos = max(0, min(plot_height-1, y_pos))
            
            # Рисуем точку
            plot_matrix[plot_height-1-y_pos][x] = '●'
            
            # Заполнение под графиком
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
        """Улучшенный waterfall с цветовым градиентом"""
        if len(self.waterfall_buffer) < 2:
            return [" " * self.width] * height
        
        waterfall_lines = []
        for i in range(min(height, len(self.waterfall_buffer))):
            line_data = self.waterfall_buffer[-(i+1)]
            line = ""
            
            # Интерполяция для ширины экрана
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
        """Улучшенный индикатор"""
        bars = int(value * width)
        bars = max(0, min(width, bars))
        
        # Цветовая градация
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
    
    def draw_mini_graph(self, data, width, height, label):
        """Мини-график для истории"""
        if len(data) < 2:
            return [f"{label}: [no data]"]
        
        data_array = np.array(list(data))
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if max_val == min_val:
            return [f"{label}: [flat]"]
        
        lines = []
        lines.append(f"{label}: [{min_val:.1f} to {max_val:.1f}]")
        
        # Простой ASCII график
        scale = (height - 1) / (max_val - min_val)
        for h in range(height):
            line = ""
            threshold = max_val - (h * (max_val - min_val) / height)
            for i in range(min(width, len(data_array))):
                if data_array[-(width-i)] >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        return lines
    
    def display(self):
        """Мега-дисплей с полной информацией"""
        print("\033[2J\033[H", end="")  # Очистка экрана
        
        # Расчет высот секций
        header_height = 4
        stats_height = 8
        time_height = (self.height - header_height - stats_height) // 3
        spectrum_height = (self.height - header_height - stats_height) // 3
        waterfall_height = self.height - header_height - stats_height - time_height - spectrum_height - 2
        
        # ═══ HEADER ═══
        print(f"{self.colors['cyan']}╔{'═' * (self.width-2)}╗{self.colors['reset']}")
        
        title = f"⚡ RF JAMMER CONTROL CENTER - {self.signal_names.get(self.signal_type, 'UNKNOWN')} MODE ⚡"
        padding = (self.width - len(title) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['bright_yellow']}{' ' * padding}{title}{' ' * (self.width - len(title) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Время работы
        uptime = time.time() - self.stats['start_time']
        uptime_str = f"⏱ Uptime: {int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
        timestamp = f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        info_line = f"{uptime_str} | {timestamp}"
        padding = (self.width - len(info_line) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['white']}{' ' * padding}{info_line}{' ' * (self.width - len(info_line) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        
        # ═══ STATISTICS PANEL ═══
        print(f"{self.colors['cyan']}║{self.colors['yellow']} 📊 REAL-TIME STATISTICS{' ' * (self.width-28)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Строка 1: Частота и мощность
        freq_mhz = self.center_freq / 1e6
        peak_freq_mhz = self.stats['peak_frequency'] / 1e6
        stats_line1 = f"📡 Center: {freq_mhz:.2f} MHz | Peak: {peak_freq_mhz:.2f} MHz | Power: {self.stats['signal_power_dbm']:.1f} dBm | SNR: {self.stats['snr_db']:.1f} dB"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line1}{' ' * (self.width - len(stats_line1) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Строка 2: Амплитуда и семплы
        stats_line2 = f"📈 Samples: {self.stats['samples_processed']:,} | Peak Amp: {self.stats['peak_amplitude']:.4f} | Avg Amp: {self.stats['avg_amplitude']:.4f}"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line2}{' ' * (self.width - len(stats_line2) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Строка 3: Bandwidth
        bw_mhz = self.stats['bandwidth_usage'] / 1e6
        sample_rate_mhz = self.sample_rate / 1e6
        stats_line3 = f"📶 Sample Rate: {sample_rate_mhz:.2f} MHz | Active BW: {bw_mhz:.2f} MHz | Noise Floor: {self.stats['noise_floor_dbm']:.1f} dBm"
        print(f"{self.colors['cyan']}║{self.colors['white']} {stats_line3}{' ' * (self.width - len(stats_line3) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Индикаторы
        if len(self.buffer) > 0:
            amp_norm = min(1.0, self.stats['avg_amplitude'])
            amp_meter = self.draw_mega_meter(amp_norm, self.width-25, "🔊 AMPLITUDE", 'yellow')
            print(f"{self.colors['cyan']}║ {amp_meter}{' ' * (self.width - len(amp_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.power_history) > 0:
            # Нормализуем мощность от -50 до 0 dBm
            power_norm = min(1.0, max(0.0, (self.stats['signal_power_dbm'] + 50) / 50))
            power_meter = self.draw_mega_meter(power_norm, self.width-25, "⚡ POWER   ", 'red')
            print(f"{self.colors['cyan']}║ {power_meter}{' ' * (self.width - len(power_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        gain_norm = self.stats['current_gain'] / 40.0
        gain_meter = self.draw_mega_meter(gain_norm, self.width-25, "🎚 GAIN     ", 'cyan')
        print(f"{self.colors['cyan']}║ {gain_meter}{' ' * (self.width - len(gain_meter) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        
        # ═══ TIME DOMAIN ═══
        print(f"{self.colors['cyan']}║{self.colors['green']} 📈 TIME DOMAIN (I/Q Signal){' ' * (self.width-32)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.buffer) > 1:
            time_plot = self.draw_highres_plot(list(self.buffer), time_height - 1, 'green', fill=True)
            for line in time_plot:
                print(f"{self.colors['cyan']}║{self.colors['green']}{line}{' ' * (self.width - len(line) - 1)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(time_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # ═══ SPECTRUM ═══
        print(f"{self.colors['cyan']}║{self.colors['blue']} 📊 FREQUENCY SPECTRUM (FFT){' ' * (self.width-33)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.spectrum_buffer) > 1:
            spec_plot = self.draw_highres_plot(list(self.spectrum_buffer), spectrum_height - 1, 'blue', fill=True)
            for line in spec_plot:
                print(f"{self.colors['cyan']}║{self.colors['blue']}{line}{' ' * (self.width - len(line) - 1)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(spectrum_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # ═══ WATERFALL ═══
        print(f"{self.colors['cyan']}║{self.colors['magenta']} 🌊 WATERFALL DISPLAY{' ' * (self.width-26)}{self.colors['cyan']}║{self.colors['reset']}")
        
        if len(self.waterfall_buffer) > 1:
            waterfall_lines = self.draw_waterfall(waterfall_height - 1)
            for line in waterfall_lines:
                # line уже содержит ANSI коды
                visible_len = self.width  # Приблизительно
                print(f"{self.colors['cyan']}║{line}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(waterfall_height - 1):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        # ═══ FOOTER ═══
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        ctrl = "🎮 Controls: Ctrl+C = Stop | Terminal Size = Auto-adjust | Gain = Random Modulation"
        print(f"{self.colors['cyan']}║{self.colors['cyan']} {ctrl}{' ' * (self.width - len(ctrl) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        print(f"{self.colors['cyan']}╚{'═' * (self.width-2)}╝{self.colors['reset']}")

class CallbackBlock(gr.sync_block):
    def __init__(self, callback, sample_rate=2e6):
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

class pmj(gr.top_block):
    def __init__(self, options=None):
        gr.top_block.__init__(self, "Power-modulated Jammer")
        
        options_dict = vars(options) if options else {}
        self.samp_rate = options_dict.get("samp_rate", 5e6)
        self.f0 = options_dict.get("f0", 2437e6)
        self.bw = options_dict.get("bw", 5e6)
        self.signal_type = options_dict.get("signal_type", 3)
        self.visualize = options_dict.get("visualize", False)
        self.current_gain = 0

        # SDR Sink
        self.osmosdr_sink_0 = osmosdr.sink(args="numchan=1 antsdr=0")
        self.osmosdr_sink_0.set_sample_rate(self.samp_rate)
        self.osmosdr_sink_0.set_center_freq(self.f0, 0)
        self.osmosdr_sink_0.set_gain(14, 0)
        self.osmosdr_sink_0.set_if_gain(20, 0)
        self.osmosdr_sink_0.set_bb_gain(20, 0)
        self.osmosdr_sink_0.set_bandwidth(self.bw, 0)

        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate, True)

        self._create_source()

        if self.visualize:
            print("🎨 Initializing MEGA HD terminal visualizer...")
            self.visualizer = MegaVisualizer(
                sample_rate=self.samp_rate,
                center_freq=self.f0,
                signal_type=self.signal_type
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
            1: "📡 SINE WAVE", 2: "🛰 QPSK", 3: "🌪 NOISE",
            4: "📶 BPSK", 5: "🎯 8PSK", 6: "⚡ OOK", 7: "📻 FSK"
        }
        
        name = signal_names.get(self.signal_type, "❓ UNKNOWN")
        print(f"🎯 Signal Type: {name}")
        
        if self.signal_type == 1:
            self.source = analog.sig_source_c(self.samp_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        elif self.signal_type == 2:
            symbols = [1+1j, -1+1j, -1-1j, 1-1j] * 1000
            self.source = blocks.vector_source_c(symbols, True)
        elif self.signal_type == 3:
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)
        elif self.signal_type == 4:
            symbols = [-1, 1] * 1000
            complex_symbols = [complex(s, 0) for s in symbols]
            self.source = blocks.vector_source_c(complex_symbols, True)
        elif self.signal_type == 5:
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            symbols = [np.exp(1j * angle) for angle in angles] * 200
            self.source = blocks.vector_source_c(symbols, True)
        elif self.signal_type == 6:
            symbols = [0, 1] * 1000
            complex_symbols = [complex(s, 0) for s in symbols]
            self.source = blocks.vector_source_c(complex_symbols, True)
        elif self.signal_type == 7:
            t = np.linspace(0, 1, int(self.samp_rate/100))
            freq1, freq2 = 1000, 2000
            signal1 = np.exp(1j * 2 * np.pi * freq1 * t)
            signal2 = np.exp(1j * 2 * np.pi * freq2 * t)
            mixed_signal = []
            for i in range(100):
                mixed_signal.extend(signal1 if i % 2 == 0 else signal2)
            self.source = blocks.vector_source_c(mixed_signal, True)
        else:
            print("⚠️ Unknown signal type, using noise")
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)

    def pulse(self):
        print('🎯 Jammer is ACTIVE. Press Ctrl+C to stop.')
        try:
            while True:
                time.sleep(0.15)  # Плавное обновление
                gain = random.randint(0, 40)
                self.current_gain = gain
                self.osmosdr_sink_0.set_if_gain(gain, 0)
                
                if hasattr(self, 'visualizer'):
                    self.visualizer.stats['current_gain'] = gain
                    self.visualizer.display()
                else:
                    print(f"🎛 Gain: {gain:2d} dB | Freq: {self.f0/1e6:.2f} MHz", end='\r')
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down jammer...")

def main():
    parser = ArgumentParser(description="⚡ RF Power-Modulated Jammer with HD Visualization")
    parser.add_argument("--f0", type=float, default=2437e6, help="Center frequency (Hz)")
    parser.add_argument("--samp_rate", type=float, default=2e6, help="Sample rate (Hz)")
    parser.add_argument("--bw", type=float, default=5e6, help="Bandwidth (Hz)")
    parser.add_argument("--signal_type", type=int, default=3, choices=[1,2,3,4,5,6,7],
                       help="Signal: 1=Sine, 2=QPSK, 3=Noise, 4=BPSK, 5=8PSK, 6=OOK, 7=FSK")
    parser.add_argument("--visualize", action="store_true", help="Enable HD terminal visualization")
    
    args = parser.parse_args()

    print("=" * 60)
    print("⚡ RF JAMMER - Power Modulated Signal Generator")
    print("=" * 60)
    print(f"📡 Frequency: {args.f0/1e6:.2f} MHz")
    print(f"📊 Sample Rate: {args.samp_rate/1e6:.2f} MHz")
    print(f"📶 Bandwidth: {args.bw/1e6:.2f} MHz")
    print(f"🎨 Visualization: {'ENABLED' if args.visualize else 'DISABLED'}")
    print("=" * 60)

    tb = pmj(options=args)

    def sig_handler(sig=None, frame=None):
        print("\n🛑 Received stop signal, shutting down...")
        tb.stop()
        tb.wait()
        print("✅ Jammer stopped successfully.")
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("🚀 Starting RF jammer...")
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
