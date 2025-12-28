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

class SpectrumMonitor:
    """Мониторинг спектра через RTL-SDR"""
    def __init__(self, center_freq=2437e6, sample_rate=2.4e6, scan_width=20e6):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.scan_width = scan_width
        
        # Частоты для сканирования
        self.scan_freqs = []
        self.current_scan_idx = 0
        
        # Детектированные цели
        self.detected_targets = {}  # {freq: {'power': dBm, 'last_seen': time, 'count': int}}
        self.detection_threshold = -60  # dBm
        self.lock = threading.Lock()
        
        # Буферы для визуализации
        self.spectrum_data = {}
        self.waterfall_buffer = deque(maxlen=50)
        
        # Статистика
        self.stats = {
            'scans_completed': 0,
            'targets_detected': 0,
            'current_freq': center_freq,
            'strongest_signal': -100,
            'strongest_freq': 0
        }
        
    def setup_scan_range(self, start_freq, stop_freq, step):
        """Настройка диапазона сканирования"""
        self.scan_freqs = list(np.arange(start_freq, stop_freq, step))
        print(f"📡 Scan range: {start_freq/1e6:.1f} - {stop_freq/1e6:.1f} MHz")
        print(f"📊 Steps: {len(self.scan_freqs)} x {step/1e6:.1f} MHz")
    
    def process_spectrum(self, data, current_freq):
        """Обработка спектральных данных"""
        if len(data) < 512:
            return
        
        # FFT
        fft_data = np.fft.fftshift(np.fft.fft(data[:512]))
        fft_magnitude = np.abs(fft_data)
        fft_db = 20 * np.log10(fft_magnitude + 1e-10)
        
        # Частотная ось
        freq_axis = np.fft.fftshift(np.fft.fftfreq(512, 1/self.sample_rate))
        freq_axis += current_freq
        
        # Поиск пиков
        threshold = self.detection_threshold
        peaks_idx = np.where(fft_db > threshold)[0]
        
        with self.lock:
            for idx in peaks_idx:
                freq = freq_axis[idx]
                power = fft_db[idx]
                
                # Обновляем или добавляем цель
                if freq in self.detected_targets:
                    self.detected_targets[freq]['power'] = power
                    self.detected_targets[freq]['last_seen'] = time.time()
                    self.detected_targets[freq]['count'] += 1
                else:
                    self.detected_targets[freq] = {
                        'power': power,
                        'last_seen': time.time(),
                        'count': 1,
                        'first_seen': time.time()
                    }
                    self.stats['targets_detected'] += 1
            
            # Обновляем статистику
            if len(fft_db) > 0:
                max_power = np.max(fft_db)
                if max_power > self.stats['strongest_signal']:
                    self.stats['strongest_signal'] = max_power
                    self.stats['strongest_freq'] = freq_axis[np.argmax(fft_db)]
            
            # Сохраняем спектр для визуализации
            self.spectrum_data[current_freq] = fft_db
            self.waterfall_buffer.append(fft_db)
    
    def get_active_targets(self, timeout=5.0):
        """Получить активные цели (видели недавно)"""
        current_time = time.time()
        with self.lock:
            active = {
                freq: data for freq, data in self.detected_targets.items()
                if current_time - data['last_seen'] < timeout
            }
        return active
    
    def get_top_targets(self, n=5):
        """Получить топ N целей по мощности"""
        active = self.get_active_targets()
        sorted_targets = sorted(
            active.items(),
            key=lambda x: x[1]['power'],
            reverse=True
        )
        return sorted_targets[:n]

class RTLSDRScanner(gr.top_block):
    """GNU Radio flowgraph для RTL-SDR"""
    def __init__(self, monitor, center_freq=2437e6, sample_rate=2.4e6):
        gr.top_block.__init__(self, "RTL-SDR Scanner")
        
        self.monitor = monitor
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        
        # RTL-SDR Source
        self.rtlsdr_source = osmosdr.source(args="numchan=1 rtl=0")
        self.rtlsdr_source.set_sample_rate(sample_rate)
        self.rtlsdr_source.set_center_freq(center_freq, 0)
        self.rtlsdr_source.set_freq_corr(0, 0)
        self.rtlsdr_source.set_gain(40, 0)  # Максимальный gain для RTL-SDR
        self.rtlsdr_source.set_if_gain(20, 0)
        self.rtlsdr_source.set_bb_gain(20, 0)
        self.rtlsdr_source.set_antenna('', 0)
        self.rtlsdr_source.set_bandwidth(0, 0)
        
        # Callback block
        self.callback_block = CallbackBlock(
            lambda data: self.monitor.process_spectrum(data, self.center_freq),
            sample_rate
        )
        
        # Подключение
        self.connect(self.rtlsdr_source, self.callback_block)
        self.connect(self.callback_block, blocks.null_sink(gr.sizeof_gr_complex))
    
    def set_frequency(self, freq):
        """Изменить частоту сканирования"""
        self.center_freq = freq
        self.rtlsdr_source.set_center_freq(freq, 0)
        self.monitor.stats['current_freq'] = freq

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

class ANTSDRJammer(gr.top_block):
    """GNU Radio flowgraph для ANTSDR E200"""
    def __init__(self, center_freq=2437e6, sample_rate=5e6, signal_type=3):
        gr.top_block.__init__(self, "ANTSDR Jammer")
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.signal_type = signal_type
        self.is_jamming = False
        
        # ANTSDR E200 Sink
        self.antsdr_sink = osmosdr.sink(args="numchan=1 antsdr=0")
        self.antsdr_sink.set_sample_rate(sample_rate)
        self.antsdr_sink.set_center_freq(center_freq, 0)
        self.antsdr_sink.set_gain(40, 0)  # RF gain
        self.antsdr_sink.set_if_gain(30, 0)
        self.antsdr_sink.set_bb_gain(30, 0)
        self.antsdr_sink.set_bandwidth(sample_rate, 0)
        
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, sample_rate, True)
        
        # Создаем источник сигнала
        self._create_source(signal_type)
        
        # Подключение
        self.connect(self.source, self.throttle, self.antsdr_sink)
    
    def _create_source(self, signal_type):
        """Создание источника сигнала"""
        if signal_type == 1:  # Sine
            self.source = analog.sig_source_c(self.sample_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        elif signal_type == 2:  # QPSK
            symbols = [1+1j, -1+1j, -1-1j, 1-1j] * 1000
            self.source = blocks.vector_source_c(symbols, True)
        elif signal_type == 3:  # Noise (лучше для джамминга)
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)
        elif signal_type == 4:  # BPSK
            symbols = [-1, 1] * 1000
            complex_symbols = [complex(s, 0) for s in symbols]
            self.source = blocks.vector_source_c(complex_symbols, True)
        elif signal_type == 5:  # Chirp (развертка)
            # Создаем chirp сигнал
            t = np.linspace(0, 1, int(self.sample_rate))
            f0, f1 = -self.sample_rate/4, self.sample_rate/4
            chirp = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
            self.source = blocks.vector_source_c(chirp.tolist(), True)
        else:
            self.source = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)
    
    def set_frequency(self, freq):
        """Изменить частоту джамминга"""
        self.center_freq = freq
        self.antsdr_sink.set_center_freq(freq, 0)
    
    def set_power(self, gain):
        """Изменить мощность (0-40)"""
        self.antsdr_sink.set_if_gain(gain, 0)
    
    def start_jamming(self):
        """Начать джамминг"""
        if not self.is_jamming:
            self.start()
            self.is_jamming = True
    
    def stop_jamming(self):
        """Остановить джамминг"""
        if self.is_jamming:
            self.stop()
            self.wait()
            self.is_jamming = False

class AdaptiveJammerSystem:
    """Главная система: детектор + джаммер"""
    def __init__(self, options):
        self.options = options
        
        # Монитор спектра
        self.monitor = SpectrumMonitor(
            center_freq=options.scan_center,
            sample_rate=options.scan_rate,
            scan_width=options.scan_width
        )
        self.monitor.detection_threshold = options.threshold
        
        # Настройка диапазона сканирования
        self.monitor.setup_scan_range(
            options.scan_start,
            options.scan_stop,
            options.scan_step
        )
        
        # RTL-SDR сканер
        print("🔍 Initializing RTL-SDR scanner...")
        self.scanner = RTLSDRScanner(
            self.monitor,
            center_freq=options.scan_center,
            sample_rate=options.scan_rate
        )
        
        # ANTSDR джаммер
        print("⚡ Initializing ANTSDR E200 jammer...")
        self.jammer = ANTSDRJammer(
            center_freq=options.jam_freq,
            sample_rate=options.jam_rate,
            signal_type=options.signal_type
        )
        
        # Режимы работы
        self.mode = options.mode  # 'scan', 'jam', 'auto'
        self.running = True
        
        # Визуализация
        self.visualizer = SystemVisualizer(self.monitor, self.jammer, options)
        
        # Потоки
        self.scan_thread = None
        self.jam_thread = None
        self.display_thread = None
    
    def start(self):
        """Запуск системы"""
        print("🚀 Starting Adaptive Jammer System...")
        
        # Запускаем сканер
        self.scanner.start()
        print("✅ RTL-SDR scanner started")
        
        # Запускаем потоки
        if self.mode in ['scan', 'auto']:
            self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
            self.scan_thread.start()
            print("✅ Scan thread started")
        
        if self.mode in ['jam', 'auto']:
            self.jam_thread = threading.Thread(target=self._jam_loop, daemon=True)
            self.jam_thread.start()
            print("✅ Jam thread started")
        
        # Визуализация
        if self.options.visualize:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
            print("✅ Display thread started")
        
        print("=" * 60)
        print("🎯 System is ACTIVE")
        print("=" * 60)
    
    def _scan_loop(self):
        """Цикл сканирования частот"""
        while self.running:
            for freq in self.monitor.scan_freqs:
                if not self.running:
                    break
                
                # Переключаем частоту
                self.scanner.set_frequency(freq)
                
                # Ждем накопления данных
                time.sleep(self.options.dwell_time)
                
                self.monitor.stats['scans_completed'] += 1
            
            # Небольшая пауза между циклами
            time.sleep(0.1)
    
    def _jam_loop(self):
        """Цикл джамминга"""
        while self.running:
            if self.mode == 'auto':
                # Автоматический режим: глушим найденные цели
                targets = self.monitor.get_top_targets(n=1)
                
                if targets:
                    target_freq, target_data = targets[0]
                    
                    # Переключаемся на цель
                    self.jammer.set_frequency(target_freq)
                    
                    # Модуляция мощности
                    gain = random.randint(20, 40)
                    self.jammer.set_power(gain)
                    
                    # Запускаем джамминг если еще не запущен
                    if not self.jammer.is_jamming:
                        self.jammer.start_jamming()
                    
                    time.sleep(0.2)
                else:
                    # Нет целей - останавливаем джамминг
                    if self.jammer.is_jamming:
                        self.jammer.stop_jamming()
                    time.sleep(0.5)
            
            elif self.mode == 'jam':
                # Ручной режим: глушим заданную частоту
                if not self.jammer.is_jamming:
                    self.jammer.start_jamming()
                
                gain = random.randint(20, 40)
                self.jammer.set_power(gain)
                time.sleep(0.2)
            
            else:
                time.sleep(0.5)
    
    def _display_loop(self):
        """Цикл отображения"""
        while self.running:
            self.visualizer.display()
            time.sleep(0.2)
    
    def stop(self):
        """Остановка системы"""
        print("\n🛑 Stopping system...")
        self.running = False
        
        if self.scanner:
            self.scanner.stop()
            self.scanner.wait()
        
        if self.jammer and self.jammer.is_jamming:
            self.jammer.stop_jamming()
        
        print("✅ System stopped")

class SystemVisualizer:
    """Визуализация всей системы"""
    def __init__(self, monitor, jammer, options):
        self.monitor = monitor
        self.jammer = jammer
        self.options = options
        
        self.width, self.height = shutil.get_terminal_size()
        self.width = min(self.width, 200)
        self.height = min(self.height, 60)
        
        self.colors = {
            'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
            'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
            'white': '\033[97m', 'reset': '\033[0m', 'gray': '\033[90m',
            'bright_red': '\033[91m\033[1m', 'bright_green': '\033[92m\033[1m'
        }
    
    def display(self):
        """Отображение статуса системы"""
        print("\033[2J\033[H", end="")  # Очистка экрана
        
        # Заголовок
        print(f"{self.colors['cyan']}╔{'═' * (self.width-2)}╗{self.colors['reset']}")
        title = f"⚡ ADAPTIVE RF JAMMER SYSTEM - RTL-SDR + ANTSDR E200 ⚡"
        padding = (self.width - len(title) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['bright_green']}{' ' * padding}{title}{' ' * (self.width - len(title) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mode_str = f"Mode: {self.options.mode.upper()}"
        info = f"🕐 {timestamp} | {mode_str}"
        padding = (self.width - len(info) - 2) // 2
        print(f"{self.colors['cyan']}║{self.colors['white']}{' ' * padding}{info}{' ' * (self.width - len(info) - padding - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        
        # RTL-SDR Scanner Status
        print(f"{self.colors['cyan']}║{self.colors['green']} 🔍 RTL-SDR SCANNER STATUS{' ' * (self.width-30)}{self.colors['cyan']}║{self.colors['reset']}")
        
        scan_freq = self.monitor.stats['current_freq'] / 1e6
        scans = self.monitor.stats['scans_completed']
        targets = self.monitor.stats['targets_detected']
        strongest = self.monitor.stats['strongest_signal']
        strongest_freq = self.monitor.stats['strongest_freq'] / 1e6
        
        line1 = f"📡 Current: {scan_freq:.2f} MHz | Scans: {scans} | Targets: {targets}"
        print(f"{self.colors['cyan']}║{self.colors['white']} {line1}{' ' * (self.width - len(line1) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        line2 = f"📊 Strongest: {strongest:.1f} dBm @ {strongest_freq:.2f} MHz | Threshold: {self.monitor.detection_threshold} dBm"
        print(f"{self.colors['cyan']}║{self.colors['white']} {line2}{' ' * (self.width - len(line2) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # Detected Targets
        print(f"{self.colors['cyan']}║{self.colors['yellow']} 🎯 DETECTED TARGETS (Active){' ' * (self.width-33)}{self.colors['cyan']}║{self.colors['reset']}")
        
        top_targets = self.monitor.get_top_targets(n=10)
        if top_targets:
            for i, (freq, data) in enumerate(top_targets[:5], 1):
                age = time.time() - data['last_seen']
                line = f"  {i}. {freq/1e6:8.2f} MHz | {data['power']:6.1f} dBm | Seen: {data['count']:4d}x | Age: {age:.1f}s"
                color = self.colors['bright_red'] if i == 1 else self.colors['yellow']
                print(f"{self.colors['cyan']}║{color}{line}{' ' * (self.width - len(line) - 1)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            line = "  No targets detected"
            print(f"{self.colors['cyan']}║{self.colors['gray']} {line}{' ' * (self.width - len(line) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        # Заполняем пустые строки
        for _ in range(5 - min(5, len(top_targets))):
            print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # ANTSDR Jammer Status
        print(f"{self.colors['cyan']}║{self.colors['red']} ⚡ ANTSDR E200 JAMMER STATUS{' ' * (self.width-35)}{self.colors['cyan']}║{self.colors['reset']}")
        
        jam_freq = self.jammer.center_freq / 1e6
        jam_status = "ACTIVE" if self.jammer.is_jamming else "IDLE"
        jam_color = self.colors['bright_red'] if self.jammer.is_jamming else self.colors['gray']
        
        line3 = f"📡 Frequency: {jam_freq:.2f} MHz | Status: {jam_status}"
        print(f"{self.colors['cyan']}║{jam_color} {line3}{' ' * (self.width - len(line3) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        line4 = f"📊 Sample Rate: {self.jammer.sample_rate/1e6:.2f} MHz | Signal Type: {self.options.signal_type}"
        print(f"{self.colors['cyan']}║{self.colors['white']} {line4}{' ' * (self.width - len(line4) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}╠{'─' * (self.width-2)}╣{self.colors['reset']}")
        
        # Waterfall (если есть данные)
        print(f"{self.colors['cyan']}║{self.colors['magenta']} 🌊 SPECTRUM WATERFALL{' ' * (self.width-27)}{self.colors['cyan']}║{self.colors['reset']}")
        
        waterfall_height = 10
        if len(self.monitor.waterfall_buffer) > 0:
            waterfall_chars = [' ', '░', '▒', '▓', '█']
            for i in range(min(waterfall_height, len(self.monitor.waterfall_buffer))):
                data = self.monitor.waterfall_buffer[-(i+1)]
                line = ""
                
                # Ресемплинг для ширины экрана
                step = max(1, len(data) // (self.width - 4))
                for j in range(0, len(data), step):
                    if j // step >= self.width - 4:
                        break
                    val = data[j]
                    # Нормализация
                    normalized = (val - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
                    char_idx = int(normalized * (len(waterfall_chars) - 1))
                    char_idx = min(char_idx, len(waterfall_chars) - 1)
                    line += waterfall_chars[char_idx]
                
                print(f"{self.colors['cyan']}║ {line}{' ' * (self.width - len(line) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        else:
            for _ in range(waterfall_height):
                print(f"{self.colors['cyan']}║{' ' * (self.width-2)}║{self.colors['reset']}")
        
        # Footer
        print(f"{self.colors['cyan']}╠{'═' * (self.width-2)}╣{self.colors['reset']}")
        ctrl = "🎮 Controls: Ctrl+C = Stop | Mode: scan/jam/auto"
        print(f"{self.colors['cyan']}║{self.colors['cyan']} {ctrl}{' ' * (self.width - len(ctrl) - 2)}{self.colors['cyan']}║{self.colors['reset']}")
        print(f"{self.colors['cyan']}╚{'═' * (self.width-2)}╝{self.colors['reset']}")

def main():
    parser = ArgumentParser(description="⚡ Adaptive RF Jammer: RTL-SDR Scanner + ANTSDR E200 Jammer")
    
    # Режим работы
    parser.add_argument("--mode", type=str, default="auto", choices=['scan', 'jam', 'auto'],
                       help="Mode: scan (only scan), jam (only jam), auto (scan+jam)")
    
    # RTL-SDR Scanner settings
    parser.add_argument("--scan_start", type=float, default=2400e6, help="Scan start frequency (Hz)")
    parser.add_argument("--scan_stop", type=float, default=2500e6, help="Scan stop frequency (Hz)")
    parser.add_argument("--scan_step", type=float, default=5e6, help="Scan step (Hz)")
    parser.add_argument("--scan_center", type=float, default=2437e6, help="Initial scan center (Hz)")
    parser.add_argument("--scan_rate", type=float, default=2.4e6, help="RTL-SDR sample rate (Hz)")
    parser.add_argument("--scan_width", type=float, default=20e6, help="Scan width (Hz)")
    parser.add_argument("--dwell_time", type=float, default=0.1, help="Dwell time per frequency (s)")
    parser.add_argument("--threshold", type=float, default=-60, help="Detection threshold (dBm)")
    
    # ANTSDR Jammer settings
    parser.add_argument("--jam_freq", type=float, default=2437e6, help="Initial jam frequency (Hz)")
    parser.add_argument("--jam_rate", type=float, default=5e6, help="ANTSDR sample rate (Hz)")
    parser.add_argument("--signal_type", type=int, default=3, choices=[1,2,3,4,5],
                       help="Signal: 1=Sine, 2=QPSK, 3=Noise, 4=BPSK, 5=Chirp")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Enable terminal visualization")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("⚡ ADAPTIVE RF JAMMER SYSTEM")
    print("=" * 80)
    print(f"🔍 RTL-SDR: Scanning {args.scan_start/1e6:.1f} - {args.scan_stop/1e6:.1f} MHz")
    print(f"⚡ ANTSDR E200: Jamming @ {args.jam_freq/1e6:.1f} MHz")
    print(f"🎯 Mode: {args.mode.upper()}")
    print(f"📊 Threshold: {args.threshold} dBm")
    print("=" * 80)
    
    # Создаем систему
    system = AdaptiveJammerSystem(args)
    
    def sig_handler(sig=None, frame=None):
        print("\n🛑 Received stop signal...")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    try:
        system.start()
        
        # Главный цикл
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()
        print("✅ Shutdown complete")

if __name__ == '__main__':
    main()
