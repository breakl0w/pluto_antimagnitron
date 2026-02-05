#!/usr/bin/env python3
"""
RHCP Transmitter для ADALM Pluto Nano
С контролем тона, модуляцией и визуализацией

Установка:
    sudo apt-get install gnuradio gr-iio libiio-dev iiod
    pip install matplotlib numpy

Проверка подключения:
    iio_info -s
    iio_attr -u ip:192.168.2.1 -d
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
import threading
import time
import sys
import queue

# GNU Radio импорты
try:
    from gnuradio import gr, blocks, analog, filter as gr_filter
    from gnuradio import iio
    GNURADIO_AVAILABLE = True
except ImportError:
    GNURADIO_AVAILABLE = False
    print("⚠ GNU Radio не найден — работаем в демо-режиме")


class ToneController:
    """Контроллер тона с разными режимами модуляции"""
    
    def __init__(self, base_freq=10e3, samp_rate=4e6):
        self.base_freq = base_freq
        self.samp_rate = samp_rate
        self.current_freq = base_freq
        
        # Параметры модуляции
        self.mod_enabled = False
        self.mod_type = 'none'  # none, sweep, lfo, chirp, random
        self.mod_rate = 1.0     # Hz
        self.mod_depth = 5e3    # Hz
        self.sweep_min = 1e3
        self.sweep_max = 20e3
        
        # Chirp параметры
        self.chirp_duration = 1.0  # секунды
        self.chirp_start = 1e3
        self.chirp_end = 20e3
        
        # Внутреннее состояние
        self._phase = 0
        self._start_time = time.time()
        
    def get_frequency(self):
        """Получить текущую частоту с учетом модуляции"""
        if not self.mod_enabled:
            return self.base_freq
            
        t = time.time() - self._start_time
        
        if self.mod_type == 'sweep':
            # Линейная развертка туда-сюда
            period = 1.0 / self.mod_rate
            phase = (t % period) / period
            if int(t / period) % 2 == 0:
                freq = self.sweep_min + (self.sweep_max - self.sweep_min) * phase
            else:
                freq = self.sweep_max - (self.sweep_max - self.sweep_min) * phase
            return freq
            
        elif self.mod_type == 'lfo':
            # Синусоидальная модуляция
            mod = np.sin(2 * np.pi * self.mod_rate * t)
            return self.base_freq + self.mod_depth * mod
            
        elif self.mod_type == 'chirp':
            # Линейный чирп с повтором
            phase = (t % self.chirp_duration) / self.chirp_duration
            return self.chirp_start + (self.chirp_end - self.chirp_start) * phase
            
        elif self.mod_type == 'random':
            # Случайные скачки
            if np.random.random() < 0.02:  # ~2% шанс смены
                self.current_freq = np.random.uniform(self.sweep_min, self.sweep_max)
            return self.current_freq
            
        elif self.mod_type == 'step':
            # Ступенчатое изменение
            steps = [1e3, 5e3, 10e3, 15e3, 20e3]
            idx = int(t * self.mod_rate) % len(steps)
            return steps[idx]
            
        return self.base_freq
    
    def set_modulation(self, mod_type, **kwargs):
        """Установить тип модуляции"""
        self.mod_type = mod_type
        self.mod_enabled = mod_type != 'none'
        self._start_time = time.time()
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        print(f"Модуляция: {mod_type}")
        if self.mod_enabled:
            print(f"  Rate: {self.mod_rate} Hz, Depth: {self.mod_depth/1e3:.1f} kHz")


class PlutoNanoRHCPTransmitter(gr.top_block if GNURADIO_AVAILABLE else object):
    """RHCP передатчик для Pluto Nano с динамическим контролем тона"""
    
    # URI варианты для подключения
    PLUTO_URIS = [
        "ip:192.168.2.1",      # USB Network (по умолчанию)
        "ip:192.168.3.1",      # Альтернативный IP
        "ip:pluto.local",      # mDNS
        "local:",              # Локальный iiod
    ]
    
    def __init__(self, freq=433.92e6, samp_rate=4e6, tone_freq=10e3, 
                 attenuation=10.0, uri=None):
        
        if GNURADIO_AVAILABLE:
            gr.top_block.__init__(self, "Pluto Nano RHCP TX")
        
        # Основные параметры
        self.freq = freq
        self.samp_rate = samp_rate
        self.attenuation = attenuation
        self.uri = uri
        
        # Контроллер тона
        self.tone_ctrl = ToneController(tone_freq, samp_rate)
        
        # Буфер для визуализации
        self.sample_queue = queue.Queue(maxsize=100)
        self.running = False
        self.connected = False
        
        # Статистика
        self.stats = {
            'tx_samples': 0,
            'underruns': 0,
            'start_time': None
        }
        
        if GNURADIO_AVAILABLE:
            self._build_flowgraph()
        
    def _find_pluto(self):
        """Автоматический поиск Pluto"""
        import subprocess
        
        print("Поиск Pluto Nano...")
        
        # Пробуем iio_info
        try:
            result = subprocess.run(['iio_info', '-s'], 
                                   capture_output=True, text=True, timeout=5)
            if 'PlutoSDR' in result.stdout or 'AD936' in result.stdout:
                # Парсим URI
                for line in result.stdout.split('\n'):
                    if 'usb:' in line or 'ip:' in line:
                        uri = line.split('[')[1].split(']')[0] if '[' in line else None
                        if uri:
                            print(f"✓ Найден Pluto: {uri}")
                            return uri
        except Exception as e:
            print(f"iio_info error: {e}")
        
        # Пробуем стандартные URI
        for uri in self.PLUTO_URIS:
            try:
                print(f"  Пробую {uri}...", end=' ')
                result = subprocess.run(['iio_attr', '-u', uri, '-d'], 
                                       capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    print("✓")
                    return uri
                print("✗")
            except:
                print("✗")
                
        return None
        
    def _build_flowgraph(self):
        """Построение flowgraph GNU Radio"""
        
        # Поиск устройства
        if self.uri is None:
            self.uri = self._find_pluto()
            
        if self.uri is None:
            print("\n⚠ Pluto не найден — используем NULL sink (демо)")
            self._build_demo_flowgraph()
            return
            
        print(f"\nПодключение к {self.uri}...")
        
        try:
            # 1. Источник сигнала — управляемый complex sinusoid
            # Это даст нам e^(j*w*t) = cos(wt) + j*sin(wt) = RHCP
            self.sig_source = analog.sig_source_c(
                self.samp_rate,
                analog.GR_COS_WAVE,
                self.tone_ctrl.base_freq,
                0.8,  # amplitude с headroom
                0     # phase offset
            )
            
            # 2. Множитель для контроля амплитуды
            self.amplitude = blocks.multiply_const_cc(1.0)
            
            # 3. Throttle для контроля потока
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate)
            
            # 4. Probe для мониторинга
            self.probe = blocks.probe_signal_c()
            
            # 5. Pluto Sink
            # Используем fmcomms2 для большего контроля
            try:
                self.sink = iio.fmcomms2_sink_fc32(
                    self.uri,
                    [True, False],  # TX1 enabled, TX2 disabled
                    32768,          # buffer size
                    True            # cyclic
                )
                self.sink.set_frequency(int(self.freq))
                self.sink.set_samplerate(int(self.samp_rate))
                self.sink.set_bandwidth(int(self.samp_rate * 1.2))
                self.sink.set_attenuation(0, self.attenuation)
                
                self.connected = True
                print(f"✓ Подключено к Pluto Nano")
                
            except Exception as e:
                print(f"fmcomms2 error: {e}")
                # Fallback на pluto_sink
                try:
                    self.sink = iio.pluto_sink(
                        self.uri,
                        int(self.freq),
                        int(self.samp_rate),
                        int(self.samp_rate * 1.2),
                        32768,
                        True,   # cyclic
                        self.attenuation
                    )
                    self.connected = True
                    print(f"✓ Подключено (pluto_sink)")
                except Exception as e2:
                    print(f"pluto_sink error: {e2}")
                    self._build_demo_flowgraph()
                    return
            
            # Соединения
            self.connect(self.sig_source, self.throttle)
            self.connect(self.throttle, self.amplitude)
            self.connect(self.amplitude, self.sink)
            self.connect(self.amplitude, self.probe)
            
            print(f"✓ Flowgraph построен")
            
        except Exception as e:
            print(f"Ошибка построения flowgraph: {e}")
            self._build_demo_flowgraph()
            
    def _build_demo_flowgraph(self):
        """Демо flowgraph без реального железа"""
        print("Запуск в демо-режиме...")
        
        self.sig_source = analog.sig_source_c(
            self.samp_rate,
            analog.GR_COS_WAVE,
            self.tone_ctrl.base_freq,
            0.8, 0
        )
        self.amplitude = blocks.multiply_const_cc(1.0)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate)
        self.probe = blocks.probe_signal_c()
        self.sink = blocks.null_sink(gr.sizeof_gr_complex)
        
        self.connect(self.sig_source, self.throttle)
        self.connect(self.throttle, self.amplitude)
        self.connect(self.amplitude, self.sink)
        self.connect(self.amplitude, self.probe)
        
        self.connected = False
        
    def start(self):
        """Запуск передачи"""
        self.running = True
        self.stats['start_time'] = time.time()
        
        if GNURADIO_AVAILABLE:
            super().start()
            
        # Поток обновления частоты
        self._freq_thread = threading.Thread(target=self._frequency_updater)
        self._freq_thread.daemon = True
        self._freq_thread.start()
        
        # Поток сбора samples
        self._sample_thread = threading.Thread(target=self._sample_collector)
        self._sample_thread.daemon = True
        self._sample_thread.start()
        
        print("▶ Передача запущена")
        
    def stop(self):
        """Остановка"""
        self.running = False
        time.sleep(0.1)
        
        if GNURADIO_AVAILABLE:
            super().stop()
            super().wait()
            
        print("■ Передача остановлена")
        
    def _frequency_updater(self):
        """Поток обновления частоты тона"""
        last_freq = self.tone_ctrl.base_freq
        
        while self.running:
            try:
                new_freq = self.tone_ctrl.get_frequency()
                
                # Обновляем только если изменилось
                if abs(new_freq - last_freq) > 10:  # 10 Hz threshold
                    self.sig_source.set_frequency(new_freq)
                    last_freq = new_freq
                    
                time.sleep(0.01)  # 100 Hz update rate
                
            except Exception as e:
                print(f"Freq update error: {e}")
                time.sleep(0.1)
                
    def _sample_collector(self):
        """Сбор samples для визуализации"""
        while self.running:
            try:
                if hasattr(self, 'probe'):
                    val = self.probe.level()
                    if val is not None:
                        # Генерируем немного samples на основе текущего состояния
                        freq = self.tone_ctrl.get_frequency()
                        t = np.linspace(0, 200/self.samp_rate, 200)
                        samples = 0.8 * np.exp(1j * 2 * np.pi * freq * t)
                        
                        if not self.sample_queue.full():
                            self.sample_queue.put(samples)
                            
                        self.stats['tx_samples'] += len(samples)
                        
            except Exception as e:
                pass
                
            time.sleep(0.05)
            
    def get_samples(self, n=200):
        """Получить samples для отображения"""
        try:
            return self.sample_queue.get_nowait()
        except queue.Empty:
            # Генерируем тестовые данные
            freq = self.tone_ctrl.get_frequency()
            t = np.linspace(0, n/self.samp_rate, n)
            return 0.8 * np.exp(1j * 2 * np.pi * freq * t)
            
    # === Управление параметрами ===
    
    def set_frequency(self, freq_hz):
        """Установить несущую частоту"""
        self.freq = freq_hz
        if self.connected and hasattr(self.sink, 'set_frequency'):
            self.sink.set_frequency(int(freq_hz))
        print(f"RF: {freq_hz/1e6:.3f} MHz")
        
    def set_tone_frequency(self, tone_hz):
        """Установить частоту тона"""
        self.tone_ctrl.base_freq = tone_hz
        if hasattr(self, 'sig_source'):
            self.sig_source.set_frequency(tone_hz)
        print(f"Tone: {tone_hz/1e3:.2f} kHz")
        
    def set_attenuation(self, atten_db):
        """Установить аттенюацию (0 = макс мощность, 89.75 = минимум)"""
        atten_db = max(0, min(89.75, atten_db))
        self.attenuation = atten_db
        
        if self.connected:
            if hasattr(self.sink, 'set_attenuation'):
                self.sink.set_attenuation(0, atten_db)
                
        print(f"Atten: {atten_db:.1f} dB")
        
    def set_amplitude(self, amp):
        """Установить амплитуду (0-1)"""
        amp = max(0, min(1, amp))
        if hasattr(self, 'amplitude'):
            self.amplitude.set_k(complex(amp))
        print(f"Amplitude: {amp:.2f}")
        
    def get_info(self):
        """Получить информацию о состоянии"""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'connected': self.connected,
            'uri': self.uri,
            'freq_mhz': self.freq / 1e6,
            'tone_khz': self.tone_ctrl.get_frequency() / 1e3,
            'attenuation': self.attenuation,
            'samp_rate_msps': self.samp_rate / 1e6,
            'modulation': self.tone_ctrl.mod_type,
            'runtime': runtime,
            'tx_samples': self.stats['tx_samples']
        }


class TransmitterGUI:
    """GUI с визуализацией и контролем"""
    
    def __init__(self, transmitter):
        self.tx = transmitter
        self.setup_figure()
        self.setup_controls()
        
    def setup_figure(self):
        """Настройка графиков"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Pluto Nano RHCP Transmitter', fontsize=14, fontweight='bold')
        
        # Сетка: 3 строки, 4 колонки
        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                                   left=0.05, right=0.95, top=0.92, bottom=0.25)
        
        # === Верхний ряд: осциллограф и спектр ===
        
        # I/Q осциллограф
        self.ax_osc = self.fig.add_subplot(gs[0, 0:2])
        self.line_i, = self.ax_osc.plot([], [], 'b-', label='I', alpha=0.8, lw=1)
        self.line_q, = self.ax_osc.plot([], [], 'r-', label='Q', alpha=0.8, lw=1)
        self.ax_osc.set_title('I/Q Waveform')
        self.ax_osc.set_xlabel('Samples')
        self.ax_osc.set_ylabel('Amplitude')
        self.ax_osc.legend(loc='upper right')
        self.ax_osc.grid(True, alpha=0.3)
        self.ax_osc.set_xlim(0, 200)
        self.ax_osc.set_ylim(-1.1, 1.1)
        
        # Спектр
        self.ax_spec = self.fig.add_subplot(gs[0, 2:4])
        self.line_spec, = self.ax_spec.plot([], [], 'purple', alpha=0.8, lw=1)
        self.ax_spec.set_title('Spectrum')
        self.ax_spec.set_xlabel('Frequency (kHz)')
        self.ax_spec.set_ylabel('Power (dB)')
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_spec.set_xlim(-50, 50)
        self.ax_spec.set_ylim(-60, 10)
        
        # === Средний ряд: констелляция, полярная, вектор ===
        
        # Констелляция (IQ plot)
        self.ax_const = self.fig.add_subplot(gs[1, 0])
        circle = plt.Circle((0, 0), 0.8, fill=False, ls='--', alpha=0.3, color='gray')
        self.ax_const.add_artist(circle)
        self.line_const, = self.ax_const.plot([], [], 'g-', alpha=0.5, lw=1)
        self.scatter_const = self.ax_const.scatter([], [], c='red', s=50, zorder=5)
        self.ax_const.set_title('RHCP Constellation')
        self.ax_const.set_xlabel('I')
        self.ax_const.set_ylabel('Q')
        self.ax_const.grid(True, alpha=0.3)
        self.ax_const.axis('equal')
        self.ax_const.set_xlim(-1.1, 1.1)
        self.ax_const.set_ylim(-1.1, 1.1)
        
        # Полярная диаграмма
        self.ax_polar = self.fig.add_subplot(gs[1, 1], projection='polar')
        self.line_polar, = self.ax_polar.plot([], [], 'b-', alpha=0.5, lw=1)
        self.scatter_polar = self.ax_polar.scatter([], [], c='orange', s=30, zorder=5)
        self.ax_polar.set_title('Phase Rotation', pad=15)
        self.ax_polar.set_theta_zero_location('E')
        self.ax_polar.set_theta_direction(-1)  # RHCP = clockwise
        self.ax_polar.set_ylim(0, 1.0)
        
        # Вектор
        self.ax_vec = self.fig.add_subplot(gs[1, 2])
        self.quiver = self.ax_vec.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', 
                                          scale=1, color='red', width=0.02,
                                          headwidth=4, headlength=5)
        circle2 = plt.Circle((0, 0), 0.8, fill=False, ls='--', alpha=0.3, color='gray')
        self.ax_vec.add_artist(circle2)
        self.ax_vec.set_title('Phasor')
        self.ax_vec.set_xlabel('Real')
        self.ax_vec.set_ylabel('Imag')
        self.ax_vec.grid(True, alpha=0.3)
        self.ax_vec.axis('equal')
        self.ax_vec.set_xlim(-1.1, 1.1)
        self.ax_vec.set_ylim(-1.1, 1.1)
        
        # Информационная панель
        self.ax_info = self.fig.add_subplot(gs[1, 3])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=9, verticalalignment='top',
                                           fontfamily='monospace',
                                           bbox=dict(boxstyle='round', facecolor='lightcyan', 
                                                    alpha=0.8))
        
        # === Нижний ряд: история частоты ===
        self.ax_freq = self.fig.add_subplot(gs[2, :])
        self.line_freq, = self.ax_freq.plot([], [], 'green', alpha=0.8, lw=1.5)
        self.ax_freq.set_title('Tone Frequency History')
        self.ax_freq.set_xlabel('Time (s)')
        self.ax_freq.set_ylabel('Frequency (kHz)')
        self.ax_freq.grid(True, alpha=0.3)
        self.ax_freq.set_xlim(0, 10)
        self.ax_freq.set_ylim(0, 25)
        
        # Буфер истории частоты
        self.freq_history = []
        self.time_history = []
        self.start_time = time.time()
        
    def setup_controls(self):
        """Настройка слайдеров и кнопок"""
        
        # Позиции контролов
        ctrl_bottom = 0.02
        ctrl_height = 0.03
        
        # Слайдер частоты тона
        ax_tone = self.fig.add_axes([0.1, ctrl_bottom + 0.12, 0.35, ctrl_height])
        self.slider_tone = Slider(ax_tone, 'Tone (kHz)', 0.1, 50, 
                                  valinit=self.tx.tone_ctrl.base_freq/1e3)
        self.slider_tone.on_changed(self._on_tone_change)
        
        # Слайдер аттенюации
        ax_atten = self.fig.add_axes([0.1, ctrl_bottom + 0.06, 0.35, ctrl_height])
        self.slider_atten = Slider(ax_atten, 'Atten (dB)', 0, 89, 
                                   valinit=self.tx.attenuation)
        self.slider_atten.on_changed(self._on_atten_change)
        
        # Слайдер RF частоты
        ax_rf = self.fig.add_axes([0.1, ctrl_bottom, 0.35, ctrl_height])
        self.slider_rf = Slider(ax_rf, 'RF (MHz)', 70, 6000, 
                                valinit=self.tx.freq/1e6)
        self.slider_rf.on_changed(self._on_rf_change)
        
        # Radio buttons для модуляции
        ax_mod = self.fig.add_axes([0.55, ctrl_bottom, 0.12, 0.15])
        self.radio_mod = RadioButtons(ax_mod, 
                                      ('None', 'Sweep', 'LFO', 'Chirp', 'Random', 'Step'),
                                      active=0)
        self.radio_mod.on_clicked(self._on_mod_change)
        
        # Слайдер скорости модуляции
        ax_rate = self.fig.add_axes([0.7, ctrl_bottom + 0.06, 0.2, ctrl_height])
        self.slider_rate = Slider(ax_rate, 'Mod Rate', 0.1, 10, valinit=1.0)
        self.slider_rate.on_changed(self._on_rate_change)
        
        # Слайдер глубины модуляции
        ax_depth = self.fig.add_axes([0.7, ctrl_bottom, 0.2, ctrl_height])
        self.slider_depth = Slider(ax_depth, 'Mod Depth (kHz)', 1, 20, valinit=5)
        self.slider_depth.on_changed(self._on_depth_change)
        
    def _on_tone_change(self, val):
        self.tx.set_tone_frequency(val * 1e3)
        
    def _on_atten_change(self, val):
        self.tx.set_attenuation(val)
        
    def _on_rf_change(self, val):
        self.tx.set_frequency(val * 1e6)
        
    def _on_mod_change(self, label):
        mod_map = {
            'None': 'none',
            'Sweep': 'sweep', 
            'LFO': 'lfo',
            'Chirp': 'chirp',
            'Random': 'random',
            'Step': 'step'
        }
        self.tx.tone_ctrl.set_modulation(mod_map.get(label, 'none'))
        
    def _on_rate_change(self, val):
        self.tx.tone_ctrl.mod_rate = val
        
    def _on_depth_change(self, val):
        self.tx.tone_ctrl.mod_depth = val * 1e3
        
    def animate(self, frame):
        """Функция анимации"""
        samples = self.tx.get_samples(200)
        
        if len(samples) < 10:
            return []
            
        i_data = samples.real
        q_data = samples.imag
        
        # Осциллограф
        x = np.arange(len(samples))
        self.line_i.set_data(x, i_data)
        self.line_q.set_data(x, q_data)
        
        # Констелляция
        self.line_const.set_data(i_data, q_data)
        self.scatter_const.set_offsets([[i_data[-1], q_data[-1]]])
        
        # Полярная
        angles = np.angle(samples)
        mags = np.abs(samples)
        self.line_polar.set_data(angles, mags)
        self.scatter_polar.set_offsets([[angles[-1], mags[-1]]])
        
        # Вектор
        self.quiver.set_UVC(i_data[-1], q_data[-1])
        
        # Спектр
        if len(samples) >= 64:
            fft = np.fft.fftshift(np.fft.fft(samples, 256))
            freqs = np.fft.fftshift(np.fft.fftfreq(256, 1/self.tx.samp_rate))
            power_db = 20 * np.log10(np.abs(fft) + 1e-10)
            self.line_spec.set_data(freqs/1e3, power_db)
            
        # История частоты
        current_time = time.time() - self.start_time
        current_freq = self.tx.tone_ctrl.get_frequency() / 1e3
        
        self.time_history.append(current_time)
        self.freq_history.append(current_freq)
        
        # Ограничиваем историю 10 секундами
        while self.time_history and self.time_history[0] < current_time - 10:
            self.time_history.pop(0)
            self.freq_history.pop(0)
            
        self.line_freq.set_data(self.time_history, self.freq_history)
        self.ax_freq.set_xlim(max(0, current_time - 10), current_time + 0.5)
        
        # Автомасштаб оси Y для частоты
        if self.freq_history:
            fmin, fmax = min(self.freq_history), max(self.freq_history)
            margin = max(1, (fmax - fmin) * 0.1)
            self.ax_freq.set_ylim(fmin - margin, fmax + margin)
        
        # Информация
        info = self.tx.get_info()
        status = "🟢 TX" if info['connected'] else "🟡 DEMO"
        
        info_str = (
            f"Status: {status}\n"
            f"URI: {info['uri'] or 'N/A'}\n"
            f"─────────────────\n"
            f"RF Freq: {info['freq_mhz']:.3f} MHz\n"
            f"Tone: {info['tone_khz']:.2f} kHz\n"
            f"Atten: {info['attenuation']:.1f} dB\n"
            f"Sample Rate: {info['samp_rate_msps']:.1f} MSPS\n"
            f"─────────────────\n"
            f"Modulation: {info['modulation']}\n"
            f"Polarization: RHCP\n"
            f"─────────────────\n"
            f"Runtime: {info['runtime']:.1f} s\n"
            f"TX Samples: {info['tx_samples']:,}"
        )
        self.info_text.set_text(info_str)
        
        return [self.line_i, self.line_q, self.line_const, self.scatter_const,
                self.line_polar, self.scatter_polar, self.quiver, self.line_spec,
                self.line_freq, self.info_text]
                
    def run(self):
        """Запуск GUI"""
        self.ani = FuncAnimation(self.fig, self.animate, interval=50, 
                                 blit=False, cache_frame_data=False)
        
        # Горячие клавиши
        def on_key(event):
            if event.key == 'q':
                plt.close()
            elif event.key == 'up':
                self.tx.set_frequency(self.tx.freq + 1e6)
                self.slider_rf.set_val(self.tx.freq / 1e6)
            elif event.key == 'down':
                self.tx.set_frequency(self.tx.freq - 1e6)
                self.slider_rf.set_val(self.tx.freq / 1e6)
            elif event.key == 'left':
                self.tx.set_tone_frequency(max(100, self.tx.tone_ctrl.base_freq - 1e3))
                self.slider_tone.set_val(self.tx.tone_ctrl.base_freq / 1e3)
            elif event.key == 'right':
                self.tx.set_tone_frequency(self.tx.tone_ctrl.base_freq + 1e3)
                self.slider_tone.set_val(self.tx.tone_ctrl.base_freq / 1e3)
                
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()


def main():
    """Главная функция"""
    print("=" * 60)
    print("  ADALM Pluto Nano — RHCP Transmitter")
    print("  With Tone Control & Modulation")
    print("=" * 60)
    
    # Параметры
    print("\nПараметры (Enter для значений по умолчанию):")
    
    try:
        freq_mhz = float(input("  RF частота (МГц) [433.92]: ") or "433.92")
        tone_khz = float(input("  Частота тона (кГц) [10]: ") or "10")
        atten_db = float(input("  Аттенюация (дБ, 0-89) [20]: ") or "20")
        uri = input("  URI Pluto (auto): ") or None
    except ValueError:
        print("Некорректный ввод, используем значения по умолчанию")
        freq_mhz, tone_khz, atten_db, uri = 433.92, 10, 20, None
    
    # Создаем передатчик
    if GNURADIO_AVAILABLE:
        tx = PlutoNanoRHCPTransmitter(
            freq=freq_mhz * 1e6,
            samp_rate=4e6,
            tone_freq=tone_khz * 1e3,
            attenuation=atten_db,
            uri=uri
        )
    else:
        print("\n⚠ GNU Radio недоступен")
        print("Для полной функциональности установите:")
        print("  sudo apt-get install gnuradio gr-iio")
        return
    
    # Запускаем
    tx.start()
    time.sleep(1)
    
    # GUI
    print("\nЗапуск интерфейса...")
    print("\nГорячие клавиши:")
    print("  ↑/↓ — RF частота ±1 МГц")
    print("  ←/→ — Тон ±1 кГц")
    print("  q   — Выход")
    
    gui = TransmitterGUI(tx)
    
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\nПрервано")
    finally:
        tx.stop()
        print("Готово!")


if __name__ == '__main__':
    main()
