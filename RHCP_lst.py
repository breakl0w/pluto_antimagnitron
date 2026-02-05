#!/usr/bin/env python3
"""
RHCP Transmitter для ADALM Pluto Nano
Версия с поддержкой osmocom (gr-osmosdr)

Установка:
    sudo apt-get install gnuradio gr-osmosdr
    pip install numpy matplotlib
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
    from gnuradio import gr, blocks, analog
    import osmosdr
    OSMO_AVAILABLE = True
    print("✓ gr-osmosdr найден")
except ImportError as e:
    OSMO_AVAILABLE = False
    print(f"⚠ gr-osmosdr не найден: {e}")
    print("  Установи: sudo apt-get install gr-osmosdr")


class ToneController:
    """Контроллер тона с модуляцией"""
    
    def __init__(self, base_freq=10e3, samp_rate=2e6):
        self.base_freq = base_freq
        self.samp_rate = samp_rate
        self.current_freq = base_freq
        
        self.mod_enabled = False
        self.mod_type = 'none'
        self.mod_rate = 1.0
        self.mod_depth = 5e3
        self.sweep_min = 1e3
        self.sweep_max = 20e3
        self.chirp_duration = 1.0
        
        self._start_time = time.time()
        
    def get_frequency(self):
        if not self.mod_enabled:
            return self.base_freq
            
        t = time.time() - self._start_time
        
        if self.mod_type == 'sweep':
            period = 1.0 / self.mod_rate
            phase = (t % period) / period
            if int(t / period) % 2 == 0:
                return self.sweep_min + (self.sweep_max - self.sweep_min) * phase
            else:
                return self.sweep_max - (self.sweep_max - self.sweep_min) * phase
                
        elif self.mod_type == 'lfo':
            mod = np.sin(2 * np.pi * self.mod_rate * t)
            return self.base_freq + self.mod_depth * mod
            
        elif self.mod_type == 'chirp':
            phase = (t % self.chirp_duration) / self.chirp_duration
            return self.sweep_min + (self.sweep_max - self.sweep_min) * phase
            
        elif self.mod_type == 'random':
            if np.random.random() < 0.02:
                self.current_freq = np.random.uniform(self.sweep_min, self.sweep_max)
            return self.current_freq
            
        elif self.mod_type == 'step':
            steps = [1e3, 5e3, 10e3, 15e3, 20e3]
            return steps[int(t * self.mod_rate) % len(steps)]
            
        return self.base_freq
    
    def set_modulation(self, mod_type, **kwargs):
        self.mod_type = mod_type
        self.mod_enabled = mod_type != 'none'
        self._start_time = time.time()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        print(f"Modulation: {mod_type}")


class OsmocomRHCPTransmitter(gr.top_block):
    """RHCP TX через gr-osmosdr — стабильно работает!"""
    
    def __init__(self, uri="usb:3.7.5", freq=433.92e6, samp_rate=2e6,
                 tone_freq=10e3, gain=60, polarization='RHCP'):
        gr.top_block.__init__(self, "RHCP TX osmocom")
        
        self.uri = uri
        self.freq = freq
        self.samp_rate = int(samp_rate)
        self.gain = gain
        self.polarization = polarization
        self.tone_ctrl = ToneController(tone_freq, samp_rate)
        
        self.running = False
        self.connected = False
        
        self.sample_queue = queue.Queue(maxsize=50)
        
        self.stats = {
            'tx_samples': 0,
            'start_time': None
        }
        
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Строим flowgraph с osmocom sink"""
        
        print(f"Подключение к PlutoSDR: {self.uri}")
        
        try:
            # 1. Источник — комплексная экспонента для RHCP/LHCP
            # RHCP: I ведёт Q на 90° → e^(+jωt)
            # LHCP: Q ведёт I на 90° → e^(-jωt)
            
            # Для RHCP: cos + j*sin
            # Для LHCP: cos - j*sin (инвертируем Q)
            
            self.source_i = analog.sig_source_f(
                self.samp_rate,
                analog.GR_COS_WAVE,
                self.tone_ctrl.base_freq,
                1.0, 0
            )
            
            self.source_q = analog.sig_source_f(
                self.samp_rate,
                analog.GR_SIN_WAVE,
                self.tone_ctrl.base_freq,
                1.0 if self.polarization == 'RHCP' else -1.0,  # Инверсия для LHCP
                0
            )
            
            # 2. Float to Complex
            self.f2c = blocks.float_to_complex(1)
            
            # 3. Усиление/амплитуда
            self.multiply = blocks.multiply_const_cc(0.8)
            
            # 4. osmocom sink
            device_args = f"plutosdr,uri={self.uri}"
            print(f"Device args: {device_args}")
            
            self.sink = osmosdr.sink(args=device_args)
            self.sink.set_sample_rate(self.samp_rate)
            self.sink.set_center_freq(self.freq, 0)
            self.sink.set_gain(self.gain, 0)
            self.sink.set_bandwidth(self.samp_rate, 0)
            
            # Проверяем подключение
            actual_freq = self.sink.get_center_freq()
            if actual_freq > 0:
                self.connected = True
                print(f"✓ Подключено!")
                print(f"  Freq: {actual_freq/1e6:.3f} MHz")
                print(f"  Gain: {self.sink.get_gain()} dB")
                print(f"  Sample Rate: {self.sink.get_sample_rate()/1e6:.1f} MSPS")
            else:
                print("⚠ Подключение не удалось")
                self.connected = False
                
            # 5. Соединения
            self.connect(self.source_i, (self.f2c, 0))
            self.connect(self.source_q, (self.f2c, 1))
            self.connect(self.f2c, self.multiply)
            self.connect(self.multiply, self.sink)
            
            print("✓ Flowgraph построен")
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            self.connected = False
            
    def start(self):
        """Запуск TX"""
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Запускаем flowgraph
        super().start()
        
        # Поток обновления частоты
        self._freq_thread = threading.Thread(target=self._frequency_updater, daemon=True)
        self._freq_thread.start()
        
        # Поток для визуализации
        self._viz_thread = threading.Thread(target=self._viz_updater, daemon=True)
        self._viz_thread.start()
        
        print("▶ TX запущен")
        
    def stop(self):
        """Остановка"""
        self.running = False
        time.sleep(0.1)
        super().stop()
        super().wait()
        print("■ TX остановлен")
        
    def _frequency_updater(self):
        """Обновление частоты тона"""
        last_freq = self.tone_ctrl.base_freq
        
        while self.running:
            try:
                new_freq = self.tone_ctrl.get_frequency()
                
                if abs(new_freq - last_freq) > 10:
                    self.source_i.set_frequency(new_freq)
                    self.source_q.set_frequency(new_freq)
                    last_freq = new_freq
                    
                time.sleep(0.02)
            except:
                time.sleep(0.1)
                
    def _viz_updater(self):
        """Генерация данных для визуализации"""
        while self.running:
            try:
                freq = self.tone_ctrl.get_frequency()
                t = np.arange(200) / self.samp_rate
                
                if self.polarization == 'RHCP':
                    samples = 0.8 * np.exp(+1j * 2 * np.pi * freq * t)
                else:
                    samples = 0.8 * np.exp(-1j * 2 * np.pi * freq * t)
                
                if not self.sample_queue.full():
                    self.sample_queue.put(samples)
                    
                self.stats['tx_samples'] += 200
                
            except:
                pass
            time.sleep(0.05)
            
    def get_samples(self, n=200):
        try:
            return self.sample_queue.get_nowait()
        except queue.Empty:
            freq = self.tone_ctrl.get_frequency()
            t = np.arange(n) / self.samp_rate
            sign = +1 if self.polarization == 'RHCP' else -1
            return 0.8 * np.exp(sign * 1j * 2 * np.pi * freq * t)
            
    # === Управление ===
    
    def set_frequency(self, freq_hz):
        """RF частота"""
        self.freq = freq_hz
        if self.connected:
            self.sink.set_center_freq(freq_hz, 0)
        print(f"RF: {freq_hz/1e6:.3f} MHz")
        
    def set_tone_frequency(self, tone_hz):
        """Частота тона"""
        self.tone_ctrl.base_freq = tone_hz
        self.source_i.set_frequency(tone_hz)
        self.source_q.set_frequency(tone_hz)
        print(f"Tone: {tone_hz/1e3:.2f} kHz")
        
    def set_gain(self, gain_db):
        """Усиление TX (0-89)"""
        gain_db = max(0, min(89, gain_db))
        self.gain = gain_db
        if self.connected:
            self.sink.set_gain(gain_db, 0)
        print(f"Gain: {gain_db} dB")
        
    def set_polarization(self, pol):
        """RHCP или LHCP"""
        self.polarization = pol.upper()
        # Меняем знак Q компоненты
        q_amp = 1.0 if self.polarization == 'RHCP' else -1.0
        self.source_q.set_amplitude(q_amp)
        print(f"Polarization: {self.polarization} {'↻' if pol == 'RHCP' else '↺'}")
        
    def get_info(self):
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        return {
            'connected': self.connected,
            'uri': self.uri,
            'freq_mhz': self.freq / 1e6,
            'tone_khz': self.tone_ctrl.get_frequency() / 1e3,
            'gain': self.gain,
            'samp_rate_msps': self.samp_rate / 1e6,
            'modulation': self.tone_ctrl.mod_type,
            'polarization': self.polarization,
            'runtime': runtime,
            'tx_samples': self.stats['tx_samples']
        }


class TransmitterGUI:
    """GUI с визуализацией"""
    
    def __init__(self, tx):
        self.tx = tx
        self.setup_figure()
        self.setup_controls()
        
    def setup_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Pluto RHCP TX (gr-osmosdr)', fontsize=14, fontweight='bold')
        
        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                                   left=0.05, right=0.95, top=0.92, bottom=0.25)
        
        # Осциллограф
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
        
        # Констелляция
        self.ax_const = self.fig.add_subplot(gs[1, 0])
        circle = plt.Circle((0, 0), 0.8, fill=False, ls='--', alpha=0.3)
        self.ax_const.add_artist(circle)
        self.line_const, = self.ax_const.plot([], [], 'g-', alpha=0.5, lw=1)
        self.scatter_const = self.ax_const.scatter([], [], c='red', s=50, zorder=5)
        self.ax_const.set_title('Circular Polarization')
        self.ax_const.set_xlabel('I')
        self.ax_const.set_ylabel('Q')
        self.ax_const.grid(True, alpha=0.3)
        self.ax_const.axis('equal')
        self.ax_const.set_xlim(-1.1, 1.1)
        self.ax_const.set_ylim(-1.1, 1.1)
        
        # Полярная
        self.ax_polar = self.fig.add_subplot(gs[1, 1], projection='polar')
        self.line_polar, = self.ax_polar.plot([], [], 'b-', alpha=0.5, lw=1)
        self.scatter_polar = self.ax_polar.scatter([], [], c='orange', s=30, zorder=5)
        self.ax_polar.set_title('Phase', pad=15)
        self.ax_polar.set_theta_zero_location('E')
        self.ax_polar.set_theta_direction(-1)
        self.ax_polar.set_ylim(0, 1.0)
        
        # Вектор
        self.ax_vec = self.fig.add_subplot(gs[1, 2])
        self.quiver = self.ax_vec.quiver(0, 0, 0, 0, angles='xy', scale_units='xy',
                                          scale=1, color='red', width=0.02)
        circle2 = plt.Circle((0, 0), 0.8, fill=False, ls='--', alpha=0.3)
        self.ax_vec.add_artist(circle2)
        self.ax_vec.set_title('Phasor')
        self.ax_vec.grid(True, alpha=0.3)
        self.ax_vec.axis('equal')
        self.ax_vec.set_xlim(-1.1, 1.1)
        self.ax_vec.set_ylim(-1.1, 1.1)
        
        # Инфо
        self.ax_info = self.fig.add_subplot(gs[1, 3])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=9, verticalalignment='top',
                                           fontfamily='monospace',
                                           bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        # История частоты
        self.ax_freq = self.fig.add_subplot(gs[2, :])
        self.line_freq, = self.ax_freq.plot([], [], 'green', alpha=0.8, lw=1.5)
        self.ax_freq.set_title('Tone Frequency')
        self.ax_freq.set_xlabel('Time (s)')
        self.ax_freq.set_ylabel('Frequency (kHz)')
        self.ax_freq.grid(True, alpha=0.3)
        
        self.freq_history = []
        self.time_history = []
        self.start_time = time.time()
        
    def setup_controls(self):
        ctrl_bottom = 0.02
        h = 0.03
        
        # Тон
        ax = self.fig.add_axes([0.1, ctrl_bottom + 0.12, 0.35, h])
        self.slider_tone = Slider(ax, 'Tone (kHz)', 0.1, 50,
                                  valinit=self.tx.tone_ctrl.base_freq/1e3)
        self.slider_tone.on_changed(lambda v: self.tx.set_tone_frequency(v * 1e3))
        
        # Gain
        ax = self.fig.add_axes([0.1, ctrl_bottom + 0.06, 0.35, h])
        self.slider_gain = Slider(ax, 'Gain (dB)', 0, 89, valinit=self.tx.gain)
        self.slider_gain.on_changed(lambda v: self.tx.set_gain(v))
        
        # RF
        ax = self.fig.add_axes([0.1, ctrl_bottom, 0.35, h])
        self.slider_rf = Slider(ax, 'RF (MHz)', 70, 6000, valinit=self.tx.freq/1e6)
        self.slider_rf.on_changed(lambda v: self.tx.set_frequency(v * 1e6))
        
        # Модуляция
        ax = self.fig.add_axes([0.55, ctrl_bottom, 0.12, 0.15])
        self.radio_mod = RadioButtons(ax, ('None', 'Sweep', 'LFO', 'Chirp', 'Random', 'Step'))
        self.radio_mod.on_clicked(self._on_mod)
        
        # Rate
        ax = self.fig.add_axes([0.7, ctrl_bottom + 0.06, 0.2, h])
        self.slider_rate = Slider(ax, 'Mod Rate', 0.1, 10, valinit=1.0)
        self.slider_rate.on_changed(lambda v: setattr(self.tx.tone_ctrl, 'mod_rate', v))
        
        # Depth
        ax = self.fig.add_axes([0.7, ctrl_bottom, 0.2, h])
        self.slider_depth = Slider(ax, 'Depth (kHz)', 1, 20, valinit=5)
        self.slider_depth.on_changed(lambda v: setattr(self.tx.tone_ctrl, 'mod_depth', v*1e3))
        
    def _on_mod(self, label):
        mod_map = {'None': 'none', 'Sweep': 'sweep', 'LFO': 'lfo',
                   'Chirp': 'chirp', 'Random': 'random', 'Step': 'step'}
        self.tx.tone_ctrl.set_modulation(mod_map.get(label, 'none'))
        
    def animate(self, frame):
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
            pwr = 20 * np.log10(np.abs(fft) + 1e-10)
            self.line_spec.set_data(freqs/1e3, pwr)
            
        # История
        t = time.time() - self.start_time
        f = self.tx.tone_ctrl.get_frequency() / 1e3
        self.time_history.append(t)
        self.freq_history.append(f)
        
        while self.time_history and self.time_history[0] < t - 10:
            self.time_history.pop(0)
            self.freq_history.pop(0)
            
        self.line_freq.set_data(self.time_history, self.freq_history)
        self.ax_freq.set_xlim(max(0, t - 10), t + 0.5)
        
        if self.freq_history:
            fmin, fmax = min(self.freq_history), max(self.freq_history)
            margin = max(1, (fmax - fmin) * 0.1)
            self.ax_freq.set_ylim(fmin - margin, fmax + margin)
        
        # Инфо
        info = self.tx.get_info()
        status = "🟢 TX" if info['connected'] else "🔴 DISCONNECTED"
        
        text = (
            f"Status: {status}\n"
            f"URI: {info['uri']}\n"
            f"───────────────\n"
            f"RF: {info['freq_mhz']:.3f} MHz\n"
            f"Tone: {info['tone_khz']:.2f} kHz\n"
            f"Gain: {info['gain']:.0f} dB\n"
            f"Rate: {info['samp_rate_msps']:.1f} MSPS\n"
            f"───────────────\n"
            f"Mod: {info['modulation']}\n"
            f"Pol: {info['polarization']} {'↻' if info['polarization']=='RHCP' else '↺'}\n"
            f"───────────────\n"
            f"Time: {info['runtime']:.1f}s\n"
            f"TX: {info['tx_samples']:,}"
        )
        self.info_text.set_text(text)
        
        return []
        
    def run(self):
        self.ani = FuncAnimation(self.fig, self.animate, interval=50, blit=False)
        
        def on_key(e):
            if e.key == 'q':
                plt.close()
            elif e.key == 'up':
                new = self.tx.freq + 1e6
                self.tx.set_frequency(new)
                self.slider_rf.set_val(new / 1e6)
            elif e.key == 'down':
                new = self.tx.freq - 1e6
                self.tx.set_frequency(new)
                self.slider_rf.set_val(new / 1e6)
            elif e.key == 'p':
                new_pol = 'LHCP' if self.tx.polarization == 'RHCP' else 'RHCP'
                self.tx.set_polarization(new_pol)
            elif e.key == 'left':
                new = max(100, self.tx.tone_ctrl.base_freq - 1e3)
                self.tx.set_tone_frequency(new)
                self.slider_tone.set_val(new / 1e3)
            elif e.key == 'right':
                new = self.tx.tone_ctrl.base_freq + 1e3
                self.tx.set_tone_frequency(new)
                self.slider_tone.set_val(new / 1e3)
                
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()


def main():
    print("=" * 60)
    print("  RHCP Transmitter — gr-osmosdr edition")
    print("=" * 60)
    
    if not OSMO_AVAILABLE:
        print("\n❌ Установи gr-osmosdr:")
        print("   sudo apt-get install gr-osmosdr")
        return
        
    print("\nПараметры (Enter = default):")
    
    try:
        uri = input("  URI [usb:3.7.5]: ") or "usb:3.7.5"
        freq = float(input("  RF MHz [433.92]: ") or "433.92")
        tone = float(input("  Tone kHz [10]: ") or "10")
        gain = float(input("  Gain dB [60]: ") or "60")
        pol = input("  Polarization [RHCP/LHCP]: ").upper() or "RHCP"
        if pol not in ['RHCP', 'LHCP']:
            pol = 'RHCP'
    except:
        uri, freq, tone, gain, pol = "usb:3.7.5", 433.92, 10, 60, "RHCP"
        
    # Создаём TX
    tx = OsmocomRHCPTransmitter(
        uri=uri,
        freq=freq * 1e6,
        samp_rate=2e6,
        tone_freq=tone * 1e3,
        gain=gain,
        polarization=pol
    )
    
    if not tx.connected:
        print("\n❌ Не удалось подключиться к SDR")
        return
    
    # Запуск
    tx.start()
    time.sleep(0.5)
    
    print("\n" + "=" * 40)
    print("Клавиши:")
    print("  ↑↓    — RF ±1 MHz")
    print("  ←→    — Tone ±1 kHz")
    print("  p     — Toggle RHCP/LHCP")
    print("  q     — Выход")
    print("=" * 40)
    
    # GUI
    gui = TransmitterGUI(tx)
    
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        tx.stop()
        print("Done!")


if __name__ == '__main__':
    main()
