# jammer_2500_core.py

from gnuradio import gr, analog
import osmosdr

class JammerApp(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Wideband Jammer 2.5 GHz")

        self.freq = 2500e6           # Центральная частота (2500 МГц)
        self.samp_rate = 10e6        # Частота дискретизации
        self.gain = -10              # Усиление передатчика
        self.bandwidth = 20e6        # Ширина полосы помехи

        # Источник сигнала: белый шум (широкополосная помеха)
        self.source = analog.sig_source_c(self.samp_rate, analog.GR_CONST_WAVE, 0, 1, 0)

        # SDR передатчик (PlutoSDR)
        self.sink = osmosdr.sink(args="driver=plutosdr")
        self.sink.set_center_freq(self.freq)
        self.sink.set_sample_rate(self.samp_rate)
        self.sink.set_gain_mode(False)  # Ручное управление усилением
        self.sink.set_gain(self.gain)
        self.sink.set_bandwidth(self.bandwidth)

        # Соединение
        self.connect(self.source, self.sink)

    def set_params(self, freq, samp_rate, gain, bandwidth):
        """Обновление параметров джаммера"""
        self.freq = freq * 1e6
        self.samp_rate = samp_rate * 1e6
        self.gain = gain
        self.bandwidth = bandwidth * 1e6

        self.sink.set_center_freq(self.freq)
        self.sink.set_sample_rate(self.samp_rate)
        self.sink.set_gain(self.gain)
        self.sink.set_bandwidth(self.bandwidth)

    def start_jamming(self):
        print(f"[INFO] Запуск джаммера на {self.freq / 1e6} МГц")
        print(f"          Sample Rate: {self.samp_rate / 1e6} MSPS")
        print(f"          Gain: {self.gain} дБ")
        print(f"          Bandwidth: {self.bandwidth / 1e6} МГц")
        self.start()

    def stop_jamming(self):
        print("[INFO] Остановка джаммера...")
        self.stop()
