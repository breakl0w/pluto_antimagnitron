#!/usr/bin/env python3
import numpy as np
from gnuradio import gr, blocks, analog, uhd
from gnuradio.filter import firdes
from gnuradio import osmosdr
import time
import sys

class SDRRepeater(gr.top_block):
    def __init__(self, rtl_freq=100e6, rtl_sample_rate=2.4e6, 
                 antsdr_addr="addr=192.168.40.2", antsdr_freq=100e6, 
                 antsdr_sample_rate=2.4e6, rtl_gain=40, antsdr_gain=30):
        gr.top_block.__init__(self, "SDR Repeater")
        
        # Параметры
        self.rtl_freq = rtl_freq
        self.rtl_sample_rate = rtl_sample_rate
        self.antsdr_addr = antsdr_addr
        self.antsdr_freq = antsdr_freq
        self.antsdr_sample_rate = antsdr_sample_rate
        self.rtl_gain = rtl_gain
        self.antsdr_gain = antsdr_gain
        
        # Источник RTL-SDR
        print(f"Инициализация RTL-SDR на частоте {self.rtl_freq/1e6} МГц")
        self.rtl_source = osmosdr.source(args="num_channels=1")
        self.rtl_source.set_sample_rate(self.rtl_sample_rate)
        self.rtl_source.set_center_freq(self.rtl_freq)
        self.rtl_source.set_gain(self.rtl_gain)
        
        # Приемник ANTSdr (передатчик)
        print(f"Инициализация ANTSdr по адресу {self.antsdr_addr} на частоте {self.antsdr_freq/1e6} МГц")
        self.antsdr_sink = uhd.usrp_sink(
            device_addr=self.antsdr_addr,
            stream_args=uhd.stream_args("fc32", "sc16"),
            num_channels=1
        )
        
        # Настройка ANTSdr
        self.antsdr_sink.set_samp_rate(self.antsdr_sample_rate)
        self.antsdr_sink.set_center_freq(self.antsdr_freq, 0)
        self.antsdr_sink.set_gain(self.antsdr_gain, 0)
        
        # Создаем поток данных
        self.src = self.rtl_source
        self.sink = self.antsdr_sink
        
        # Соединяем блоки
        self.connect(self.src, self.sink)

if __name__ == '__main__':
    try:
        print("Запуск SDR Repeater...")
        # Создание и запуск потока
        tb = SDRRepeater()
        tb.start()
        print("SDR Repeater запущен. Нажмите Ctrl+C для остановки.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
        tb.stop()
        tb.wait()
        print("SDR Repeater остановлен.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)