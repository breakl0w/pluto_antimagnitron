# jammer_2500_gui.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from jammer_2500_core import JammerApp

class WidebandJammerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.jammer = JammerApp()

        self.setWindowTitle("Wideband Jammer 2.5 ГГц")

        layout = QVBoxLayout()

        # Поля ввода
        self.freq_input = QLineEdit("2500")
        self.samp_rate_input = QLineEdit("10")
        self.gain_input = QLineEdit("-10")
        self.bandwidth_input = QLineEdit("20")

        layout.addWidget(QLabel("Частота (МГц)"))
        layout.addWidget(self.freq_input)
        layout.addWidget(QLabel("Частота дискретизации (МГц)"))
        layout.addWidget(self.samp_rate_input)
        layout.addWidget(QLabel("Усиление (дБ)"))
        layout.addWidget(self.gain_input)
        layout.addWidget(QLabel("Полоса пропускания (МГц)"))
        layout.addWidget(self.bandwidth_input)

        # Кнопки
        self.start_button = QPushButton("Запустить джаммер")
        self.stop_button = QPushButton("Остановить джаммер")

        self.start_button.clicked.connect(self.start_jamming)
        self.stop_button.clicked.connect(self.stop_jamming)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def start_jamming(self):
        try:
            freq = float(self.freq_input.text())
            samp_rate = float(self.samp_rate_input.text())
            gain = float(self.gain_input.text())
            bandwidth = float(self.bandwidth_input.text())

            self.jammer.set_params(freq, samp_rate, gain, bandwidth)
            self.jammer.start_jamming()
        except Exception as e:
            print("[ERROR]", str(e))

    def stop_jamming(self):
        self.jammer.stop_jamming()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WidebandJammerGUI()
    window.show()
    sys.exit(app.exec_())
