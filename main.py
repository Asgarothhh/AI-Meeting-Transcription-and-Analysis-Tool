from src.app import ProtonApp
from PyQt6.QtWidgets import QApplication
import sys


def main():
    app = QApplication(sys.argv)
    window = ProtonApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
