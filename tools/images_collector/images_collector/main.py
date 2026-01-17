"""
Images collector main entry point.
"""

import sys
from PyQt6.QtWidgets import QApplication
from images_collector.gui import MainWindow


def main():
    """Main entry point for images collector application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
