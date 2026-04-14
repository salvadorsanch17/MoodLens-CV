"""
Quick test harness for the StressBreakDialog and BreathingOverlay.
Run:  python3.12 test_breathing.py
"""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog

# Re-use the real classes from the main GUI
from moodlens_gui import StressBreakDialog, BreathingOverlay


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    breathing = BreathingOverlay()

    def on_finished():
        print("✅ Breathing session finished!")
        app.quit()

    breathing.finished.connect(on_finished)

    # Show the stress-break dialog immediately
    dlg = StressBreakDialog()
    result = dlg.exec_()

    if result == QDialog.Accepted:
        print("User accepted — launching breathing overlay...")
        breathing.start()
        sys.exit(app.exec_())
    else:
        print("User declined — exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
