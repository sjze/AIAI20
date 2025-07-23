#Custom_messages.py
# ______________________________________
# AISG-AIAP20 submission file: Custom_messages.py
# by Unni Krishnan Ambady (Sxxxx664B)
# email: ambady1960@hotmail.com
# Submitted through private GitHub repository:
# https://github.com/UkAmbady/aiap20-Unni-Krishnan-Ambady-664B


import os
import logging
from datetime import datetime
import pytz

class LocalFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=pytz.timezone('Asia/Singapore')):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, tz=self.tz)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        # Format time without microseconds
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

class CustomMessages:
    def __init__(self, messages_dir=None):
        # Always point to root's Messages folder
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if messages_dir is None:
            messages_dir = os.path.join(root_dir, "artifacts", "logs")
            # Route all logs into artifacts/logs
        self.messages_dir = messages_dir
        self.message_file = os.path.join(self.messages_dir, "CustomMessages.txt")
        self._setup_directories()
        self._setup_logger()

    def _setup_directories(self):
        """Ensure the messages directory exists."""
        os.makedirs(self.messages_dir, exist_ok=True)

    def _setup_logger(self):
        """Set up logger to write to a single file with formatted messages."""
        self.logger = logging.getLogger('CustomMessagesLogger')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.message_file)

        formatter = LocalFormatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def rtm(self, input_string):
        """Log a run-time message."""
        print(f"Logging message: {input_string}")
        # Just realised that print works in GitHub Actions...

        try:
            # replace unicode arrow with ASCII
            safe_string = input_string.replace('→', '->')
            self.logger.info(safe_string)
        except Exception as e:
            print(f"Logging failed: {str(e)}")

if __name__ == "__main__":
    cm = CustomMessages()
    cm.rtm("Testing  CustomMessages module.")
