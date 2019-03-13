import sys
from ..logger import Logger
from ..logger import Severity


class ConsoleLogger(Logger):
    """Console logger for turret engine.

    Attribute:
        threshold(turret.Severity): The threshold of logging.
    """
    def __init__(self, threshold=Severity.INFO):
        self.threshold = threshold

    def log(self, severity, message):
        """Logging message.

        Args:
            severity(turret.Severity): The logging level.
            message(str): The message for logging.
        """
        if severity <= self.threshold:
            sys.stderr.write("[{}] {}\n".format(severity, message))
            sys.stderr.flush()
