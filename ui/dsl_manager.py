from abc import ABC, abstractmethod


class DslManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def trigger_command(self, command, path):
        pass
