from abc            import ABC, abstractmethod
from .DataModels    import Experience

class ILearner(ABC):
    @abstractmethod
    def Learn(self, e: list[Experience]):
        pass

class IActionTaker(ABC):
    @abstractmethod
    def NextAction(self, state: int):
        pass

class ITrainable(ABC):
    @abstractmethod
    def Train(self):
        pass

class ISavableToDisk(ABC):
    @abstractmethod
    def SaveToDisk(self, path : str):
        pass

    @abstractmethod
    def LoadFromDisk(self, path : str):
        pass