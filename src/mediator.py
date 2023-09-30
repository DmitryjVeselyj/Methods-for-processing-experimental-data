from abc import ABC, abstractmethod

class AbstractMediator(ABC):
    @abstractmethod
    def notify(self, sender, event):
        pass


class Mediator(AbstractMediator):
    def __init__(self, InOuter, Model, Analyzer, Processor):
        pass
    
    def notify(self, sender, event):
        pass


class BaseComponent:
    def __init__(self, mediator = None) -> None:
        self._mediator = mediator
    
    @property
    def mediator(self) -> Mediator:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator