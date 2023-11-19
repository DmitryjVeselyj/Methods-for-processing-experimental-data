from src.handler import AbstractHandler
from src.mediator import BaseComponent


class InOuter(AbstractHandler, BaseComponent):
    def handle(self, data):
        pass

