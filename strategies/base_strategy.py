# strategies/base_strategy.py

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signals(self, df):
        """Return DataFrame with strategy-specific signals"""
        pass

    @abstractmethod
    def score_stock(self, df):
        """Return numeric score for ranking stocks"""
        pass
