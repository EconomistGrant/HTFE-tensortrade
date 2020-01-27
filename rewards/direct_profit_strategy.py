#Written by Songhao Li, HUATAI Securities
import os
import sys
ttpath = os.path.abspath('..')
sys.path.append(ttpath)

import pandas as pd
import numpy as np

from tensortrade.rewards import RewardStrategy
from tensortrade.trades import TradeType, Trade, FutureTradeType


class DirectProfitStrategy(RewardStrategy):
    '''This reward = how much money that the strategy earns'''
    def reset(self):
        """Necessary to reset the open amount and the last price"""
        self._open_amount= 0
        self._last_price = 0

    def get_reward(self, current_step: int, trade: Trade) -> float:
        last_price = self._last_price
        price = trade.price
        last_amount = self._open_amount

        #reset values
        if trade.is_hold:
            pass
        elif trade.is_buy:
            self._open_amount += trade.amount
        elif trade.is_sell:
            self._open_amount -= trade.amount

        last_price = self._last_price

        self._last_price = trade.price        
        
        if trade.is_hold:
            return last_amount * (price - last_price)
        else:
            return last_amount * (price - last_price) - trade.amount * trade.price * 0.0003