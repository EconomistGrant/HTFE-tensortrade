#Written by Songhao Li, HUATAI Securities
import os
import sys
ttpath = os.path.abspath('..')
sys.path.append(ttpath)

import pandas as pd
import numpy as np

from tensortrade.rewards import RewardStrategy
from tensortrade.trades import TradeType, Trade, FutureTradeType


class PositionReward(RewardStrategy):
    '''This reward = how much money that the strategy earns'''
    def reset(self):
        """Necessary to reset the open amount and the last price"""
        self.amount = 0

    def get_reward(self, current_step: int, trade: Trade) -> float:
        price = trade.price
        next_price = trade.next_price
        #reset values

        if trade.is_hold:
            pass
        elif trade.is_buy:
            self.amount += trade.amount
        elif trade.is_sell:
            self.amount -= trade.amount
        profit = (next_price-price) * self.amount - trade.amount * price * 0.0003 
        assert self.amount <= 1.001 and self.amount >= -1.001
        return profit