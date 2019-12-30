# Written by Songhao Li, Huatai Securities


import numpy as np
from typing import Union
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, FutureTradeType


class FutureActionStrategy(ActionStrategy):
    """Action strategy to be used on future markets. Amount is fixed to be one unit per trade for now."""

    def __init__(self, n_actions: int = 3, instrument_symbol: str = 'BTC', max_allowed_slippage_percent: float = 1.0):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'.
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an instrument. Defaults to 1.0 (i.e. 1%).
        """
        super().__init__(action_space=Discrete(n_actions), dtype=np.int64)
        #最后生成action数值的地方
        self.instrument_symbol = instrument_symbol
        self.max_allowed_slippage_percent = max_allowed_slippage_percent


    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `SimpleDiscreteStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """
        The trade type is determined by `action`, when there are only three types of trades:
        hold, buy and sell, implied by FutureTradeType.
        ACTION is determined, by default, discrete(3), and it should only include (0,1,2) 
        """
        trade_type = FutureTradeType(action)
        amount = 0.1
        current_price = self._exchange.current_price(symbol=self.instrument_symbol)

        price = current_price


        return Trade(self.instrument_symbol, trade_type, amount, price)
