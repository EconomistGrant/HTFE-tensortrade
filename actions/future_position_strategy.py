# Written by Songhao Li, Huatai Securities
# 11/25/2019
import numpy as np
from typing import Union
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, FutureTradeType


class FuturePositionStrategy(ActionStrategy):
    """
    Action strategy to be used on future markets. 
    Genetic programming _timing implies the relationship between factor VALUE and POSITION (a state);
    future_action_strategy implies the relationship betwween factor VALUE and TRADE (a change).

    So an intuitively better way to do this is linking factor VALUE and POSITION again.

    WARNING:
    GP can maintain the position for a length of time if the factor is 'smooth' over time given the nature of _timing.
    If the factor is larger than the threshold (80 percentile) for 359 minutes, then it will hold 359 minutes.

    Even if the factor we put into RL is 'smooth' over time (which they should, since they are "selected" by GP),
    its unknown that how the 'black box' will handle the data and whether it will generate a continuing period with the same position.
    Besides, position (at-1) is not included in the observation to be handled by the 'blackbox',
    so there might be no way for the RL to learn that it should favor the previous position to lower transaction cost.

    So INTUITIVELY I recommend using mid- or low-frequency data when using this action strategy.
    I will make more tests and verify this argument.
    """

    def __init__(self, n_actions: int = 3, instrument_symbol: str = 'BTC', max_allowed_slippage_percent: float = 1.0):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'.
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an instrument. Defaults to 1.0 (i.e. 1%).
        """
        super().__init__(action_space=Discrete(n_actions), dtype=np.int64)
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
        0: SHORT
        1: NEUTRAL
        2: LONG
        """
        trade_type = FutureTradeType(action)
        amount = 1
        current_price = self._exchange.current_price(symbol=self.instrument_symbol)

        price = current_price


        return Trade(self.instrument_symbol, trade_type, amount, price)