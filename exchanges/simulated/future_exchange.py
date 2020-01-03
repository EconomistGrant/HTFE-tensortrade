# Written by Songhao Li, HUATAI Securities
# Adopted from simulated_exchange.py
# Used on futures, _data_frame only include signal columns and _tag include price and for_trade


import numpy as np
import pandas as pd

from abc import abstractmethod
from gym.spaces import Space, Box
from typing import List, Dict, Generator

from tensortrade.trades import Trade, TradeType, FutureTradeType
from tensortrade.exchanges import InstrumentExchange
from tensortrade.slippage import RandomUniformSlippageModel

import datetime

class FutureExchange(InstrumentExchange):
    """An instrument exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(base_instrument=kwargs.get('base_instrument', 'USD'),
                         dtype=kwargs.get('dtype', np.float16),
                         feature_pipeline=kwargs.get('feature_pipeline', None))
        self._observe_position = kwargs.get('observe_position', False)
        self._should_pretransform_obs = kwargs.get('should_pretransform_obs', False)
        self._feature_pipeline = kwargs.get('feature_pipeline', None)
        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._instrument_precision = kwargs.get('instrument_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E4)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-3)
        self._window_size = kwargs.get('window_size', 1)
        self._min_trade_price = kwargs.get('min_trade_price', 1E-6)
        self._max_trade_price = kwargs.get('max_trade_price', 1E6)
        self._min_trade_amount = kwargs.get('min_trade_amount', 1E-3)
        self._max_trade_amount = kwargs.get('max_trade_amount', 1E6)
        self._feature_pipeline = kwargs.get('feature_pipeline', None)
        self._response_time = kwargs.get('response_time', 0)
        self._exclude_close = kwargs.get('exclude_close', False)
        self._active_holds = 0
        self._passive_holds = 0
        #self._episode_trades = pd.DataFrame([],columns = ['trades','active_holds','passive_holds'])

        max_allowed_slippage_percent = kwargs.get('max_allowed_slippage_percent', 1.0)

        SlippageModelClass = kwargs.get('slippage_model', RandomUniformSlippageModel)
        self._slippage_model = SlippageModelClass(max_allowed_slippage_percent)
        if data_frame is not None:
            self.data_frame = data_frame.astype(self._dtype)
            self.price = data_frame.astype(self._dtype)
    @property
    def price(self) -> pd.DataFrame:
        """The price from the original DataFrame."""
        return self._price
        
    @price.setter
    def price(self, data_frame: pd.DataFrame):
        self._price = data_frame[['close']]
    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return self._data_frame
    
    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        if self._exclude_close:
            self._data_frame = data_frame.drop(columns = ['close'])
        else:
            self._data_frame = data_frame
        
        if self._observe_position:
            self._data_frame['position'] = np.zeros(len(self._data_frame))
        else:
            pass
        
        if self._should_pretransform_obs and self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(
                self._data_frame, self.generated_space)
            print('DataFrame set: pipeline used')
        else:
            print('DataFrame set: pipeline unused')


    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def generated_space(self) -> Space:
        low = np.array([0]*self.data_frame.shape[1],dtype = 'float16')#np.array(self.data_frame.min() / 10000)
        high = np.array([np.inf]*self.data_frame.shape[1],dtype = 'float16')#np.array(self.data_frame.max() * 10000)
        return Box(low=low, high=high, dtype='float')

    @property
    def generated_columns(self) -> List[str]:
        return list(self._data_frame.columns)


    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _create_observation_generator(self) -> Generator[pd.DataFrame, None, None]:
        if self._window_size == 1:
            data = np.array(self._data_frame).T
            for step in range(self._current_step, data.shape[1]):
                self._current_step = step
                obs = np.zeros(data.shape[0],dtype = 'float16')
                for i in range(0, data.shape[0]):
                    obs[i] = data[i][self._current_step]
                obs = pd.DataFrame(obs).T
                obs.columns = self._data_frame.columns
                if not self._observe_position:
                    pass
                else:
                    obs['position'] = self._portfolio.get('BTC', 0)
                    #这个btc是默认的产品标识
                #print(obs)
                yield obs
            
        else:
            for step in range(self._current_step, len(self._data_frame)):
                self._current_step = step
                
                obs = self._data_frame.iloc[step - self._window_size + 1:step + 1]
    
                if not self._should_pretransform_obs and self._feature_pipeline is not None:
                    obs = self._feature_pipeline.transform(obs, self.generated_space)

                if self._observe_position:
                    raise NotImplementedError('Please implement this by the same logic as shown in windowsize == 1 ----Songhao')
    
                yield obs

        raise StopIteration

    def current_price(self, symbol: str) -> float:
        #if len(self._data_frame) is 0:
            #self.next_observation()
        return float(self.price['close'].values[self._current_step])

    def _is_valid_trade(self, trade: Trade) -> bool:

        #交易后的|open_amount| <= 1, 以及|open_value| < |net_worth|
        open_amount = self._portfolio.get(trade.symbol, 0)
        if trade.trade_type is TradeType.HOLD or trade.trade_type is FutureTradeType.HOLD:
            self._active_holds += 1
        elif trade.trade_type is TradeType.MARKET_BUY or trade.trade_type is TradeType.LIMIT_BUY or trade.trade_type is FutureTradeType.BUY:
            open_amount = self._portfolio.get(trade.symbol, 0) + trade.amount
        elif trade.trade_type is TradeType.MARKET_SELL or trade.trade_type is TradeType.LIMIT_SELL or trade.trade_type is FutureTradeType.SELL:
            open_amount = self._portfolio.get(trade.symbol, 0) - trade.amount

        if abs(open_amount) <= 1.001 and abs(open_amount * trade.price) < self.net_worth:
            return True
        else:
            self._passive_holds += 1
            return False


    def _update_account(self, trade: Trade):
        if trade.amount > 0:
            '''
            self._trades = self._trades.append({
                'step': self._current_step,
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'amount': trade.amount,
                'price': trade.price,
                'volume':trade.price * trade.amount
            }, ignore_index=True)
            '''
        if trade.is_buy:
            self._balance -= trade.amount * (1.0003) * trade.price
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.amount

        elif trade.is_sell:
            self._balance += trade.amount * (0.9997) * trade.price
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.amount

        self._portfolio[self._base_instrument] = self._balance
        
        step = self._current_step
        self._performance[0][step] = self.balance
        self._performance[1][step] = self.net_worth
        self._performance[2][step] = self._portfolio.get(trade.symbol, 0)
        #print(trade.symbol)
        self._performance[3][step] = trade.price

    def execute_trade(self, trade: Trade) -> Trade:
        current_price = self.current_price(symbol=trade.symbol)

        commission = self._commission_percent / 100

        filled_trade = trade.copy()
        
        if filled_trade.is_hold or not self._is_valid_trade(filled_trade):
            filled_trade.amount = 0
        '''
        elif filled_trade.is_buy:
            price_adjustment = price_adjustment = (1 + commission)
            filled_trade.price = max(round(current_price * price_adjustment,
                                           self._base_precision), self.base_precision)
            filled_trade.amount = round(
                (filled_trade.price * filled_trade.amount) / filled_trade.price, self._instrument_precision)
        elif filled_trade.is_sell:
            price_adjustment = (1 - commission)
            filled_trade.price = round(current_price * price_adjustment, self._base_precision)
            filled_trade.amount = round(filled_trade.amount, self._instrument_precision)
        '''
        filled_trade = self._slippage_model.fill_order(filled_trade, current_price)

        self._update_account(filled_trade)

        return filled_trade

    def reset(self):
        super().reset()

        self._balance = self._initial_balance
        self._portfolio = {self._base_instrument: self._balance}
        '''
        if hasattr(self, 'trades'):
            self._episode_trades = self._episode_trades.append({
                'trades':len(self._trades),
                'active_holds':self._active_holds,
                'passive_holds':self._passive_holds
            },ignore_index = True)
            print('trades:' + str(len(self._trades)))
            print(self._performance.tail())
        else:
            pass
        
        self._active_holds = 0
        self._passive_holds = 0
        '''
        if hasattr(self, '_performance'):
            performance =(pd.DataFrame(data = self._performance.T, columns = ['balance','net_worth','open_amount','price']))
            print(performance.tail(5))
        #self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price','volume'])
        self._performance = np.zeros([4, len(self._data_frame) - 1])
        # 1 = balance
        # 2 = net_worth
        # 3 = open_amount
        # 4 = price
        self._current_step = 0
