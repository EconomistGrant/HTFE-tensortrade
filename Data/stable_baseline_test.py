import os
import sys
import warnings
import numpy
import pandas as pd

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
numpy.seterr(divide = 'ignore') 

from tensortrade.strategies import StableBaselinesTradingStrategy

'''
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
reward_strategy = SimpleProfitStrategy()
action_strategy = DiscreteActionStrategy(n_actions=20, instrument_symbol='HS')
'''

from tensortrade.rewards import DirectProfitStrategy
from tensortrade.actions import FutureActionStrategy
reward_strategy = DirectProfitStrategy()
action_strategy = FutureActionStrategy()


from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.exchanges.simulated.simulated_exchange import SimulatedExchange
from tensortrade.exchanges.simulated.future_exchange import FutureExchange

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])
#处理信号,normalize+difference



data = pd.read_csv('RB.csv',index_col = 0)
data = data.tail(100)

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = False)


from tensortrade.environments import TradingEnvironment

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy
params = {"learning_rate": 0.001,'nminibatches':1}

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

strategy = StableBaselinesTradingStrategy(environment=environment, model = model, policy = policy, model_kwargs = params)

performance = strategy.run(episodes=5)

