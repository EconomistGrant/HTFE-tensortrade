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
#%% Module version check
import tensorforce
import tensorflow
assert tensorflow.__version__ == '1.13.1'
assert tensorforce.__version__ == '0.5.2'

#%% Actions & Rewards
from strategies import TensorforceTradingStrategy
from rewards import DirectProfitStrategy

from actions import FutureActionStrategy
reward_strategy = DirectProfitStrategy()
action_strategy = FutureActionStrategy()

#%% Feature Pipeline
from features.stationarity import FractionalDifference
from features.scalers import MinMaxNormalizer
from features import FeaturePipeline

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[])

#%% Data Input
from exchanges.simulated.future_exchange import FutureExchange
data = pd.read_csv('Data/TA.csv',index_col = 0)

data = data[data.index % 60 == 0]
data = data.reset_index(drop = True)

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

#%%Environment Setup
from environments import TradingEnvironment

network_spec = [
    dict(type='dense', size=128, activation="tanh"),
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

agent_spec = {
    "type": "a2c",
    "learning_rate": 0.0003,
    "discount": 1.0,
    "estimate_terminal": False,
    "max_episode_timesteps": 200000,
    "network": network_spec,
    "batch_size": 10,
    "update_frequency":10
}

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

strategy = TensorforceTradingStrategy(environment = environment, 
                                      agent_spec = agent_spec, save_best_agent = False)
#%%Start Over
performance = strategy.run(episodes=500, evaluation=False)
#manually store agent
#strategy.save_agent(directory = 'save/', filename = '01')

#%% Restore and Continue 
'''
strategy.restore_agent(directory = 'save/', filename = 'best-model')
performance = strategy.run(episodes=(strategy._runner.agent.episodes + 20), evaluation=False)
'''
