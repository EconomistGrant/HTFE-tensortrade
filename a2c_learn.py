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
assert tensorflow.__version__ == '1.13.1' or tensorflow.__version__ == '1.14.1'
assert tensorforce.__version__ == '0.5.2'

#%% Actions & Rewards
from strategies import TensorforceTradingStrategy
from rewards import PositionReward
from actions import FuturePositionStrategy

reward_strategy = PositionReward()
action_strategy = FuturePositionStrategy(n_actions = 11)

#%% Feature Pipeline
from features.stationarity import FractionalDifference
from features.scalers import MinMaxNormalizer
from features import FeaturePipeline

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[])

#%% Data Input (Exchanges)
from exchanges.simulated.future_exchange_position import FutureExchangePosition
data = pd.read_csv('Data/TA.csv',index_col = 0)

data = data[data.index % 60 == 0]
data = data.reset_index(drop = True)


exchange = FutureExchangePosition(data, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 50000, observe_position = True)

#%% Agent Specification
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
    "max_episode_timesteps": 20000,#理解成在update之前至少多少个timestep
    "network": network_spec,
    "batch_size": 100,
    "update_frequency":100
}


#%%Environment Setup
from environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

strategy = TensorforceTradingStrategy(environment = environment, 
                                      agent_spec = agent_spec, save_best_agent = False)
#%%Start Over
performance = strategy.run(episodes=100, evaluation=False)

#manually store agent
#strategy.save_agent(directory = 'save/', filename = '01')

#%% Restore and Continue 
'''
strategy.restore_agent(directory = 'save/', filename = '01')
performance = strategy.run(episodes=(strategy._runner.agent.episodes + 20), evaluation=False)
'''
#%% Analysis
performance =(pd.DataFrame(data = self._performance.T, columns = ['balance','net_worth','open_amount','price']))
learning_curve = pd.Series(strategy._runner.episode_rewards)
learning_curve.plot()
