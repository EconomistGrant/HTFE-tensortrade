import os
import sys
import warnings
import numpy
import pandas as pd

def warn(*args, **kwargs):
    pass
sys.path.append("../")
print(sys.path)
assert False
warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
numpy.seterr(divide = 'ignore') 
import tensorflow
assert tensorflow.__version__ == '1.13.1'

from tensortrade.strategies import TensorforceTradingStrategy
from tensortrade.rewards import DirectProfitStrategy
from tensortrade.actions import FutureActionStrategy
reward_strategy = DirectProfitStrategy()
action_strategy = FutureActionStrategy()

#%% Feature Pipeline
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[])

#%% Data Input
from tensortrade.exchanges.simulated.future_exchange import FutureExchange
data = pd.read_csv('TA.csv',index_col = 0)

data = data[data.index % 30 == 0]
data = data.reset_index(drop = True)

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

#%%Environment Setup
from tensortrade.environments import TradingEnvironment

network_spec = [
    dict(type='dense', size=128, activation="tanh"),
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

agent_spec = {
    "type": "ppo",
    "learning_rate": 0.0003,
    "discount": 1.0,
    "likelihood_ratio_clipping": 0.2,
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

performance = strategy.run(episodes=1000, evaluation=False)
#manually store agent
#strategy.save_agent(directory = 'test/', filename = '01')

#%% Restore and Continue 
'''
strategy.restore_agent(directory = 'a/', filename = 'best-model')
performance = strategy.run(episodes=(strategy._runner.agent.episodes + 20), evaluation=False)
'''
