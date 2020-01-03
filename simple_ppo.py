import os
import sys
import warnings
import numpy
import pandas as pd
from shutil import copyfile
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
def warn(*args, **kwargs):
    pass
import matplotlib.pyplot as plt
warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
numpy.seterr(divide = 'ignore') 

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

#%% Data Input (Exchange)
from exchanges.simulated.future_exchange import FutureExchange
data = pd.read_csv('Data/TA.csv',index_col = 0)

data = data[data.index % 60 == 0]

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, observe_position = True)

#%% Agent Specification
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
    "update_frequency":10,
    "critic_network":network_spec,
    "critic_optimizer":'adam'
}

#%% Environment Setup
from environments import TradingEnvironment
environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)


strategy = TensorforceTradingStrategy(environment = environment, 
                                      agent_spec = agent_spec, save_best_agent = False)

#%% Simple Running
performance = strategy.run(episodes=1000, evaluation=False)

#%% Analysis
performance = pd.DataFrame(data = performance.T, columns = ('balance','net_worth',
                                                            'open_amount','price'))
learning_curve = pd.Series(strategy._runner.episode_rewards)
print('学习曲线:')
learning_curve.plot()
print('参数迭代次数:' + strategy.agent.updates)