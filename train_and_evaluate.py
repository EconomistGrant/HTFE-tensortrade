import os
import sys
import warnings
import numpy
import pandas as pd
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
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
data = pd.read_csv('Data/TAfun.csv',index_col = 0)

data = data[data.index % 60 == 0]
data = data.reset_index(drop = True)

length_in_sample = int(len(data) * 0.8)
length_out_sample = len(data) - length_in_sample
in_sample = data.head(length_in_sample)
out_sample = data.tail(length_out_sample).reset_index(drop = True)

exchange_in_sample = FutureExchange(in_sample, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

exchange_out_sample = FutureExchange(out_sample, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

#%%Environment Setup
from environments import TradingEnvironment

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

environment_in_sample = TradingEnvironment(exchange=exchange_in_sample,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

environment_out_sample = TradingEnvironment(exchange=exchange_out_sample,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

strategy_in_sample = TensorforceTradingStrategy(environment = environment_in_sample, 
                                      agent_spec = agent_spec, save_best_agent = False)

strategy_out_sample = TensorforceTradingStrategy(environment = environment_out_sample, 
                                      agent_spec = agent_spec, save_best_agent = False)

#%%Start 
total_episodes = 100
out_sample_start = 5
out_sample_frequency = 20 # should be multiples of update_frequency
out_sample_trials = 5

out_sample_rewards = [] #记录每次outsample的每回合rewards
out_sample_avrg_rewards = [] #记录每次outsample所有回合rewards平均值，判断收敛

performance = strategy_in_sample.run(episodes=out_sample_start, evaluation=False)
strategy_in_sample.save_agent(directory = 'save/temp/', filename = 'temp')
episode = strategy_in_sample.agent.episodes

while  episode < total_episodes:
    print('------in_sample------')
    performance = strategy_in_sample.run(episodes = (episode + out_sample_frequency), evaluation = False)
    strategy_in_sample.save_agent(directory = 'save/temp/', filename = 'temp')

    #runner数据会重制嘛？等会看最后一代episode数量就好
    print('------out_sample------')
    episode = strategy_in_sample.agent.episodes
    strategy_out_sample.restore_agent(directory = 'save/temp', filename = 'temp')
    out_sample_performance = strategy_out_sample.run(episodes=(episode + out_sample_trials), evaluation=False)
    avrg_reward = numpy.mean(strategy_out_sample._runner.episode_rewards[-out_sample_trials:])
    out_sample_rewards = out_sample_rewards + strategy_out_sample._runner.episode_rewards

in_sample_rewards = strategy_in_sample._runner.episode_rewards
#manually store agent
#strategy.save_agent(directory = 'save/', filename = '01')

#%% Restore and Continue 
'''
strategy.restore_agent(directory = 'save/', filename = 'best-model')
performance = strategy.run(episodes=(strategy._runner.agent.episodes + 20), evaluation=False)
'''

performance = pd.DataFrame(data = performance.T, columns = ('balance','net_worth',
                                                            'open_amount','price'))

