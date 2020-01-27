import os
import sys
import warnings
import numpy
import pandas as pd
<<<<<<< HEAD

=======
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
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
<<<<<<< HEAD

=======
>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
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
<<<<<<< HEAD
data = pd.read_csv('Data/TA.csv',index_col = 0)
=======
<<<<<<< HEAD
data = pd.read_csv('Data/TA.csv',index_col = 0)

data = data[data.index % 30 == 0]
data = data.reset_index(drop = True)
data = data.tail(50)

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = True,
=======
data = pd.read_csv('Data/TAfun.csv',index_col = 0)
>>>>>>> 08330b7897a3ef2bf6720a6216fb2bdae1032e6c

data = data[data.index % 60 == 0]
data = data.reset_index(drop = True)

length_in_sample = int(len(data) * 0.8)
length_out_sample = len(data) - length_in_sample
in_sample = data.head(length_in_sample)
out_sample = data.tail(length_out_sample).reset_index(drop = True)

exchange = FutureExchange(data, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

exchange_in_sample = FutureExchange(in_sample, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, should_pretransform_obs = False)

exchange_out_sample = FutureExchange(out_sample, base_instrument = 'RMB', exclude_close = True,
>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
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
<<<<<<< HEAD
    "batch_size": 5,
    "update_frequency":5
}

environment = TradingEnvironment(exchange=exchange,
=======
    "batch_size": 10,
    "update_frequency":10,
    "critic_network":network_spec,
    "critic_optimizer":'adam'
}


environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

environment_in_sample = TradingEnvironment(exchange=exchange_in_sample,
>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

<<<<<<< HEAD
strategy = TensorforceTradingStrategy(environment = environment, 
                                      agent_spec = agent_spec, save_best_agent = False)
#%%Start Over
performance = strategy.run(episodes=50, evaluation=False)
=======
environment_out_sample = TradingEnvironment(exchange=exchange_out_sample,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)

strategy = TensorforceTradingStrategy(environment = environment, 
                                      agent_spec = agent_spec, save_best_agent = False)

strategy_in_sample = TensorforceTradingStrategy(environment = environment_in_sample, 
                                      agent_spec = agent_spec, save_best_agent = False)

strategy_out_sample = TensorforceTradingStrategy(environment = environment_out_sample, 
                                      agent_spec = agent_spec, save_best_agent = False)


#%% Start
'''
performance = strategy.run(episodes = 800, evaluation = False)
'''
#%%Start in|out sample 
total_episodes = 800
out_sample_start = 200
out_sample_frequency = 10 # should be multiples of update_frequency
out_sample_trials = 5


out_sample_rewards = [] #记录每次outsample的每回合rewards
out_sample_avrg_rewards = [] #记录每次outsample所有回合rewards平均值，判断收敛

performance = strategy_in_sample.run(episodes=out_sample_start, evaluation=False)
strategy_in_sample.save_agent(directory = 'save/temp/', filename = 'temp')
episode = strategy_in_sample.agent.episodes

num_out_sample = 1 # 用于存储标记计数
while episode < total_episodes:
    print('------in_sample------')
    performance = strategy_in_sample.run(episodes = (episode + out_sample_frequency), evaluation = False)
    strategy_in_sample.save_agent(directory = 'save/temp/', filename = 'temp')

    print('------out_sample------')
    episode = strategy_in_sample.agent.episodes
    strategy_out_sample.restore_agent(directory = 'save/temp', filename = 'temp')

<<<<<<< HEAD
    trial = 0
    rewards = []
    temp_ndarray = numpy.zeros(shape = (out_sample_trials, length_out_sample -1))
    
    while trial < out_sample_trials:
        out_sample_nav = strategy_out_sample.run(episodes=(episode + trial + 1), evaluation=False)[1]
        temp_ndarray[trial] = out_sample_nav
        reward = strategy_out_sample._runner.episode_rewards[-1]
        rewards.append(reward)
        out_sample_rewards.append(reward)
        trial += 1
        
    numpy.save('save/out_sample_npy/{}.npy'.format(num_out_sample),temp_ndarray)
    num_out_sample += 1
    
    avrg_reward = numpy.mean(rewards)
    out_sample_avrg_rewards.append(avrg_reward)


in_sample_rewards = pd.Series(strategy_in_sample._runner.episode_rewards)
=======
in_sample_rewards = strategy_in_sample._runner.episode_rewards
>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
>>>>>>> 08330b7897a3ef2bf6720a6216fb2bdae1032e6c
#manually store agent
#strategy.save_agent(directory = 'save/', filename = '01')

#%% Restore and Continue 

strategy.restore_agent(directory = 'save/', filename = 'best-model')
performance = strategy.run(episodes=(strategy._runner.agent.episodes + 20), evaluation=False)
<<<<<<< HEAD

=======
'''
<<<<<<< HEAD
=======
>>>>>>> 08330b7897a3ef2bf6720a6216fb2bdae1032e6c

performance = pd.DataFrame(data = performance.T, columns = ('balance','net_worth',
                                                            'open_amount','price'))

>>>>>>> 82fe9208809cb591dac8c7c57860b42dcaf11cf7
