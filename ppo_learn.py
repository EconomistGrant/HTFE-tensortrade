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

#%% Data Input (Exchanges)
from exchanges.simulated.future_exchange import FutureExchange
data = pd.read_csv('Data/TA.csv',index_col = 0)

data = data[data.index % 60 == 0]
data = data.reset_index(drop = True)

length_in_sample = int(len(data) * 0.9)
length_out_sample = len(data) - length_in_sample
in_sample = data.head(length_in_sample)
out_sample = data.tail(length_out_sample).reset_index(drop = True)

exchange_in_sample = FutureExchange(in_sample, base_instrument = 'RMB', exclude_close = True,
                          initial_balance = 10000, observe_position = True)

exchange_out_sample = FutureExchange(out_sample, base_instrument = 'RMB', exclude_close = True,
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

#%%Environment Setup
from environments import TradingEnvironment


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

#%% Start Running
total_episodes = 1000
out_sample_start = 200
out_sample_frequency = 10 # should be multiples of update_frequency
out_sample_trials = 5 # should be smaller than update_frequency

save_path = 'save/60,0.9,1000,0'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists('{}/out_sample_npy/'.format(save_path)):
    os.makedirs('{}/out_sample_npy/'.format(save_path))
if not os.path.exists('{}/out_sample_fig/'.format(save_path)):
    os.makedirs('{}/out_sample_fig/'.format(save_path))
if not os.path.exists('{}/in_sample_fig/'.format(save_path)):
    os.makedirs('{}/in_sample_fig/'.format(save_path))

out_sample_rewards = [] #记录每次outsample的每回合rewards
out_sample_avrg_rewards = [] #记录每次outsample所有回合rewards平均值，可以判断收敛

strategy_in_sample.save_agent(directory = save_path, filename = 'agent')
episode = strategy_in_sample.agent.episodes

num_out_sample = 1 # 用于存储标记计数
while episode < total_episodes:
    print('------in_sample------')
    performance = strategy_in_sample.run(episodes = (episode + out_sample_frequency), evaluation = False)
    fig = pd.DataFrame(performance[1].T).plot().get_figure()
    fig.savefig('{}/in_sample_fig/{}.png'.format(save_path, strategy_in_sample.agent.episodes))
    fig.clf()
    strategy_in_sample.save_agent(directory = save_path, filename = 'agent')

    print('------out_sample------')
    episode = strategy_in_sample.agent.episodes
    strategy_out_sample.restore_agent(directory = save_path, filename = 'agent')

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
        
    numpy.save('{}/out_sample_npy/{}.npy'.format(save_path, num_out_sample),temp_ndarray)
    df = pd.DataFrame(temp_ndarray.T)
    df['mean'] = df.mean(axis = 1)
    ax = df['mean'].plot()
    fig = ax.get_figure()
    fig.savefig('{}/out_sample_fig/{}.png'.format(save_path, num_out_sample))
    fig.clf()
    num_out_sample += 1
    
    avrg_reward = numpy.mean(rewards)
    out_sample_avrg_rewards.append(avrg_reward)

#%% save files
in_sample_rewards = pd.Series(strategy_in_sample._runner.episode_rewards)
numpy.save('{}/in_sample_rewards.npy'.format(save_path), in_sample_rewards)
numpy.save('{}/out_sample_rewards.npy'.format(save_path), out_sample_rewards)
numpy.save('{}/out_sample_avrg_rewards.npy'.format(save_path), out_sample_avrg_rewards)
copyfile('ppo_learn.py','{}/ppo_learn.py'.format(save_path))
 
in_sample_rewards.plot().get_figure().savefig('{}/in_sample_rewards.png'.format(save_path))
plt.cla()
pd.Series(out_sample_rewards).plot().get_figure().savefig('{}/out_sample_rewards.png'.format(save_path))
plt.cla()
pd.Series(out_sample_avrg_rewards).plot().get_figure().savefig('{}/out_sample_avrg_rewards.png'.format(save_path))
plt.cla()


performance = pd.DataFrame(data = performance.T, columns = ('balance','net_worth',
                                                            'open_amount','price'))

