# Tensortrade
李松浩，华泰证券实习生
修改自https://github.com/notadamking/tensortrade
运行前请确认tensorflow版本为1.13.1或1.14.0, tensorforce版本为0.5.2
Tensortrade是一个模块化的量化强化学习环境，经过大幅度修改，已经实现量化数据传入、期货环境适配、tensorforce算法的部署
## 主要内容：模块概述｜运行逻辑｜模块详细介绍｜案例分析(PPO/A2C)

# 模块概述
目前已经研究、修改、投入使用的tensortrade模块主要是两大部分：交易环境和强化学习智能体Agent
其中，交易环境由environment.TradingEnvironment管理，主要包括以下子模块：
1. 数据输入（状态空间）exchange
2. 行为生成（行为空间）action
3. 奖励函数reward
4. 交易属性trade
这四个模块已经按照原来的框架重新写过了适合期货交易环境的Class，后续会有详细介绍
还有两个子模块feature_pipeline(传入数据标准化)和slippage没有做深入研究
数据标准化可以在传入之前就处理，自带的处理可能会用到未来数据(比如minmaxnormalizer)，所以程序调用为空
slippage用的是默认设置

强化学习agent主要是通过strategies.tensorforce_trading_strategy文件，调用tensorforce强化学习库
tensorforce强化学习库是用tensorflow搭建，运行速度理想，存储方便
原版tensortrade也适配了stable_baselines强化学习库，初步使用速度较慢，未做研究

主文件夹中有上述所有模块的独立子文件夹：
exchanges, actions, environments, exchanges, features, rewards, slippage, strategies
还有两个文件夹Data和save用作运行数据的读取和保存，详见后续案例分析

# 运行逻辑
通过传入各个模块设置好环境后，调用tensortrade.strategies.tensorforce_trading_strategies.run开始运行
run调用的是tensorforce包中 tensorforce.execution.Runner.run开始运行
主要的逻辑是对回合(episode)进行循环，每一回合中调用run_episode, 在回合中对步(timestep)进行循环
## 最小循环单位：timestep的运行
上一步environment.execute 生成这一步的环境 （即，智能体可以观测的变量，open/close/high/low或者因子值）
1. 调用agent.act，  传入这一步的环境state(由上一个timestep生成)， 智能体根据状态和policy生成一个action(自然数)
2. 调用environment.execute， 将action传入进交易环境， 交易环境进行解读（exchange|action|reward|trade)， 生成step_reward， 判断是否回合终点， 更新下一步的环境state
在交易环境中，先是通过action模块将自然数解读为交易信号，生成trade实例，然后在reward模块中根据trade实例生成回报，再通过exchange生成下一步环境state
3. 调用agent.observe, 判断、 进行并记录是否更新参数(update)， 记录奖励(reward)， 进入下一个timestep

# 模块详细介绍
主要介绍exchange，action，reward，trade这几个模块，并简单说一下tensorforce学习库的调用
## exchange
exchange模块的主要功能是我们狭义理解的“市场环境”。在程序运行前，输入并整理数据；在运行过程中，在每一step上生成观测值，观测值用于输入tensorflow进行计算，生成行为。
我根据simulated_exchange仿写了future_exchange和future_exchange_position，父类是instrument_exchange

### future_exchange用于【action对应交易】的环境
主要改动有：
#### Pandas - Numpy
尽量把所有涉及到Pandas的计算改成了Numpy，提升运行速度
#### exclude_close （传入参数）
若为True，则输入的数据中，obs不包括close，而close只用于生成price(价格)更新账户和计算收益
#### current_price （方法) 
从self.price中读数据，和用于生成观测的dataframe隔离，和上面是一个道理
#### is_valid_trade  (方法) 
持仓产品量小于等于1以及持仓价值小于等于账户净值
#### observe_position (传入参数) 
是否把当前持仓传入tensorflow中作为可以“学习”的对象。这个在逻辑上非常重要。对应修改了data_frame.setter和obs生成器，主要思路是程序运行前在dataframe里面新增一个position空列，然后在生成obs的时候把这一step的position替换成当前持仓
#### reset（方法）
每次reset会把记录运行表现的Performance(Numpy.array)清空重置。我在重置前把Performance转换成DataFrame，并把最后五行在console打印出来。如果为了提高运行速度可以舍弃。

### future_exchange_position用于【action对应持仓】的环境
是在future_exchange基础上进一步改的，主要改动有：
#### is_valid_trade(方法)
因为action对应持仓，所以持仓产品量直接就被限制了。这里直接return True
#### next_price
观测下一个timestep的价格。这里是为了评价在t时刻的【action即持仓】的reward，比较适合按步更新

## action
action模块的主要功能是生成动作空间（用于tensorflow生成动作的空间），并将tensorflow计算出的action（常为整数值）解读成交易信号。我根据discrete_action_strategy仿写了future_action_strategy和future_position_strategy
### future_action_strategy用于【action对应交易】的环境
默认每次交易0.1个产品， 只有买｜不交易｜卖三种行为。 如果要增加行为种类可以参照下面future_position_strategy进行修改


### future_position_strategy用于【action对应持仓】的环境
在future_action_strategy的基础上修改，思路是：
每一个action对应的是持仓大小。类似的，n_action 设置成5，那么总共会生成0｜1｜2｜3｜4 五种action，size = 2， 对应的这一步的持仓就是 -1｜-0.5｜0｜0.5｜1
主要修改：
增加了last_position和next_price，读取上一次持仓，通过change生成交易；读取下一次价格，评价本次持仓动作盈亏

## reward
### DirectProfitStrategy用于【action对应交易】的环境
是我重写的，其实本质上，这个适用于【按照回合更新】的环境而完全不适用【按照步更新】的环境，但是为了统一化，标题如上写
每一个timestep上的收益是根据：
上一个timestep执行交易后的持仓量 *（本timestep价格-上timestep价格）- 本timestep产生的交易手续费
尽管在timestep上，reward并不能描述此次action；
但是在整个episode中，reward的总和能描述整个policy的情况，所以适用于【按照回合更新】的环境
### PositionReward用于【action对应持仓】的环境
这是更新后的思路。为了确保在每一个timestep上，reward能够描述此次action（即持仓）的收益
每一个timestep上的收益是根据：
这次持仓的量*（下次价格-这次价格） - （为了达成）这次持仓产生的交易手续费
这个收益就是完全描述这一次持仓的盈亏了

## trade
用于支持trade实例生成，以在环境内数据流动
主要增加了一个next_price属性，初始化实例的时候如果不传就是0，这样不需要用到这个属性的交易策略也不用修改代码
同时还有trade_type类，用来支持trade的，也略做修改

# 案例分析：PPO
每一个###对应程序中的一个代码块
## 环境设置
### Action & Rewards
设置调用的行为生成模块和奖励函数模块。调用DirectProfitStrategy和FutureActionStrategy，如上所述
### Feature_pipeline
数据处理模块。未做调用，留在代码中展示调用方式。如调用，则feature_pipeline = FeaturePipeline(steps=[normalize, difference])
### Data Input (调用Exchanges)
数据输入。具体格式可以参照Data/TA.csv
必须得有一列，列名为close以生成价格序列，其他的观测序列数量和列名可自定 
### Agent Specification
神经网络和智能体超参数的设置
agent_spec中，learning_rate, discount和likelihood_ratio_clipping详见PPO算法介绍，最好不要改动
network是agent网络，critic_network是critic网络
batch_size是update网络参数的时候，传入进去的回合数量(PPO按回合更新)
update_frequency是每多少个回合更新一次。底层代码用的是同余，回合序数被整除就更新。一般来说这两个参数应该相同，按照tensorforce的默认值是10
### Environment Setup
把所有的环境整合到一起。先用TradingEnvironment把exchange, action, reward, feature_pipeline整合到一起定义强化学习的环境，再用TensorForceTradingStrategy把强化学习环境和强化学习智能体整合到一起。至此，环境设置完成。

## 运行参数
### 简单运行
simple_PPO.py为例
performance = strategy.run(episode = 1000, evaluation = False)
其中Evaluation是runner中自带的评价方式，和我们写的reward无关，如果要用就要写evaluation的方法和频率。举个例子就是我们可以用累计净值做策略的迭代评价，但是evaluate这里可以用夏普
performance是最后一个回合运行的结果

### 样本内外
PPO_learn.py 为例
首先在之前设置环境的时候，exchange, environment, strategy设置in_sample和out_sample两份，除了数据不一样外其他的参数都一样
思路上是，在in_sample上连续训练，然后在中途把智能体agent存档（本质上是神经网络的参数），然后让out_sample读取存档，并在out_sample数据上运行策略。每次存档的时候也保存图片
#### 备注
在读取存档的时候，也会读取存档中agent update和episode的次数，所以传参的时候要在现有episode的基础上增加多少个想要运行的episode
out_sample_frequency应该是上面agent update_frequency的倍数。只有agent更新之后，再检验out_sample才有意义，不然本质上是同样的策略重新跑一次out_sample
out_sample_trials是out_sample运行的次数，然后取净值曲线的平均值生成图片。可以理解成n个AI交易员每个人分到1/n资金，根据同样的非确定性策略进行交易，然后所有人净值的平均值。次数应当小于update_frequency，不然超过的参数就已经更新了。。。 
## 结果分析
有以下几个数据是可以从运行后的结果导出
### 最后一回合运行情况
performance = pd.DataFrame(data = performance.T, columns = ('balance','net_worth',
                                                            'open_amount','price'))
上面提到performance是最后一个回合运行的结果，本质上是excahnge里面，一个回合中不断update_account留下的
最后一个回合之前的所有回合运行完之后都会调用exchange里的reset，最后一个回合没有reset，留了下来
为了提高运行速度，代码过程中用的是numpy，最后转换成dataframe
第一列是当前step的现金；第二列是资产净值，从initial_balance开始；第三列是持仓的产品数量；第四列是当前产品价格
(理论上net_worth = balance + open_amount * price)
### update(优化/神经网络参数迭代)次数
strategy._runner.updates 或
strategy.agent.updates
### 学习曲线
横轴：episode
纵轴：这一个episode的total rewards
strategy._runner.episode_rewards 即为记录reward的list

### 备注
performance是每一次调用run的时候返回，只能得到最后一次运行的结果。因为运行的策略是非确定性策略，所以具有随机性。如果要得到诸多dataframe用于分析，可参照【样本内外】程序的写法
学习曲线和update都是记录在strategy实例中的属性，如果重新调用了可能会清空，但是用spyder就可以运行完后再在console里面调出来，也可以直接在程序里面写了调用出来
# 案例分析：A2C
## 思路
考虑A2C算法主要是因为A2C在原理上可以按步更新，与PPO按回合更新不同
如果要按步更新，就必须在每一步上reward和action能对应上，如果按照【action对应交易】的模块，reward和action是错位的，甚至reward对应的是许许多多之前的action（交易）积累起来的持仓值，不合理
于是就采用了【action对应持仓】这一套模块
## 环境设置
### Action & Rewards
调用PositionReward和FuturePositionStrategy
### Feature Pipeline
同PPO不做处理
### Data Input (Exchanges)
同PPO
### Agent Specification
max_episode_timesteps 这个参数有些奇怪， 程序中感觉是“程序刚开始时， 前多少个timesteps是不更新的”。比如说数据总共有20000行(20000steps)，设置这个参数为30000，就相当于前一个半回合不更新。具体可以看tensorfoce\core\models\tensorforce.py line 445 tf_core_observe这个函数，self.estimator.capacity是这个值
batch_size 和 update_frequency 原理跟ppo案例是一样的，只不过单位从episode换成了timestep
## 运行
performance = strategy.run(episodes=100, evaluation=False)
learning_curve = pd.Series(strategy._runner.episode_rewards)

## 按步更新特别注意：agent.observe
tensorforce.agent.agent.py line 441
调用observe后，在这里进一步判断是否要再往深判断要不要observe
if terminal > 0 or index == self.buffer_observe or query is not None:
terminal即为回合终点, 其他两个条件正常情况均不满足
所以，如果不在回合终点，就不会进入是否update的观察
这是我个人的理解，可能对tensorforce理解不够深入
手动将这个条件改成or true后a2c可以运行并有一定成果，实测不会影响ppo的程序
