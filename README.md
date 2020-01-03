# Tesortrade
Songhao Li
Modified from https://github.com/notadamking/tensortrade 

Tensortrade is a modulized quantitative reinforcement learning project. According to my adjustment, this project has achieved self-provided data input, futures market environment configuration, and deployment of tensorforce agents.

# Module Description
There are two major modules: trading environment and learning agent. Trading environment is managed by environment.TradingEnvironment, including the following sub-modules:
1. Exchanges (state space)
2. Actions (action space)
3. Reward 
4. Trade (support data flow in the program)
This four sub-moduls have been expanded with new classed to fit futures market environment; there will be specified introduction later in this document.

Learning agent is managed by strategies.tensorforce_trading_strategy to call and configure a tensorforce learning agent.
Tensorforce agent is set up via tensorflow framework

# Running the Program
Two tutorails using sample codes/data in the project will be offered; the runner of the program is set up in strategies.tensorforce_trading_strategies.run. 
Basically, the program iterates on every episodes, and in each episode the program iterates on every timesteps
## inside a timestep:
1. Call agent.act: agent make action(a natural number) based on the state
2. Call environment.execute: bringing action to the trading environment; interpreting the natural number to be a trade; generate reward for the trade (action); generate next state
3. Call agent.observe: judging whether to update networks; recording reward; going into next timestep

# Details of Modules