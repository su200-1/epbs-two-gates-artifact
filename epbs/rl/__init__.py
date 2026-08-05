"""Reinforcement-learning Strategy Optimizer (SO) for the ePBS simulation.

A PPO + LSTM agent that trains on `epbs.env.EPBSFreeOptionEnv`. Tier 1 goal:
verify the RL pipeline converges to the known analytically-optimal free-option
strategy (`epbs.env.optimal_strategic_policy`).
"""
