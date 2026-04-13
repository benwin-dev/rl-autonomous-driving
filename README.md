# Reinforcement Learning for Tactical Decision-Making in Autonomous Driving

## Project Overview
This project explores how **Reinforcement Learning (RL)** can be applied to tactical decision-making problems in **autonomous driving**, with a primary focus on **unprotected left turns at intersections**.

In real-world driving, unprotected left turns are challenging because the vehicle must decide **when it is safe to proceed across oncoming traffic** without the assistance of a dedicated traffic signal. The decision depends on estimating the speed, distance, and timing of surrounding vehicles while balancing both **safety** and **efficiency**.

The goal of this project is to train an RL agent that can learn safe and efficient driving decisions in this uncertain traffic scenario using simulation.

---

## Team Members
- **Benwin George**
- **Thilak Sampath Kumaran**
- **Matthew A Phillips**

---

## Project Objective
The primary objective of this project is to:

- Model an **unprotected left-turn intersection scenario**
- Train an **RL agent** to make tactical decisions
- Evaluate whether the learned policy can safely and efficiently navigate the intersection
- Compare the RL agent against a **simple rule-based baseline**

If time permits, the project will also be extended to a **highway merging scenario** to study whether the learned behavior can generalize to other tactical driving tasks.

---

## Problem Statement
Autonomous vehicles must operate in environments where uncertainty and interaction with other vehicles are unavoidable. One particularly difficult situation is the **unprotected left turn**, where the ego vehicle must determine the right time to turn while avoiding collisions with oncoming traffic.

This project frames the problem as a **decision-making task** in which the agent must learn when to:

- **Wait**
- **Move forward / creep**
- **Initiate the turn**

The challenge is to learn a policy that is:
- **Safe** (avoids collisions)
- **Efficient** (does not wait unnecessarily long)
- **Robust** (handles varying traffic conditions)

---

## Tools and Technologies
This project is implemented using:

- **Python**
- **Gymnasium**
- **highway-env**
- **Stable-Baselines3**
- **Proximal Policy Optimization (PPO)**
- **Matplotlib / NumPy / Pandas** (for analysis and visualization)

---

## Simulation Environment
We use the `intersection-v0` environment from **highway-env** as the starting simulation platform.

The environment provides:
- A controllable traffic intersection
- Multiple surrounding vehicles
- Vehicle dynamics and traffic flow simulation
- A testbed for tactical decision-making

The RL agent controls the **ego vehicle** and learns from repeated interaction with the simulated environment.

---

## Reinforcement Learning Setup

### State / Observation Space
The agent observes information about:
- Ego vehicle position
- Ego vehicle speed
- Nearby vehicle positions
- Nearby vehicle speeds
- Relative distances / traffic conditions

### Action Space
The agent selects actions that affect its driving behavior in the intersection, such as:
- Waiting
- Accelerating / moving forward
- Decelerating
- Turning maneuvers

### Reward Function
The reward function is designed to encourage:
- **Successful completion of the turn**
- **Collision avoidance**
- **Efficient decision-making**
- **Reduced unnecessary waiting**

Example reward considerations:
- Positive reward for safely crossing the intersection
- Large penalty for collisions
- Small penalty for excessive waiting or unsafe maneuvers

---

## Learning Algorithm
We use **Proximal Policy Optimization (PPO)** for training the agent.

PPO is chosen because it is:
- Stable and widely used in RL research
- Well-suited for continuous interaction environments
- Easy to implement using Stable-Baselines3

---

## Evaluation Metrics
The performance of the RL agent will be evaluated using:

- **Success Rate**  
  Percentage of episodes where the vehicle successfully completes the turn

- **Collision Rate**  
  Percentage of episodes ending in collision

- **Average Waiting Time**  
  Time spent waiting before making the turn

- **Episode Reward**  
  Total cumulative reward over an episode

The RL-based policy will also be compared with a **simple baseline strategy** to measure improvements.

---

## Running Evaluation
After training, run the evaluator to compute proposal-aligned metrics and compare PPO to a simple baseline:

```bash
python evaluate_ppo.py --episodes 100 --model-path ppo_intersection_model
```

Save report-ready files:

```bash
python evaluate_ppo.py --episodes 100 \
  --output-json results/eval_report.json \
  --output-csv results/eval_report.csv
```

Metrics reported:
- Success rate
- Collision rate
- Timeout rate
- Average episode reward
- Average waiting time (steps and seconds)

You can disable the random baseline comparison with:

```bash
python evaluate_ppo.py --episodes 100 --no-random-baseline
```

---

## Reward Variant Experiments
Train reward-focused variants (safety ablation):

```bash
./train_reward_variants.sh 100000
```

This script trains:
- `collision_strong` (`collision_reward` increased in magnitude)
- `near_miss_penalty` (extra penalty when ego gets too close to other vehicles)
- `waiting_penalty` (small penalty when ego stays near zero speed)

You can also train a single variant:

```bash
python train_ppo.py --reward-variant near_miss_penalty --timesteps 100000
```

Then evaluate each model with:

```bash
python evaluate_ppo.py --episodes 100 --model-path ppo_intersection_model_near_miss_penalty
```

---
