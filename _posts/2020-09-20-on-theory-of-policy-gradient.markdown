---
layout: post
mathjax: true
title: "On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift"
date: 2020-09-20 10:00:00 -0700
tags: reinforcement-learning policy-gradient
author: <a href='https://gomahajan.github.io/'>Gaurav Mahajan</a>
excerpt: Policy gradient methods are among the most effective methods in challenging reinforcement learning problems with large state and/or action spaces. However, little is known about even their most basic theoretical convergence properties, including &#58; if and how fast they converge to a globally optimal solution or how they cope with approximation error due to using a restricted class of parametric policies. This work provides provable characterizations of the computational, approximation, and sample size properties of policy gradient methods in the context of discounted Markov Decision Processes (MDPs). We focus on both &#58; "tabular" policy parameterization, where the optimal policy is contained in the class and where we show global convergence to the optimal policy; and parametric policy classes (considering both log-linear and neural policy classes), which may not contain the optimal policy and where we provide agnostic learning results. One central contribution of this work is in providing approximation guarantees that are average case -- which avoid explicit worst-case dependencies on the size of state space -- by making a formal connection to supervised learning under distribution shift. This characterization shows an important interplay between estimation error, approximation error, and exploration (as characterized through a precisely defined condition number).
---

Reinforcement learning is a standard learning framework for studying how an agent learns and plans in an uncertain environment, guided by a reward signal. In a reinforcement learning problem, an agent learns a course of actions (called a 'policy') through its interaction with the environment and the goal is to find a policy which maximizes some measure of long term reward. 

Owing to the generality and flexibility of algorithms in RL, it has been crucial in a lot of recent empirical successes of artificial intelligence (see e.g. AlphaGo [[Silver et al., 2016]()], Atari [[Mnih et al., 2013]()], Robotics [[Mnih et al., 2015]()]).

{:refdef: style="text-align: center;"}
<div>
  <!--<img src="/assets/2020-09-20-policy-gradient/atari-breakout.png" width="200" height="250" style="margin: 20px 20px 20px 0px">
  <img src="/assets/2020-09-20-policy-gradient/mario.png" width="200" height="250" style="margin: 20px 20px 20px 0px">
  <img src="/assets/2020-09-20-policy-gradient/atari-pong.png" width="200" height="250" style="margin: 20px 20px 30px 0px">-->
  <img src="/assets/2020-09-20-policy-gradient/go.png" width="237" height="250" style="margin: 20px 40px 30px 0px">
  <img src="/assets/2020-09-20-policy-gradient/shadowhandc.png" width="367" height="250" style="margin: 20px 40px 30px 0px">
</div>
{:refdef}

One such attractive class of algorithms is policy gradient methods [[Williams,1992](), [Sutton et al., 1999](), [Konda and Tsitsiklis, 2000](), [Kakade, 2001]()]. Despite the large body of empirical work around these methods, their convergence properties are only established at a **relatively coarse level**; in particular, the folklore guarantee is that these methods **converge to a stationary point** of the objective, assuming adequate smoothness properties hold.  However, this local convergence viewpoint does not address some of the most basic theoretical convergence questions, including:
  1. If and how fast they **converge to a globally optimal solution**; 
  2. How they cope with **approximation error** due to using a restricted class of parametric policies.

In recent work with [Alekh Agarwal](), [Sham Kakade]() and [Jason Lee](), we place policy gradient methods under a solid theoretical footing,
analogous to the global convergence guarantees of iterative value function based algorithms.

### Markov Decision Processes

We first define a Markov Decision Process (MDP). A (finite) Markov Decision Process (MDP) $M = (\mathcal{S}, \mathcal{A}, P, r, \gamma,\rho)$ is specified by: a finite state space $\mathcal{S}$; a finite action space $\mathcal{A}$; a transition model $P$ 
where $P(s^\prime \| s, a)$ is the probability of transitioning into state $s^\prime$ upon taking action $a$ in state $s$; a reward function $r: \mathcal{S}\times \mathcal{A} \to [0,1]$ where $r(s,a)$ is the immediate reward associated with taking action $a$ in state $s$; a discount factor $\gamma \in [0, 1)$; a starting state distribution $\rho$ over $\mathcal{S}$.

{:refdef: style="text-align: center;"}
<div>
  <img src="/assets/2020-09-20-policy-gradient/mdp.png" width="700" height="250" style="margin: 0px 10px 0px 10px">
</div>
{:refdef}

A stochastic, stationary policy $\pi: \mathcal{S} \to \Delta(\mathcal{A}$  (where $\Delta(\mathcal{A})$ is the probability simplex over $\mathcal{A}$) specifies a decision-making strategy in which the agent chooses actions adaptively based on the current state, i.e., $a_t \sim \pi(\cdot \|s_t)$.

A policy induces a distribution over trajectories $\tau = (s_t, a_t, r_t)\_\{t=0\}^\infty$, where $s_0$ is drawn from the starting state distribution $\rho$, and, for all subsequent timesteps $t$, $a_t \sim \pi(\cdot \| s_t)$ and $s_{t+1} \sim P(\cdot \| s_t, a_t)$. The value function $V^\pi: \mathcal{S} \to  \mathbb{R}$ is defined as the discounted sum of future rewards starting at state $s$ and executing $\pi$, i.e.
\\[
V^\pi(s) := \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)
\| \pi, s_0 = s\right] \, ,
\\] where the expectation is with respect to the randomness of the trajectory $\tau$ induced by $\pi$ in $M$.

{:refdef: style="text-align: center;"}
<div>
  <img src="/assets/2020-09-20-policy-gradient/atari-breakout.png" width="200" height="250" style="margin: 20px 20px 30px 0px">
  <img src="/assets/2020-09-20-policy-gradient/mario.png" width="200" height="250" style="margin: 20px 20px 30px 0px">
  <img src="/assets/2020-09-20-policy-gradient/atari-pong.png" width="200" height="250" style="margin: 20px 20px 30px 0px">
</div>
{:refdef}

We further define $V^\pi(\rho)$ as the expected value under the initial state distribution $\rho$, i.e. 
\\[
V^\pi(\rho) := \mathbb{E}\_{s_0\sim \rho} [ V^\pi(s_0)] .
\\]

The goal of the agent is to find a policy $\pi$ that maximizes the expected value from the initial state, i.e. the optimization problem the agent seeks to solve is: 
\\[
\max_\pi V^{\pi}(\rho)
\\]
where the $\max$ is over all policies.

### Challenges in RL


### Policy Gradient Algorithm

### Softmax Policy Class: Asymptotic Convergence, without Regularization

### Softmax Policy Class: PG with Relative Entropy Regularization

### Softmax Policy Class: Natural Policy Gradient
