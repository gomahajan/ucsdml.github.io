---
layout: post
mathjax: true
title: "On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift"
date: 2020-09-20 10:00:00 -0700
tags: reinforcement-learning policy-gradient
author: <a href='https://gomahajan.github.io/'>Gaurav Mahajan</a>
excerpt: Policy gradient methods are among the most effective methods in challenging reinforcement learning problems with large state and/or action spaces. However, little is known about even their most basic theoretical convergence properties, including &#58; if and how fast they converge to a globally optimal solution or how they cope with approximation error due to using a restricted class of parametric policies. This work provides provable characterizations of the computational, approximation, and sample size properties of policy gradient methods in the context of discounted Markov Decision Processes (MDPs). We focus on both &#58; "tabular" policy parameterization, where the optimal policy is contained in the class and where we show global convergence to the optimal policy; and parametric policy classes (considering both log-linear and neural policy classes), which may not contain the optimal policy and where we provide agnostic learning results. One central contribution of this work is in providing approximation guarantees that are average case -- which avoid explicit worst-case dependencies on the size of state space -- by making a formal connection to supervised learning under distribution shift. This characterization shows an important interplay between estimation error, approximation error, and exploration (as characterized through a precisely defined condition number).
---

Policy gradient methods have a long history in the reinforcement learning (RL) literature [[Williams,1992](), [Sutton et al., 1999](), [Konda and Tsitsiklis, 2000](), [Kakade, 2001]()] and are an attractive class of algorithms in practice as they are simple to implement, you can estimate the gradient using only simulations; easily deal with large state and/or action spaces; model free, directly optimize the cost function of interest; and applicable to any differentiable policy parameterization.

{:refdef: style="text-align: center;"}
<div>
  <img src="/assets/2020-09-20-policy-gradient/atari-breakout.png" width="200" height="250" style="margin: 20px 30px 30px 0px">
  <img src="/assets/2020-09-20-policy-gradient/atari-pong.png" width="200" height="250" style="margin: 20px 30px 30px 0px">
  <img src="/assets/2020-09-20-policy-gradient/go.png" width="237" height="250" style="margin: 20px 0px 30px 0px">
</div>
{:refdef}

Despite the large body of empirical work around these methods, their convergence properties are only established at a relatively coarse level; in particular, the folklore guarantee is that these methods converge to a stationary point of the objective, assuming adequate smoothness properties hold.  However, this local convergence viewpoint does not address some of the most basic theoretical convergence questions, including:
  1. If and how fast they converge to a globally optimal solution; 
  2. How they cope with approximation error due to using a restricted class of parametric policies.

In this blog post, which is based on [recent work]() with [Alekh Agarwal](), [Sham Kakade]() and [Jason Lee](), we will provide a provable characterization of the policy gradient methods and try to understand both optimization and approximation issues underlying these algorithms.

# Markov Decision Processes

We first define a Markov Decision Process (MDP). A (finite) Markov Decision Process (MDP) $M = (\mathcal{S}, \mathcal{A}, P, r, \gamma,\rho)$ is specified by: a finite state space $\mathcal{S}$; a finite action space $\mathcal{A}$; a transition model $P$ 
where $P(s^\prime \| s, a)$ is the probability of transitioning into state $s^\prime$ upon taking action $a$ in state $s$; a reward function $r: \mathcal{S}\times \mathcal{A} \to [0,1]$ where $r(s,a)$ is the immediate reward associated with taking action $a$ in state $s$; a discount factor $\gamma \in [0, 1)$; a starting state distribution $\rho$ over $\mathcal{S}$.

{:refdef: style="text-align: center;"}
<div>
  <img src="/assets/2020-09-20-policy-gradient/mdpc.png" width="547" height="240" style="margin: 0px 10px 0px 10px">
</div>
{:refdef}

A stochastic, stationary policy $\pi: \mathcal{S} \to \Delta(\mathcal{A}$  (where $\Delta(\mathcal{A})$ is the probability simplex over $\mathcal{A}$) specifies a decision-making strategy in which the agent chooses actions adaptively based on the current state, i.e., $a_t \sim \pi(\cdot \|s_t)$. A policy induces a distribution over trajectories $\tau = (s_t, a_t, r_t)\_\{t=0\}^\infty$, where $s_0$ is drawn from the starting state distribution $\rho$, and, for all subsequent timesteps $t$, $a_t \sim \pi(\cdot \| s_t)$ and $s_{t+1} \sim P(\cdot \| s_t, a_t)$. 

The goal of the agent is to find a policy $\pi$ that maximizes the expected value from the initial state, i.e. the optimization problem the agent seeks to solve is: 
\\[
\max_\pi V^{\pi}(\rho) := \mathbb{E}\_{s\sim  \rho}\left[\sum_{t=0}^\infty \gamma^t  r(s_t, a_t)
			\big|s_0 = s\right] \, .
\\]
where the $\max$ is over all policies.

# Policy Gradient Algorithm and Policy Class
This blog studies ascent methods for the optimization problem:
\\[
  \max_{\theta\in \Theta} V^{\pi_\theta}(\rho) ,
\\]
where $\\{\pi_\theta|\theta \in \Theta\\}$ is some class of parametric (stochastic) policies. These methods update the parameters in the direction of the gradient.
\\[
  \theta^{(t+1)} = \theta^{(t)} +  \eta \Delta_\theta V^{(t)}(\mu)
\\] with a slight caveat that we compute the gradient under starting state distribution $\mu$ (which can be different than $\rho$ for better exploration). 

These methods optimize over a parameterized policy class, and we will consider some very natural policy classes of the form: 
\\[
\pi_{\theta}(a|s) \propto \exp(f_\theta(s,a))
\\] for some function $f_\theta$. These classes are as follows:
1. Softmax policy class: $f_\theta(s,a) = \theta_{s,a}$.
2. Log-Linear policy class: $f_\theta(s,a) = \theta^\top \phi(s,a)$ where $\phi(s,a)\in \mathbb{R}^d$.
3. Neural policy class: $f_\theta(s,a)$ is a neural network.

# Non-Concave Maximization
<div class="lemma">
			There is a MDP such that $V^{\pi_\theta} (s)$ is non-concave for the softmax	parameterization.
</div>

# Softmax Policy Class: Asymptotic Convergence, without Regularization
<div class="theorem">
				Assume $\mu(s)>0$ for all states $s$. Then for all states $s$,
</div>
\\[
V^{(t)}(s) \rightarrow V^\ast(s) \quad \text{as} \quad t \rightarrow \infty\, .
\\]

# Softmax Policy Class: PG with Relative Entropy Regularization
<div class="theorem">
		After $T$ iterations, for all starting state distributions $\rho$ and for uniform starting state distribution $\mu$, PG with entropy regularization satisfies:
    </div>
\\[
		\min_{t< T} \left\\{ V^\ast(\rho) - V^{(t)}(\rho)\right\\}
		\leq \frac{20 |\mathcal{S}|^2 |\mathcal{A}| }{(1-\gamma)^3 \sqrt{T}}\, .
\\]

# Softmax Policy Class: Natural Policy Gradient
