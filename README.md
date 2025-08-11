# PyTorch Implementation of Policy Gradient Methods (REINFORCE & Baseline)

<p align="center">
  <img src="https://gymnasium.farama.org/_images/lunar_lander.gif" width="300"/>
</p>

Clean PyTorch implementations of PG algorithms with comparisons, gymnasium
envs support, logging, etc.

![Static Badge](https://img.shields.io/badge/status-WIP-blue)

---

## Code is being completely refactored
<p align="center">
  <img src="https://i.pinimg.com/236x/6b/7a/f4/6b7af47cf6889a90ea178ed89c089a82.jpg" width="100"/>
</p>

### Work in progress:
1. Implement reusable Agent, Buffer, and Net files/classes: ✅
2. Rewrite the shitcode and comment it ✅
3. Rewrite `algorithms/reinforce.py` ✅
4. Rewrite `algorithms/baseline.py` ✅
5. Implement `algorithms/a2c`
6. Implement other things

---

## Algorithms Overview
### 1. REINFORCE (Vanilla Policy Gradient)
The basic [Policy Gradient Method](https://en.wikipedia.org/wiki/Policy_gradient_method) implementation:
1. **Full episode play**
2. **Calculate discounted returns**:
   ![Return](https://latex.codecogs.com/svg.image?\%20$Q_{k,t}%20=%20\sum_{i=0}^{T-t}%20\gamma^i%20r_{t+i}$)
3. **Normalize returns**
4. **Calculate loss**:
   ![Loss](https://latex.codecogs.com/svg.image?\%20$L%20=%20-\sum_{t=0}^{T}%20\log%20\pi_\theta(a_t%20\mid%20s_t)\,R_t$)
5. **Update policy**

### 2. REINFORCE with Baseline
Enhanced version with reduced variance:
- Uses reward baseline: `b = 𝔼[R]`
- Advantage function: `A(s,a) = R - b`
- More stable learning dynamics (kind of)

---

## Installation
1. Clone the repository:  
```bash
git clone https://github.com/buuuuulat/pytorch_policy_gradients.git
cd pytorch_policy_gradients
```

2. Install requirements:
```bash
pip install torch gymnasium numpy
```

---

## Run
### How to run
All algorithms can be found in the `algorithms/` directory and are ready
to be used. You can adjust hyperparameters and choose any gymnasium compatible
environment.

#### Example Usage
```bash
python3 algorithms/reinforce.py
```

Note: Continuous action space is not yet supported.

### Basic Training Process Overview
```pseudocode
while num_episodes < n:
    for step in episode:
        action = agent.choose_action(obs)
        exp = env.play_step(action)
        buffer.append(exp)
        buffer.calc_rewards()
        agent.update_grads()
```

---

## Comparison between **Vanilla REINFORCE** and **REINFORCE with mean baseline**
Not much better really

![Algorithm Comparison](data/graph1.png)

However, in some cases, baseline show 2x boost in convergence
speed and highest reward

![Algorithm Comparison](data/graph2.png)

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

### License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
> **Educational Focus**: Clear implementations for learning Policy Gradient methods, with practical comparison between
> vanilla and baseline-enhanced versions, so the speed and optimization were not the main goal.
>
> In order to optimize it, cuda support should be properly added and List usages in buffer should be avoided as well
> as the transitions between torch Tensors and numpy Arrays.