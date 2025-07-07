# PyTorch implementation of Monte Carlo Policy Gradient Method (REINFORCE Algorithm)

<p align="center">
  <img src="https://gymnasium.farama.org/_images/lunar_lander.gif" width="300"/>
</p>

---

### Table of Contents
* [Algorithm overview](#algorithm-overview)
* [Installation](#installation)
* [Run](#run)
* [Usage examples](#usage-examples)
* [Configuration & Hyperparameters](#configuration--hyperparameters)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [License](#license)

---

### Algorithm Overview
This is the [Policy Gradient Method](https://en.wikipedia.org/wiki/Policy_gradient_method) that
consists of the following steps:
1. **Network initialization**  
2. **Full episode play**  
3. **Calculation of the discounted return** for every step:
![Return](https://latex.codecogs.com/svg.image?\%20$Q_{k,t}%20=%20\sum_{i=0}^{T-t}%20\gamma^i%20r_{t+i}$)
4. **Calculation of the Loss Function**:
![Loss](https://latex.codecogs.com/svg.image?\%20$L%20=%20-\sum_{t=0}^{T}%20\log%20\pi_\theta(a_t%20\mid%20s_t)\,R_t$)
5. **Optimizer update**  
6. **Repeat** from step 2 until convergence  

---

### Installation
1. Clone the repository:  
```bash
   git clone https://github.com/yourusername/policy-gradients.git
   cd policy-gradients
```

2. (Optional) Create & activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. Install requirements:

```bash
   pip install -r requirements.txt
```

---

### Run

Train a policy on CartPole‑v1 for 1,500 episodes, then evaluate with rendering:

```bash
# Train
python play.py --train_episodes 1500

# Evaluate / Play with human render
python play.py --play --render
```

---

### Usage examples

#### In Python scripts

```python
import gymnasium as gym
from reinforce import Net, Agent

# 1. Setup training env (no rendering)
train_env = gym.make("CartPole-v1")
net       = Net(in_features=train_env.observation_space.shape[0],
                out_features=train_env.action_space.n)
agent     = Agent(net, train_env)

# 2. Train
agent.learn(1500)

# 3. Evaluate in human‑render mode
eval_env  = gym.make("CartPole-v1", render_mode="human")
eval_agent = Agent(net, eval_env)
reward    = eval_agent.play(render=True)
print("Evaluation reward:", reward)

# 4. Close the evaluation env
eval_env.close()
```

---

### Configuration & Hyperparameters

You can tweak the following parameters in `play.py` or directly in your script:

* **`learning_rate`** (default `1e-3`)
* **`gamma`** (discount factor, default `0.99`)
* **Network architecture** (hidden sizes: 128 → 256 → 128)
* **Batch normalization / baselines** (not implemented by default)

---

### Dependencies

* Python 3.8+
* [gymnasium](https://gymnasium.farama.org/)
* [torch](https://pytorch.org/)
* [numpy](https://numpy.org/)

Install via:

```bash
pip install gymnasium torch numpy
```

---

### Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

### License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

