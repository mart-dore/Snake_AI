# Snake AI — Deep Q-Learning from Scratch

A reinforcement learning agent that learns to play Snake with no rules given to it and no
supervision — only a reward signal and 100 000 remembered transitions.

![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-000000?style=flat-square)

---

## How the agent sees the game

The agent never receives the board. It receives an **11-value binary state vector**, which is the
design decision that makes the whole thing tractable:

| Values | Meaning |
|---|---|
| 3 | Danger straight ahead / to the right / to the left, one block away |
| 4 | Current direction of travel (one-hot) |
| 4 | Where the food is, relative to the head (left / right / up / down) |

That is the entire input. It is deliberately relative rather than absolute — the agent learns
"there is a wall to my right", not "there is a wall at x=280". This keeps the state space tiny
and makes what it learns transfer across the board.

Three possible actions: go straight, turn right, turn left.

## The model

```
Linear_QNet(11, 256, 3)     input → 256 hidden (ReLU) → 3 Q-values
```

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | MSE on the Bellman target |
| Discount factor γ | 0.9 |
| Replay memory | 100 000 transitions |
| Batch size | 1 000 |
| Exploration | ε-greedy, ε = 150 − *games played* |

Training combines **short-term learning** (one gradient step per move) with **experience replay**
(a batch of 1 000 sampled transitions replayed at the end of each game). The replay buffer is what
stops the network from overfitting to whatever it did in the last few seconds.

Exploration decays linearly and stops around game 150, after which the agent plays purely on
what it has learned. The best model so far is checkpointed to `model/model.pth` whenever it beats
its own record.

## Running it

```bash
git clone https://github.com/mart-dore/Snake_AI
cd Snake_AI
pip install -r requirements.txt
```

Train from scratch — a live plot of score and rolling mean opens as it learns:

```bash
python agent.py
```

Watch the trained agent play:

```bash
python agent_trained.py
```

## Project structure

```
├── snake.py            # The game itself: rules, collisions, rendering
├── model.py            # Linear_QNet + QTrainer (Bellman update)
├── agent.py            # State encoding, replay memory, training loop
├── agent_trained.py    # Loads model/model.pth and plays
├── helper.py           # Live training plot
└── model/model.pth     # Best checkpoint
```

## Known limitations

The agent only sees danger **one block ahead**. It has no notion of the space it is enclosing,
so it reliably learns to chase food and just as reliably traps itself in its own tail once it
gets long. That is a limitation of the state design, not of the training — fixing it means
giving the agent a longer horizon, not more episodes.

## Next steps

- Add a flood-fill feature to the state, so the agent can see when it is about to trap itself
- Double DQN and duelling architectures, to reduce Q-value overestimation
- Convolutional network on the raw board, to compare against hand-crafted features

## License

MIT — see [LICENSE](LICENSE).
