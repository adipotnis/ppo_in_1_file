# Writing PPO in 1 file

I have been thinking of studying PPO and reimplementing it in 1 file for my personal learning. 
I'll be documenting what I learn and some nice visualizations from it as I go through it. 

Like any big project which can be hard to start off, I'll try to break it down into subcomponents that are easier to learn step by step. 

I'm using gym to keep it simple and make it work on a cpu, maybe will go more advanced later. Also planning to add a jax version 

## Plan / order of implementation

Ordered simplest → hardest, so each step runs and is testable before the next builds on it. CartPole first (discrete, MLP, CPU), then continuous actions, then pixels.

### Phase 0 — scaffolding
- [ ] Get a gym env stepping with a random agent (`CartPole-v1`), print episode return
- [ ] Set up the single-file skeleton: arg parsing, seeding, device selection

### Phase 1 — networks
- [x] `mlp` builder (list of layer sizes → `nn.Sequential`)
- [x] Actor head: `CategoricalActor` for discrete actions (softmax over logits)
- [x] Critic head: value network `v(s)` (scalar output)
- [x] Combine into an `ActorCritic` (`ac`) module

### Phase 2 — rollout collection
- [x] `PPOBuffer` to store `(obs, act, rew, val, logp)` for a trajectory
- [ ] `PPOBuffer` -> `path_finish`, `path_finish`
- [ ] Collect a fixed number of steps per update by acting in the env

### Phase 3 — advantage & returns
- [ ] Reward-to-go (discounted return) via reverse sweep
- [ ] GAE-λ advantages via reverse sweep (with episode-boundary reset)
- [ ] Advantage normalization

### Phase 4 — the PPO update
- [ ] Policy loss: clipped surrogate objective (`ratio`, `clamp`, `min`)
- [ ] Value function loss (MSE to returns)
- [ ] Entropy bonus for exploration
- [ ] Training loop: K epochs over the buffer, minibatches
- [ ] Early stop on approx-KL threshold (`--kl`)

### Phase 5 — make CartPole learn
- [ ] End-to-end training loop with logging (return, loss, KL, entropy)
- [ ] Save / load checkpoints (`--load_from`) and `--render` rollout
- [ ] Plot learning curve

### Phase 6 — continuous actions
- [ ] `GaussianActor` (mean net + learnable `log_std`)
- [ ] Verify on a continuous control env

### Phase 7 — learning from pixels
- [x] `cnn` feature extractor — Atari-style conv stack, flatten 
- [ ] `FramePreStacking` (stack n frames → position/velocity via diffs)
- [ ] Train on Pong

### Phase 8 — later / stretch
- [ ] Vectorized envs for faster rollouts
- [ ] JAX version
