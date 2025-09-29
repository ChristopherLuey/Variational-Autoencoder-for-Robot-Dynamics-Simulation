# Real-Time Gait Discovery and Dynamics Learning Roadmap

This document sketches concrete engineering steps to extend the current offline Variational Autoencoder (VAE) training pipeline toward online, real-time gait discovery and dynamics learning while a robot is walking.

## 1. Streamed Data Interface

* Replace the current SQL query that bulk-loads trajectories before training with a streaming data API.
* Implement a producer (e.g., `EnvSampler`) that collects the most recent window of sensor/state/action tuples from the simulator or robot and pushes them to a ring buffer on every control tick.
* Update the training loop so that `BasicAutoencoder.train_single_epoch` consumes micro-batches from this buffer instead of the entire static tensor. Use asynchronous CUDA streams to overlap collection and training.

## 2. Latent Dynamics Head

* Augment `BasicAutoencoder` with a small latent dynamics model (e.g., GRU or linear state-space layer) that predicts the next latent vector from the previous latent and current action. Train it jointly with the reconstruction loss and a prediction loss between predicted and actual latent states.
* Export both the decoder and dynamics head so that a Model Predictive Control (MPC) or shooting method can plan actions directly in latent space during runtime.

## 3. Continual / Online Optimisation

* Use experience replay with prioritisation to retain high-reward trajectories while allowing fresh samples to dominate gradients.
* Adopt low learning rate Adam or RMSProp with gradient clipping already present to stabilise online updates, and add EMA (exponential moving average) weights for deployment.

## 4. Real-Time Inference Path

* Preload the trained decoder on the control computer and expose a `decode(latent, condition)` interface that runs in under the control-loop latency budget (e.g., <1 ms on GPU).
* Convert the PyTorch modules to TorchScript or TensorRT for deterministic latency when executing on embedded hardware.

## 5. Safety Guards

* Implement predictive checks that reject latent rollouts when the decoder reconstructs torques outside joint limits or when the latent dynamics head predicts instability, falling back to a safe gait library.

## 6. Evaluation Protocol

* Create benchmarks that measure convergence speed of online updates versus offline retraining, both in simulation and (if available) on hardware logs.

## 7. Prototype Implementation

An initial reference implementation of this online loop now lives in [`online_training.py`](../online_training.py). It streams new gaits as they are executed, perturbs them for exploration, and applies `BasicAutoencoder.online_update` on every rollout while maintaining a replay buffer for continual rehearsal. Results, best-performing gaits, and updated weights are written to `results/online_<timestamp>/` for inspection.

