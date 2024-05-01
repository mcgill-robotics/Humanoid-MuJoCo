# python3 train.py --algo ppo --n-envs 1 --cpu --log-name PPO --rand-init 1.0 --n-steps 10_000_000 --reward-goal -1
python3 train.py --algo sac --log-name SAC_GPU_10 --rand-init 1.0 --n-steps 15_000_000 --reward-goal 0
# python3 train.py --algo sac --cpu --log-name SAC_CPU --rand-init 1.0 --n-steps 10_000_000 --reward-goal 0
# python3 train.py --algo td3 --cpu --log-name TD3 --rand-init 0.0