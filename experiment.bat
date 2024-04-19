python3 train.py --algo ppo --cpu --log-name PPO --rand-init 1.0 --reward-goal 2000
python3 train.py --algo sac --cpu --log-name SAC --rand-init 1.0 --n-steps 10000000 --reward-goal 2000
@REM python3 train.py --algo td3 --cpu --log-name TD3 --rand-init 0.0