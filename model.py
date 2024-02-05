#TODO -> implement RL model here

# STATE/REWARD INFO FROM https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=HlRhFs_d3WLP

# STATE
    # joint positions     5 · 20          Joint positions in radians (stacked last 5 timesteps)
    # linear acceleration 5 · 3           Linear acceleration from IMU (stacked)
    # angular velocity    5 · 3           Angular velocity (roll, pitch, yaw) from IMU (stacked)
    # foot pressure       5 · 8           Pressure values from foot sensors (stacked)
    # gravity             5 · 3           Gravity direction, derived from angular velocity using Madgwick filter (stacked)
    # agent velocity      5 · 2           X and Y velocity of robot torso (stacked)
    # previous action     5 · 20          Action filter state (stacked)
    
# REWARD
# Velocity The magnitude of the player’s forward velocity. - 0.1
# Termination A penalty, equal to −1 if the player is on the ground - 0.5
# Upright 0 if the robot is upside down or if the tilt angle is greater
    # than 0.4 radians. Increases linearly, and is equal to +1 if the
    # tilt angle is less than 0.2 radians. - 0.02
# Joint torque A penalty, equal to the magnitude of the torque measured at
    # the player’s knees. This discourages the player from learning
    # gaits which cause high forces on the knees, for example
    # during ground impacts, which can damage a physical robot. - 0.01