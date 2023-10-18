from gym.envs.registration import register

register(
    id='MagnetoWorld-v0',
    entry_point='magneto_gym.envs:MagnetoEnv',
    max_episode_steps=300,
)
