from gym.envs.registration import register

register(
    id='magneto_env-v0',
    entry_point='magneto_gym.envs:MagnetoEnv',
    max_episode_steps=200,
)
