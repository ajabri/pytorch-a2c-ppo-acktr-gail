from mjrl.utils.gym_env import GymEnv
from a2c_ppo_acktr.wrappers import *
import mj_envs

e = GymEnv('pen-v0')
obs_interval, predict_interval, no_op = 1, 1, False
test_e = AsyncWrapper(e, obs_interval=obs_interval, predict_interval=predict_interval, no_op=no_op, no_op_action = None)

obs = test_e.reset()
done = False
step = 0
# while step < env._horizon and done == False:
# while not done:
for _ in range(200):
    step += 1
    action = e.action_space.sample()
    mask = np.zeros((1))
    real_act = np.concatenate((mask, action))
    obs, r, done, info = test_e.step(real_act)
    print(step, r, done, info)


# import gym
# from gym.envs.registration import EnvSpec
# import pybullet_envs
# import vizdoomgym
# import vizdoom.vizdoom as vzd
# env = gym.make("VizdoomDefendLine-v0")
# x = env.game.get_game_variable(vzd.GameVariable.AMMO2)
