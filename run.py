from stable_baselines3 import PPO
from airport_env_V3 import AirEnv,FlattenDictWrapper,DistanceLoggerCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


'''env=FlattenDictWrapper(AirEnv())
check_env(env,skip_render_check=True,warn=True)'''
env=FlattenDictWrapper(AirEnv())
env=Monitor(env,'./airport/mixed_fleet/logs/')
callback = DistanceLoggerCallback(log_dir='./airport/mixed_fleet/logs/')
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=2000000,progress_bar=True,callback=callback)
model.save('./airport/mixed_fleet/ppo_airport_simple_version_3_3')


'''done=False
while done==False:
    action=env.action_space.sample()
    print("action:",action)
    obs,reward,done,truncated,info=env.step(action)
    print("reward:",reward)'''
