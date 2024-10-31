from stable_baselines3 import PPO
from airport_env_V5_2 import AirEnv,FlattenDictWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

#test_model用训练好的模型测试环境；test_random随机动作选取测试环境；train进行训练；check检查环境
mode='train'

env=FlattenDictWrapper(AirEnv())
if mode=='check':
    check_env(env,skip_render_check=True,warn=True)
elif mode=='train':
    env=Monitor(env,'./airport/mixed_fleet/logs/')
    #callback = DistanceLoggerCallback(log_dir='./airport/mixed_fleet/logs/')
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=600000,progress_bar=True)
    model.save('./airport/mixed_fleet/ppo_airport_simple_version_5_2')
elif mode=='test_model' or mode=='test_random':
    model=PPO.load('./airport/mixed_fleet/ppo_airport_simple_version_5_1')
    done=False
    obs,info=env.reset()
    while done==False:
        if mode=='test_random':
            action=env.action_space.sample()
        else:
            action = model.predict(obs)
        obs,reward,done,truncated,info=env.step(action)