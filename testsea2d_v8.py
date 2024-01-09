import gymnasium as gym
from gym import spaces
import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN
import time
from ray.tune.registry import register_env
from trainsea2dv8multi13 import MyModel
from ray.rllib.models import ModelCatalog
import time

from gym.envs.sea.sea2dv7mt import sea2dv7mt
from gym.envs.sea.sea2dv8mt import sea2dv8mt
import cv2
from ray.rllib.models.preprocessors import get_preprocessor
import extest2
import json



#generateMaps.outmap()##随机生成地图
map = cv2.imread("./processedMaps/test.png", cv2.IMREAD_GRAYSCALE)
#map = cv2.imread("./generatedMaps/convex.png", cv2.IMREAD_GRAYSCALE)
map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE);
map = cv2.flip(map, 0)
# cv2.imshow("mat",map)

env = sea2dv8mt(
    rendermode="human", obstacle_map=map, num_agents=8, resettar=True,randmap=True,
    natapos=extest2.agent_pos, maxstp=500, endrate=0.90, finichecknum=30, endfois=135, finisavoirnum=34,
   target_dict=extest2.target_dict, ctarget_dict=extest2.ctarget_dict)  # 注册环境extest2.ctarget_dict，extest2.target_dict
register_env("sea2d-v7mt", lambda config: env)
# 注册预处理器
prep = get_preprocessor(env.observation_space)


def create_env(env_config):
    return env


ray.init()
ModelCatalog.register_custom_model("my_model", MyModel)
config = PPOConfig().framework("torch")


def policy_mapping_fn(agent_id: str, observation: dict, **kwargs):
    return "policy_0"


config = config.update_from_dict(
    {  # 'model': {'fcnet_hiddens': [128, 128]},
        # "fcnet_activation": "relu",
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "framework": "torch",
        "entropy_coeff": 0,
        "explore": False,

        "env": "sea2d-v6m",
        "multiagent": {
            "multiagent_observations": True,
            "policy_mapping_fn": policy_mapping_fn,
            "policies": {
                "policy_0": (None, env.observation_space["agent_0"], env.action_space["agent_0"], {
                    "model": {
                        "custom_model": "my_model",
                        "custom_model_config": {},
                    },

                }),
            },
        },

        # "preprocessors": {"state": "rllib", "obs":DictFlatteningPreprocessor},

    }
)
config = config.resources(num_gpus=0)
config = config.training(gamma=0.93, lr=1e-5, train_batch_size=1000)
config = config.environment('sea2d-v7mt')
config.explore = False
explore_config = {
    "type": "StochasticSampling",
    # "random_timesteps":0 ,
}

config.exploration(explore=False, exploration_config=explore_config)
checkpoint_path = r"D:\Projects\IWannaCup\9_10nuit_train\最新程序x银河麒麟x操作指北\checkpoint_009400"  # 模型路径和checkpoint编号#2650,3200#29003050
# checkpoint_path = "./checkpoint_000900"

agent = PPOTrainer(config=config)
agent.restore(checkpoint_path)
po0 = agent.get_policy("policy_0")
po0.export_model("./tmp/my_nn_model")
# policies = agent.get_policy("policy_0")
# print(policies.model)

arguments = {}


episode_reward = 0;
action = {}
obs, info = env.reset();
done = {}
print('info:{0}'.format(info))
done['__all__'] = False;
stp = 0
ag_reward = {}
targetDetectedlist=[]
def rl_action(sensorList):
    global obs,stp,targetDetectedlist
    lonlat=[[0,0]]*env.num_agents
    for sensor in sensorList:#加入新的目标
        ids=sensor["id"]
        if ids not in targetDetectedlist:
            targetDetectedlist.append(ids)
            xy=[int(np.around(extest2.coord2xy(sensor["lon"],sensor["lat"]))[0]),int(np.around(extest2.coord2xy(sensor["lon"],sensor["lat"]))[1])]
            tmpDict={'id': ids,'coord':xy,'angle': (sensor["angle"][0]+sensor["angle"][1])/2,
                    "restCheck":sensor["restCheck"],'controler_fois': 0,'savoir': False, 'en': True,
                    "restTime":sensor["restTime"],"threatRadius":sensor["threatRadius"],"certainRadius":sensor["certainRadius"]}
            env.target_dict.append(tmpDict)
            print("sssss\n",env.target_dict)
    for i in range(env.num_agents):
            action[f"agent_{i}"] = agent.compute_single_action(obs[f"agent_{i}"], policy_id="policy_0", explore=False)
        # action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action);
    stp += 1
    # 显示五个通道,opncv配置不上时可选择注释掉以下循环内容
    for i in range(5):
        c = obs["agent_0"][:, :, i] * 100
        cv2.imshow(f"c{i}", c)
        cv2.imwrite(f"./img/c{i}.png", c)
    # print('reward = {0},action={1},stp={2}; info:{3}'.format(reward,action, stp,info))
    print('steps ={0};'.format(stp))
    # print('action = {0}; '.format(action))
    # print('reward = {0};'.format( reward))
    print('info:{0}'.format(info))
    print(action)
    env.render()
    for i in range(env.num_agents):
        lonlat[i]=extest2.xy2coord(info[f"agent_{i}"]['road'][0],info[f"agent_{i}"]['road'][1])
    return lonlat

if __name__ == '__main__':
    while not done['__all__']:
        for i in range(env.num_agents):
            action[f"agent_{i}"] = agent.compute_single_action(obs[f"agent_{i}"], policy_id="policy_0", explore=False)
        # action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action);
        stp += 1
        # 显示五个通道,opncv配置不上时可选择注释掉以下循环内容
        # for i in range(5):
        #     c = obs["agent_0"][:, :, i] * 100
        #     cv2.imshow(f"c{i}", c)
        #     cv2.imwrite(f"./img/c{i}.png", c)
        # print('reward = {0},action={1},stp={2}; info:{3}'.format(reward,action, stp,info))
        print('steps ={0};'.format(stp))
        # print('action = {0}; '.format(action))
        # print('reward = {0};'.format( reward))
        print('info:{0}'.format(info))
        print()
        # episode_reward+=reward

        env.render()
        #time.sleep(2)
        for key in reward.keys():
            if key in ag_reward:
                ag_reward[key] += reward[key]
            else:
                ag_reward[key] = reward[key]
        if done["__all__"]:
            print('terminated')
            break
    total_reward = sum(ag_reward.values())
    print(f'agents_reward:', ag_reward)
    print(f'Total_reward=:', total_reward)

        

    env.close()
