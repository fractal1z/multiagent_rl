import gymnasium as gym
from gym import spaces
import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN
import time
from ray.tune.registry import register_env
from trainsea2dv7multi13 import MyModel
from ray.rllib.models import ModelCatalog
import time

from gym.envs.sea.sea2dv7mt import sea2dv7mt

import cv2
from ray.rllib.models.preprocessors import get_preprocessor
import extest2
import json

raw = {
    "id": 1,
    "method": "notice-event",
    "content": {
        "arguments": {
            "vesPostion": {
                "101": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "102": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "103": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "104": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "105": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "106": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "107": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                },
                "108": {
                    "coord": [
                        121.64994868,
                        38.82776073
                    ],
                    "spd": 10,
                    "course": 123
                }
            },
            "road": [
                {
                    "id": "101",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "102",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "103",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "104",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "105",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "106",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "107",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                },
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "108",
                    "path": [
                        {
                            "shape": "LineString",
                            "points": [
                                {
                                    "coord": [121, 22],
                                    "spd": 10
                                }

                            ]
                        }
                    ]
                }
            ]
        }
    }
}
map = cv2.imread("./processedMaps/test.png", cv2.IMREAD_GRAYSCALE)
map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE);
map = cv2.flip(map, 0)
# cv2.imshow("mat",map)

env = sea2dv7mt(
    rendermode="human", obstacle_map=map, num_agents=8, resettar=False,
    natapos=extest2.agent_pos, maxstp=400, endrate=0.90, finichecknum=30, endfois=135, finisavoirnum=35,
    target_dict=extest2.target_dict, ctarget_dict=extest2.ctarget_dict)  # 注册环境
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
config = config.training(gamma=0.99, lr=1e-5, train_batch_size=1000)
config = config.environment('sea2d-v7mt')
config.explore = False
explore_config = {
    "type": "StochasticSampling",
    # "random_timesteps":0 ,
}

config.exploration(explore=False, exploration_config=explore_config)
checkpoint_path = "./checkpoint_003250"  # 模型路径和checkpoint编号#2650,3200#29003050
# checkpoint_path = "./checkpoint_000900"

agent = PPOTrainer(config=config)
agent.restore(checkpoint_path)
po0 = agent.get_policy("policy_0")
po0.export_model("./tmp/my_nn_model")
# policies = agent.get_policy("policy_0")
# print(policies.model)

arguments = {}

fl = True
while fl:
    fl = False
    episode_reward = 0;
    action = {}
    obs, info = env.reset();
    done = {}
    print('info:{0}'.format(info))
    for t in range(max(info["init_time"])):
        print(t)
        file_path = f"./dataset_coord/{str(t)}.json"
        for i in range(env.num_agents):
            if t < info["init_time"][i]:
                raw["content"]["arguments"]["vesPostion"][f"10{i + 1}"] = {
                    "coord": extest2.xy2coord(info['init_pos'][f"agent_{i}"]["pos"][t][0],
                                              info['init_pos'][f"agent_{i}"]["pos"][t][1]),
                    "spd": 20,
                    "course": 90
                }
                raw["content"]["arguments"]["road"][i]["path"][0]["points"] = {
                    "coord": extest2.xy2coord(info['init_pos'][f"agent_{i}"]["road"][t][0],
                                              info['init_pos'][f"agent_{i}"]["road"][t][1]),
                    "spd": 20}
        # 将数据字典写入 JSON 文件
        with open(file_path, 'w') as file:
            json.dump(raw, file, indent=4)

    done['__all__'] = False;
    stp = 0
    ag_reward = {}
    while not done['__all__']:
        for i in range(env.num_agents):
            action[f"agent_{i}"] = agent.compute_single_action(obs[f"agent_{i}"], policy_id="policy_0", explore=False)
        # action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action);
        stp += 1;
        t += 1
        file_path = f"./dataset_coord/{str(t)}.json"
        for i in range(env.num_agents):
            raw["content"]["arguments"]["vesPostion"][f"10{i + 1}"] = {
                "coord": extest2.xy2coord(info[f"agent_{i}"]["pos"][0], info[f"agent_{i}"]["pos"][1]),
                # [info[f"agent_{i}"]["pos"][0],info[f"agent_{i}"]["pos"][1]],
                "spd": info[f"agent_{i}"]["spd"],
                "course": info[f"agent_{i}"]["course"]
            }
            raw["content"]["arguments"]["road"][i]["path"][0]["points"] = {
                "coord": extest2.xy2coord(info[f"agent_{i}"]["road"][0], info[f"agent_{i}"]["road"][1]),
                # [info[f"agent_{i}"]["road"][0],info[f"agent_{i}"]["road"][1]],#
                "spd": info[f"agent_{i}"]["spd"]}
        # 将数据字典写入 JSON 文件
        with open(file_path, 'w') as file:
            json.dump(raw, file, indent=4)
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
        print()
        # episode_reward+=reward

        # env.render()
        # time.sleep(0.02)
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

    time.sleep(4)

env.close()
