import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib import agents
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
import matplotlib.pyplot as plt
import time
# from gym.envs.classic_control.sea.sea2dv6m import sea2dv6m
import os
from gym.envs.sea.sea2dv8mt import sea2dv8mt
import cv2
from ray.rllib.models.preprocessors import get_preprocessor
import extest2
from datetime import datetime

map = cv2.imread("processedMaps/test.png", cv2.IMREAD_GRAYSCALE)
map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
map = cv2.flip(map, 0)
# cv2.imshow("mat",map)
torch, nn = try_import_torch()

env = sea2dv8mt(  # 1,1,1.5
    rendermode="0", obstacle_map=map, num_agents=8, resettar=True, randmap=True,
    natapos=extest2.agent_pos, maxstp=500, endrate=0.85, finichecknum=30, endfois=120, finisavoirnum=34,
    target_dict=extest2.target_dict, ctarget_dict=extest2.ctarget_dict)  # gym.make("sea2d-v6m",rendermode="0")
# print(print(type(env).__bases__))
register_env("sea2d-v8mt", lambda config: env)


# 注册预处理器

class MyModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.features1 = nn.Sequential(
            # nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(5, 16, kernel_size=(8, 6), stride=(4, 3), padding=0),  # 49,49
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=6, stride=3, padding=0),  # 16,16
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # 8,8
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 256, kernel_size=(8, 8), stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.features2 = nn.Sequential(
            # nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(5, 16, kernel_size=(8, 6), stride=(4, 3), padding=0),  # 49,49
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=6, stride=3, padding=0),  # 16,16
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # 8,8
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 256, kernel_size=(8, 8), stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.a = nn.Sequential(
            self.features1,
            nn.Flatten(),
            # nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 128),
            # # nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            # # nn.Linear(512, 256),
            # # nn.BatchNorm1d(256),
            nn.Linear(256, num_outputs),
            # nn.Softmax(dim=-1)
            # nn.Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))
        )
        self.c = nn.Sequential(
            self.features2,
            # nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            nn.Flatten(),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            # nn.Linear(128*5*7, 256),
            # #nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # # nn.Linear(256, 256),
            # # nn.BatchNorm1d(256),
            # # nn.LeakyReLU(),
            # # nn.Linear(512, 256),
            # # nn.BatchNorm1d(256),
            nn.Linear(256, 1)
            # nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        )

        self.obs = None

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict["obs"].shape)
        self.obs = input_dict["obs"].float().permute(0, 3, 2, 1)

        # torch.flatten(obs, start_dim=1)
        action_logits = self.a(self.obs)
        action_logits = torch.flatten(action_logits, 1)
        # print(action_logits)
        return action_logits, state

    def value_function(self):
        # torch.flatten(obs, start_dim=1)
        c = self.c(self.obs)
        c = torch.flatten(c, 0)
        # print(c)
        return c


if __name__ == '__main__':
    ModelCatalog.register_custom_model("my_model", MyModel)
    ray.init()

    config = PPOConfig()


    def policy_mapping_fn(agent_id: str, observation: dict, **kwargs):
        return "policy_0"


    config = config.update_from_dict({
        "num_gpus": 1,
        "num_workers": 32,
        "num_envs_per_worker": 1,
        "env": "sea2d-v8mt",
        "multiagent": {
            "multiagent_observations": True,
            "policy_mapping_fn": policy_mapping_fn,
            "policies": {
                "policy_0": (None, env.observation_space["agent_0"], env.action_space["agent_0"], {
                    "model": {
                        "custom_model": "my_model",
                        "custom_model_config": {},
                    },
                    "lr": 5e-5,
                    "gamma": 0.93,
                    "clip_param": 0.15,
                    "use_gae": True,
                    "framework": "torch",
                    "entropy_coeff": 0.03,

                }),
            },
        },
        "lr": 5e-5,
        "gamma": 0.93,
        "clip_param": 0.15,
        "use_gae": True,
        "framework": "torch",
        "entropy_coeff": 0.03,

        "framework": "torch",
        "train_batch_size": 2048,
        "rollout_fragment_length": 8,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 4,
    })
    explore_config = {
        "type": "StochasticSampling",
        # "random_timesteps":0 ,
    }
    config.exploration(explore=True, exploration_config=explore_config)
    config.environment(disable_env_checking=True)

    print(config.to_dict())
    trainer = PPOTrainer(config=config)
    # checkpoint_path = "./mappo_t15/checkpoint_009400"  # 模型路径和checkpoint编号
    # trainer.restore(checkpoint_path)

    # checkpoint_path = trainer.save(checkpoint_dir="./mappo_13")
    # po0=trainer.get_policy("policy_0")
    # po0.export_model("./tmp/my_nn_model")
    # print(po0)

    

    fig_path="./fig_mappo_txx"
    checkpoint_path="./checkpoint_mappo_txx"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print("Folder created: ./fig_mappo_txx")
    with open(fig_path+'/log.txt', 'a') as file:
        file.write('=' * 30 +'#' *20 + '=' * 30+'\n')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间并格式化
        file.write('Current time: ' + current_time + '\n')
    count = 0
    episode_rewards = []
    while True:
        count += 1
        start_time = time.time()
        results = trainer.train()
        episode_rewards.append(results['episode_reward_mean'])

        if results['episode_reward_mean'] >= 10000000 or count >= 84000:
            break
        end_time = time.time()
        run_time = end_time - start_time

        print()
        print('count=', count)
        print('episode total=', results['episodes_total'])
        print('timesteps total=', results['timesteps_total'])
        print('episode_reward_mean=', results['episode_reward_mean'])
        print('count train time={:.2f}s'.format(run_time))
        # 打开文本文件，以追加模式写入
        with open(fig_path+'/log.txt', 'a') as file:  # 写入信息
            file.write('\n')
            file.write('count=' + str(count) + '\n')
            file.write('episode total=' + str(results['episodes_total']) + '\n')
            file.write('timesteps total=' + str(results['timesteps_total']) + '\n')
            file.write('episode_reward_mean=' + str(results['episode_reward_mean']) + '\n')
            file.write(f'count train time = {run_time:.2f}s\n')
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间并格式化
            file.write('current time: ' + current_time + '\n')
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
        plt.xlabel('Training Episode')
        plt.ylabel('Average Reward')
        plt.title('Reward over Training')
        if count % 50 == 0:
            plt.savefig(fig_path+'/fig{}'.format(count))
            checkpoint_path = trainer.save(checkpoint_dir=checkpoint_path)

    print('=' * 20)
    print('episode total=', results['episodes_total'])
    print('timesteps total=', results['timesteps_total'])
    print('count=', count)
    checkpoint_path = trainer.save(checkpoint_dir=checkpoint_path)
    with open("episode_rewards.txt", "w") as f:
        for reward in episode_rewards:
            f.write(str(reward) + "\n")
    plt.show()

    ray.shutdown()
