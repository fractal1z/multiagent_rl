import gym
import time
import cv2
# from gym.envs.classic_control.sea.sea2dv6m import sea2dv6m
# from gym.envs.classic_control.sea.sea2dv7mt import sea2dv7mt
from gym.envs.sea.sea2dv7mt import sea2dv7mt
# from gym.envs.classic_control.sea.sea2dv7mtt import sea2dv7mtt
# from gym.envs.classic_control.sea.sea2dv7mtc import sea2dv7mtc
import extest2

map = cv2.imread("processedMaps/test.png", cv2.IMREAD_GRAYSCALE)
map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
map = cv2.flip(map, 0)
env = sea2dv7mt(
    rendermode="human", obstacle_map=map, num_agents=8,
    natapos=extest2.agent_pos, maxstp=800, endrate=0.9,
    target_dict=extest2.target_dict, ctarget_dict=extest2.ctarget_dict)  # gym.make("sea2d-v6m",rendermode="0")

obs, _ = env.reset()
done = {}
done['__all__'] = False
total_reward = {}
stps = 0
start_time = time.time()
while not done['__all__']:
    env.render()
    stps += 1
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    # opncv配置不上时可选择注释掉以下循环内容
    for i in range(5):
        c = obs["agent_0"][:, :, i]*100
        cv2.imshow(f"c{i}", c)

    print(done)
    print('steps ={0};'.format(stps))
    print('action = {0}; '.format(action))
    print('reward = {0};'.format(reward))
    print('info:{0}'.format(info))
    print()
    # time.sleep(0.05)
    for key in reward.keys():
        if key in total_reward:
            total_reward[key] += reward[key]
        else:
            total_reward[key] = reward[key]

end_time = time.time();
run_time = end_time - start_time
print('time={:.2f}s'.format(run_time))
print(f'Total reward:')
print(f'Total steps: {stps}')
env.close()
