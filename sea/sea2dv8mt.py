import pygame
import numpy as np
import random
#import gymnasium
#import gymnasium as gym
from gym import spaces
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import cv2
'''
  ____多智能体区域覆盖和目标探测环境-v7____
 2023/4/8   __v1__
    实现10*10小地图带障碍的全覆盖，测试强化学习算法dqn，ppo
 2023/4/15   __v2__
    范围扩大到20*20,加入凹形目标的探测,实现探测覆盖同时进行，测试强化学习算法dqn，ppo
 2023/4/20   __v3__
    范围扩大到160*120，对应4*3海里，栅格大小50m，智能体探测范围3
    加入self._oeil(),实现智能体探测范围视野获取和智能体已知信息地图迷雾的实现，
    self.obs智能体已知信息,self.grid全局信息
 2023/4/26   __v4__
    加入self._tar_update,实现目标的放置与更新
    修改render函数，使用pygame进行可视化
 2023/5/3   __v5__
    重新写了reset，step,_oeil  等，加快了运行速度
 2023/5/5   __v6__
    修改为多智能体环境，加入8个智能体，加入地图读取,灰度图，待探查区域用255表示，边界为0，南沙美济岛地图
    修改了self._oeil(),step，reset，_tar_update，实现环境中多智能体状态空间和观测空间和算法的交互
    加入_get_closest_agent_distances函数，计算智能体距离防止进入威胁区
 2023/5/10   __v7__
    修改地图为挑战杯科目测试版，大小150*200，加入使用地理信息库读取经纬度并转换成地图程序extest1，接下来都在v7基础上修改
    修改了self._oeil(),step，reset，_tar_update，更改了奖励函数
    增加了targets_reset函数，配合is_valid_position，随机生成符合边界距离要求和目标距离要求的25个目标
    将_get_observation函数独立出去，对每个智能体使用当前已知信息提取多通道图片作为状态空间
 2023/9/7   __v8__ 
    增加了重访问时间，随机地图生成   
 使用示例
 from xxxxxxxxxx import sea2dv7mt
 env = sea2dv7mt(
    rendermode="0",obstacle_map=map,num_agents=8,resettar=True,
    natapos=extest1.agent_pos,maxstp=400, endrate=0.85,finichecknum=20,
    target_dict=extest1.target_dict,ctarget_dict=extest1.ctarget_dict)
'''
class pos:#环境内一个表示智能体编号位置的类
    def __init__(self):
        self.number =None#智能体编号的类
        self.x = None#智能体位置x
        self.y = None#智能体位置y

class sea2dv8mt(MultiAgentEnv):#主环境类
    #类实例化参数
    def __init__(self, 
        width=150, height=200,#地图宽高
        target_dict=[],ctarget_dict=[],#目标和可变目标信息，以元素为字典的列表输入
        obstacle_list = [],#已弃用，障碍物列表
        obstacle_map=None,#障碍物地图，灰度图，待探查区域用255表示，边界为0，
        rendermode="0",#是否使用render可视化，rendermode="human"使用可视化，训练时关闭否则会弹窗可视化窗口
        num_agents=8,#智能体数目
        maxstp=400,#最大时间步长，一个时间步在现实中是4.86s（400步32.4分钟）
        natapos=[],#已弃用,智能体出生位置
        endrate=0.95,#目标覆盖率
        finichecknum=30,#探查目标个数
        finisavoirnum=35,#找到目标数
        endfois=85,#目标有效探测次数
        resettar=False,#是否随机设置目标位置（训练时用，测试时用确定的地图
        randmap=False
        ):

        super().__init__()

        #类内参数
        self.width = width#地图宽度x
        self.height = height#地图高度y
        self.num_agents=num_agents#智能体数量
        self.maxstp=maxstp#设置最大步数
        self.natapos=natapos#出生点位置
        self.resettar=resettar#设置是否随机设置目标位置（训练时用，测试时用确定的地图
        self.rendermode=rendermode#设置是否使用可视化
        self.endrate=endrate#设置目标覆盖率
        self.finichecknum=finichecknum#设置目标探查目标个数
        self.finisavoirnum=finisavoirnum#找到目标数
        self.endfois=endfois#设置目标有效探测次数
        self.droitefois=0#当前有效探测次数
        self.action_space = spaces.Dict(#多智能体动作空间，每个智能体4个动作
    {f"agent_{i}":spaces.Discrete(4) for i in range(self.num_agents)})
        self.observation_space =spaces.Dict(#多智能体状态空间，5通道图像
    {f"agent_{i}":spaces.Box(low=0, high=255, shape=(self.width, self.height,5), dtype=np.float32) for i in range(self.num_agents)})
        
        #不同的栅格占据编码self.grid
        self.ag_num=range(10,10+self.num_agents)#0号智能体位置编码
        self.check=3 #凹形区凹口兴趣点
        self.obst=2 #障碍
        self.free=0#自由海域
        self.pied=1#智能体走过的位置（已弃用）
        self.safer=2#智能体安全距离（100m）
        self.tarj_0=5#待探查目标
        self.tarj_1=6#已探查目标
        self.mys=8#未知区域
        #目标，障碍设置
        self.obstacle_map = obstacle_map  # 输入障碍位置（直接输入二值图像）
        self.step_count=0#计数器，计算当前时间步，超过一定时间给出惩罚并结束
        #self.obstacle_list=obstacle_list#输入障碍位置（已弃用，直接输入二值图像）
        self.resttarget_dict=copy.deepcopy(target_dict)#开辟内存空间保存初始目标状态列表，reset时使用防止被覆盖
        self.resctarget_dict=copy.deepcopy(ctarget_dict)#可变目标列表，程序中不修改，只使用编号，角度和变化后角度
        self.acp=[-1]*(self.num_agents)#上次动作，判断智能体是否转向，掉头
        self.agfeaturelist=[(0,-1),(0,1),(-1,0),(1,0)]#智能体特征增强+
        self.check_list=[(0,-3),(1,-3),(2,-3),(3,-3),(3,-2),(3,-1),(3,0),#角度n*15转变为智能体距离150m的格子
                        (3,1),(3,2),(3,3),(2,3),(1,3),(0,3),
                        (-1,3),(-2,3),(-3,3),(-3,2),(-3,1),(-3,0),
                        (-3,-1),(-3,-2),(-3,-3),(-2,-3),(-1,-3),(0,-3)]
        self.resttime_ves_tars=np.zeros((100, 100, 1), dtype=np.int32)#重访问时间矩阵
        self.randmap=randmap
        #屏幕显示相关参数
        self.showwindowhight=850#窗口高度，对不同分辨率和大小的屏幕需要修改以适应
        self.scale = int(self.showwindowhight/self.height);#格子显示像素大小
        self.screen_width = self.width * self.scale#显示宽度和高度
        self.screen_height = self.height * self.scale#显示宽度和高度
        if rendermode=="human":#防止训练时弹窗，只在手动设置显示才建立窗口
            pygame.init();pygame.display.init()
            global screen#这里不能用self.screen，因为Surface对象包含C语言指针，ray进程间没法传递（坑1），所以用全局变量
            global font
            screen= pygame.display.set_mode((self.screen_width, self.screen_height))#窗口
            font = pygame.font.SysFont('Arial', 24,bold=False)#字体

        #初始化
        self.reset()




    def _get_info(self):#info必须是一个字典，对于没有的情况进行占位
        return {
            "distance": 1
        }
    



    def _get_observation(self):#已知信息提取为多通道图像
        observation = [None]*(self.num_agents)
        
        for i in range(self.num_agents):
            sag_channel=np.zeros((self.width, self.height, 1), dtype=np.uint8)
            ag_channel=np.zeros((self.width, self.height, 1), dtype=np.uint8)

            for j in range(self.num_agents):
                if i==j:
                    continue
                ag_channel[self.agentpos[j].x, self.agentpos[j].y] = 2
                for x,y in self.agfeaturelist:
                    ax=self.agentpos[j].x+x;ay=self.agentpos[j].y+y
                    if (ax>=0 and ax<self.width)and(ay>=0 and ay<self.height):
                        ag_channel[ax, ay] = 2

            sag_channel[self.agentpos[i].x, self.agentpos[i].y] = 2
            for x,y in self.agfeaturelist:
                ax=self.agentpos[i].x+x;ay=self.agentpos[i].y+y
                if (ax>=0 and ax<self.width)and(ay>=0 and ay<self.height):
                    sag_channel[ax, ay] = 2
            
            mys_channel=np.where(self.obs == self.mys, 0, 1).astype(np.uint8)
            
            obs_channel=np.where(self.obs == self.obst, 0, 1).astype(np.uint8)

            tar_channel=np.zeros((self.width, self.height, 1), dtype=np.uint8)
            tarmask1=(self.obs == self.check)
            rows, cols ,_= np.where(tarmask1)
            for row, col  in zip(rows, cols):
                tar_channel[row-1:row+2, col-1:col+2] = 1
            tar_channel[tarmask1]=2
            for tar in self.target_dict:#遮掩在重访问时间的智能体对当前目标的视角
                if self.resttime_ves_tars[i][tar["id"]-1]>0:
                    tar_channel[tar["coord"][0]-4:tar["coord"][0]+5, tar["coord"][1]-4:tar["coord"][1]+5] = 0

            
            
            observation[i]= np.concatenate((sag_channel,ag_channel,mys_channel,obs_channel,tar_channel),axis=2)#当前智能体状态空间
        return observation
    



    def _oeil(self):#视野函数，智能体获取探查范围内的信息
        obs_range = 6#探测距离
        num=[0]*(self.num_agents)#这一步新看到的栅格数量，作为返回值计算奖励
        for i in range(self.num_agents):
            obs_x_min = max(0, self.agentpos[i].x-obs_range)#以下四个变量是取出待更新区域，每次不更新整张地图，加快计算速度
            obs_x_max = min(self.width, self.agentpos[i].x+obs_range+1)
            obs_y_min = max(0, self.agentpos[i].y-obs_range)
            obs_y_max = min(self.height, self.agentpos[i].y+obs_range+1)
        
            for x in range(obs_x_min,obs_x_max):
                for y in range(obs_y_min,obs_y_max):
                    distance = ((x-self.agentpos[i].x)**2 + (y-self.agentpos[i].y)**2)**0.5#abs(self.agentpos[i].x-x)+abs(self.agentpos[i].y-y)改曼哈顿距离#((x-self.agentpos[i].x)**2 + (y-self.agentpos[i].y)**2)**0.5
                    if distance <= obs_range+0.2:
                        if self.obs[x,y,:]==self.mys:
                            num[i]+=0.5*1#视野奖励，覆盖区域####################


        for i in range(self.num_agents):
            obs_x_min = max(0, self.agentpos[i].x-obs_range)
            obs_x_max = min(self.width, self.agentpos[i].x+obs_range+1)
            obs_y_min = max(0, self.agentpos[i].y-obs_range)
            obs_y_max = min(self.height, self.agentpos[i].y+obs_range+1)

            for x in range(obs_x_min,obs_x_max):
                for y in range(obs_y_min,obs_y_max):
                    distance = ((x-self.agentpos[i].x)**2 + (y-self.agentpos[i].y)**2)**0.5#abs(self.agentpos[i].x-x)+abs(self.agentpos[i].y-y)改曼哈顿距离#((x-self.agentpos[i].x)**2 + (y-self.agentpos[i].y)**2)**0.5
                    if distance <= obs_range+0.2:
                        self.obs[x,y,:] = self.grid[x,y,:]
                        if self.grid[x,y,:]==self.tarj_0:#找到一个目标
                            self.obs[x-3:x+4,y-3:y+4,:] = self.grid[x-3:x+4,y-3:y+4,:]#获取目标角度信息
                            for tar in self.target_dict: 
                                tarpos=tar["coord"]
                                if tarpos[0]==x and tarpos[1]==y:#确定目标已知
                                    if tar["savoir"]==False:
                                        self.target_dict[tar["id"]-1]["savoir"]=True
                                        num[i]+=80*1#视野奖励，找到目标
                                
        # for i in range(self.num_agents):
        #     for x,y in self.upoeillist:
        #         x=self.agentpos[i].x+x;y=self.agentpos[i].y+y
        #         if (x>=0 and x<self.width)and(y>=0 and y<self.height):
        #             if self.obs[x,y,:]==self.mys:
        #                 num[i]+=1
        # for i in range(self.num_agents):
        #     for x,y in self.upoeillist:
        #         x=self.agentpos[i].x+x;y=self.agentpos[i].y+y
        #         if (x>=0 and x<self.width)and(y>=0 and y<self.height):
        #             self.obs[x,y,:] = self.grid[x,y,:]

        # obs_slice[0]=sum(self.obs[self.agentpos[i].x,0:self.agentpos[i].y]==self.mys)/self.height
        # obs_slice[1]=sum(self.obs[self.agentpos[i].x,self.agentpos[i].y+1:self.height]==self.mys)/self.height
        # obs_slice[2]=sum(self.obs[0:self.agentpos[i].x,self.agentpos[i].y]==self.mys)/self.width
        # obs_slice[3]=sum(self.obs[self.agentpos[i].x+1:self.width,self.agentpos[i].y]==self.mys)/self.width
        # #obs_slice = self.obs[obs_x_min:obs_x_max, obs_y_min:obs_y_max, :]
        # #self.obs[obs_x_min:obs_x_max, obs_y_min:obs_y_max, :] = obs_slice
        # obs_slice=np.insert(obs_slice, 0, self.agentpos[i].y/self.height)
        # obs_slice=np.insert(obs_slice, 0, self.agentpos[i].x/self.width)
        return num




    def _get_closest_agent_distances(self):#智能体之间的距离
        # 初始化一个二维列表，用于存储智能体之间的距离
        distances = [[99999] * self.num_agents for _ in range(self.num_agents)]
        # 遍历每个智能体
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                # 计算智能体i和j之间的距离
                dist = np.sqrt((self.agentpos[i].x - self.agentpos[j].x)**2 + (self.agentpos[i].y - self.agentpos[j].y)**2)
                # 将距离存储在二维列表中
                distances[i][j] = dist
                distances[j][i] = dist
        # 初始化一个列表，用于存储每个智能体距离最近的智能体的距离
        closest_distances = [0] * self.num_agents
        # 遍历二维列表，找到每个智能体距离最近的智能体的距离
        for i in range(self.num_agents):
            closest_distances[i] = min(distances[i])
        return closest_distances    
    



    def _tar_update(self):#更新目标状态，在初始化和智能体走到兴趣点（相当于指定角度探测1次）更新
        i=0;r=[0]*(self.num_agents);
        #print(self.ctarget_dict)
        for i in range(len(self.ctarget_dict)):#检查可变目标
            if self.ctarget_dict[i]['changeCheck']<=self.target_dict[self.ctarget_dict[i]["id"]-1]['controler_fois']:#改变探查角度
                self.target_dict[self.ctarget_dict[i]["id"]-1]['angle']=self.ctarget_dict[i]["angle"]
                self.target_dict[self.ctarget_dict[i]["id"]-1]['controler_fois']=0
                self.target_dict[self.ctarget_dict[i]["id"]-1]['restCheck']=self.ctarget_dict[i]["restCheck"]
                self.ctarget_dict[i]["changeCheck"]=200

        course=[-1]*self.num_agents
       # print("ssssssss",self.target_dict)
        for tar in self.target_dict: 
            tarpos=tar["coord"]
            if tar["controler_fois"]>=tar["restCheck"]:#已经探测wl，跳过
                self.target_dict[tar["id"]-1]["en"]=False
                self.grid[tarpos[0],tarpos[1]]=self.tarj_1#设为已完成探查
                continue
            else:
                self.grid[tarpos[0],tarpos[1]]=self.tarj_0;#未完成探查
            if tar["savoir"]==False:
                continue
        
            self.grid[tarpos[0]-1:tarpos[0]+2,tarpos[1]-1:tarpos[1]+2]=self.obst #障碍物包裹目标，等于威胁区域，撞到惩罚
            self.grid[tarpos[0],tarpos[1]]=self.tarj_0;

            #未在探测的目标，刷新待探查角度，凹形方向
            fl=True
            #x,y=self.check_list[int(tar["angle"]/15)]
            x=int(np.round(3*np.sin(tar["angle"]*np.pi/180)))
            y=int(np.round(-3*np.cos(tar["angle"]*np.pi/180)))
            #x=0;y=3

            for i in range(self.num_agents):#all智能体出当前目标探测范围
                #d=np.sqrt((self.agentpos[i].x - tarpos[0])**2 + (self.agentpos[i].y - tarpos[1])**2)
                if abs(self.agentpos[i].x - tarpos[0])<=3 and abs(self.agentpos[i].y - tarpos[1])<=3:
                    fl=False
                    break
            if fl:
                self.grid[tarpos[0]-3:tarpos[0]+4:,tarpos[1]-3:tarpos[1]+4][self.grid[tarpos[0]-3:tarpos[0]+4:,tarpos[1]-3:tarpos[1]+4]==self.check]=self.free
                self.grid[tarpos[0]+x,tarpos[1]+y]=self.check
                if self.target_dict[tar["id"]-1]["en"]==True:
                    self.toutfois+=1
                    self.target_dict[tar["id"]-1]["en"]=False
            #探测目标，从未探查状态到已探查状态
            
            for i in range(self.num_agents):
                if  (self.agentpos[i].x,self.agentpos[i].y)==(tarpos[0]+x,tarpos[1]+y)and self.target_dict[tar["id"]-1]["en"]==False :
                    if self.resttime_ves_tars[i][tar["id"]-1]<=0:#resttime重访问间隔判断
                        r[i]+=50*1.5;#正确探测奖励######################################################
                        self.resttime_ves_tars[i][tar["id"]-1]=20#重访问间隔更新
                        course[i]=self.target_dict[tar["id"]-1]["angle"]
                        self.target_dict[tar["id"]-1]["controler_fois"]+=1
                        self.target_dict[tar["id"]-1]["en"]=True
                        self.droitefois+=1
            if self.target_dict[tar["id"]-1]["controler_fois"]>=self.target_dict[tar["id"]-1]["restCheck"]:
                self.grid[tarpos[0],tarpos[1]]=self.tarj_1        
            #更新目标点附近
            self.obs[tarpos[0]-3:tarpos[0]+4,tarpos[1]-3:tarpos[1]+4,:] = self.grid[tarpos[0]-3:tarpos[0]+4,tarpos[1]-3:tarpos[1]+4,:]
        # for target in self.target_dict:
        #     print(target)
        return r,course




    def is_valid_position(self,x, y,id):
        min_tar_bound=4#智能体和边界最小距离
        min_tar_tar=13#智能体之间最小距离
        min_tar_ag=4#最小目标与初始智能体距离
        r=True
        # 检查与所有其他目标之间的距离是否大于min_tar_bound
        for target in self.target_dict:
            if id==target["id"]:
                continue
            if ((x - target['coord'][0]) ** 2 + (y - target['coord'][1]) ** 2) ** 0.5 <= min_tar_tar:
                r= False
            
        # 检查目标周围min_tar_bound范围内是否存在边界
        x_min = max(0, x-min_tar_bound)
        x_max = min(self.width, x+min_tar_bound+1)
        y_min = max(0, y-min_tar_bound)
        y_max = min(self.height, y+min_tar_bound+1)
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                if self.grid[i,j]==self.obst:
                    r= False
                
        # 检查目标周围min_tar_ag范围内是否存在智能体      
        x_min = max(0, x-min_tar_ag)
        x_max = min(self.width, x+min_tar_ag+1)
        y_min = max(0, y-min_tar_ag)
        y_max = min(self.height, y+min_tar_ag+1)
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                if self.grid[i,j] in self.ag_num:
                    r= False
                
        return r      
          



    def targets_reset(self):#随机生成目标位置
        x_range = (10, 139)
        y_range = (10, 189)
        angle_choices = [i * 15 for i in range(24)]  # 0-360度以n*15度形式随机选取
        for tar in self.target_dict: 
            self.target_dict[tar["id"]-1]["coord"]=[0,0]
        for tar in self.target_dict: 
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])
            angle = random.choice(angle_choices)
            while not self.is_valid_position(x,y,id=tar["id"]):
                x = random.randint(x_range[0], x_range[1])
                y = random.randint(y_range[0], y_range[1])
            self.target_dict[tar["id"]-1]["coord"]=[x,y]   
    



    def reset(self, seed=None, return_info=False, options=None):#初始化函数
        #super().reset(seed=seed)
        self.step_count=0#参数初始化
        self.droitefois=0
        self.toutfois=0
        self.dangereux_fois=0
        self.resttime_ves_tars=self.resttime_ves_tars-1
        self.grid = np.zeros((self.width, self.height, 1), dtype=np.uint8)
        self.obs= np.zeros((self.width, self.height, 1), dtype=np.uint8)+self.mys
        self.hiting=[True]*(self.num_agents)
        if self.randmap:
            self.obstacle_map=cv2.imread(f"./generatedMaps/convex{random.randint(0, 3999)}.png", cv2.IMREAD_GRAYSCALE)
            self.obstacle_map = cv2.rotate(self.obstacle_map, cv2.ROTATE_90_CLOCKWISE);
            self.obstacle_map = cv2.flip(self.obstacle_map, 0)
        if self.obstacle_map is not None:
            #创建障碍物
            self.grid[self.obstacle_map < 128] = self.obst
            self.obs[self.obstacle_map < 128] = self.obst
        
        self.surface=np.count_nonzero(self.obs != self.obst)
        #出生点0，0
        self.agentpos=[pos() for _ in range(self.num_agents)]
        initstep_time=[0]*self.num_agents;
        initstep={f"agent_{i}":{"pos":[],"road":[],"spd":20,"course":90} for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.agentpos[i].y=self.natapos[i][1];
            self.agentpos[i].number=i;
            k=0 #self.natapos[i][0];
            self.agentpos[i].x=k;
            
            while self.grid[self.agentpos[i].x, self.agentpos[i].y,0] != self.free:
                initstep[f"agent_{i}"]["pos"].append([k,self.agentpos[i].y])
                initstep[f"agent_{i}"]["road"].append([k+1,self.agentpos[i].y])
                self.agentpos[i].x=k;k+=1;initstep_time[i]+=1;
                
      
        #随机出生地
        # while True:
        #     self.agentpos[i].x = random.randint(0, self.width-1)
        #     self.agentpos[i].y = random.randint(0, self.height-1)
        #     if self.grid[self.agentpos[i].x, self.agentpos[i].y] == self.free:  # 如果智能体的初始位置不在障碍物里面，则退出循环
        #         break

        for i in range(self.num_agents):
            self.grid[self.agentpos[i].x, self.agentpos[i].y] = self.ag_num[i]   #agent位置

        #初始化目标
        self.target_dict=copy.deepcopy(self.resttarget_dict)
        self.ctarget_dict=copy.deepcopy(self.resctarget_dict)
        if self.rendermode=="human":
                for target in self.target_dict:
                    print(target)

        if self.resettar:#生成随机目标位置
            self.targets_reset()
        
        self._tar_update()
        r=self._oeil()#更新探测范围内视野
        self._tar_update()
        r=self._oeil()#更新探测范围内视野
        observation=self._get_observation()
        info = {"init_time":initstep_time,
                "init_pos":initstep
        #         **{
        #     f"agent_{i}":{"pos":[self.agentpos[i].x,self.agentpos[i].y],"road":[self.agentpos[i].x,self.agentpos[i].y],"spd":20,"course":90} for i in range(self.num_agents)
        # }
        }

        self.treward=[0]*(self.num_agents);

        observation = {f"agent_{i}": space for i, space in enumerate(observation)}
        
        return observation, info




    def step(self, action):#时间步
        self.resttime_ves_tars=self.resttime_ves_tars-1#重访问时间步
        reward=[-1]*(self.num_agents);self.step_count+=1#原地踏步给小惩罚
        done = {f"agent_{i}": False for i in range(self.num_agents)}
        done['__all__'] = False
        pre_pos=copy.deepcopy(self.agentpos)#重新申请内存,暂存上一步位置
        #print(action)
        for i in range(self.num_agents):
            self.grid[self.agentpos[i].x, self.agentpos[i].y]=self.free;self.obs[self.agentpos[i].x, self.agentpos[i].y]=self.free
            #判断撞墙或者撞边界，撞了reward=-15
            if action[f"agent_{i}"] == 0 and(self.agentpos[i].y <=0 or self.grid[self.agentpos[i].x, self.agentpos[i].y-1] == self.obst): # 向上移动
                reward[i] -= 15
                if self.hiting[i]==False:
                    self.hiting[i]=True
                    self.dangereux_fois+=1#计一次协同失败
            elif action[f"agent_{i}"] == 1 and (self.agentpos[i].y >= self.height-1 or self.grid[self.agentpos[i].x, self.agentpos[i].y+1] == self.obst): # 向下移动
                reward[i] -= 15
                if self.hiting[i]==False:
                    self.hiting[i]=True
                    self.dangereux_fois+=1#计一次协同失败
            elif action[f"agent_{i}"] == 2 and (self.agentpos[i].x <= 0 or self.grid[self.agentpos[i].x-1, self.agentpos[i].y] == self.obst): # 向左移动
                reward[i] -= 15
                if self.hiting[i]==False:
                    self.hiting[i]=True
                    self.dangereux_fois+=1#计一次协同失败
            elif action[f"agent_{i}"] == 3 and (self.agentpos[i].x >= self.width-1 or self.grid[self.agentpos[i].x+1, self.agentpos[i].y] == self.obst): # 向右移动
                reward[i] -= 15
                if self.hiting[i]==False:
                    self.hiting[i]=True
                    self.dangereux_fois+=1#计一次协同失败
            else:#如果不撞边界，移动智能体
                self.hiting[i]=False
                if action[f"agent_{i}"] == 0 and self.agentpos[i].y > 0 and self.grid[self.agentpos[i].x, self.agentpos[i].y-1] != self.obst: # 向上移动
                    self.agentpos[i].y -= 1
                    if self.acp[i]==0:
                        reward[i] +=0.5
                    elif self.acp[i]==1:
                        reward[i] -=0.5
                elif action[f"agent_{i}"] == 1 and self.agentpos[i].y < self.height-1 and self.grid[self.agentpos[i].x, self.agentpos[i].y+1] != self.obst: # 向下移动
                    self.agentpos[i].y += 1
                    if self.acp[i]==1:
                        reward[i] +=0.5
                    elif self.acp[i]==0:
                        reward[i] -=0.5
                elif action[f"agent_{i}"] == 2 and self.agentpos[i].x > 0 and self.grid[self.agentpos[i].x-1, self.agentpos[i].y] != self.obst: # 向左移动
                    self.agentpos[i].x -= 1
                    if self.acp[i]==2:
                        reward[i] +=0.5
                    elif self.acp[i]==3:
                        reward[i] -=0.5
                elif action[f"agent_{i}"] == 3 and self.agentpos[i].x < self.width-1 and self.grid[self.agentpos[i].x+1, self.agentpos[i].y] != self.obst: # 向右移动
                    self.agentpos[i].x += 1
                    if self.acp[i]==3:
                        reward[i] +=0.5
                    elif self.acp[i]==2:
                        reward[i] -=0.5
                self.acp[i]=action[f"agent_{i}"]
            if self.grid[self.agentpos[i].x, self.agentpos[i].y] in self.ag_num:#智能体过近，=-35，撞船了
                reward[i] -= 35;self.agentpos[i]=pre_pos[i]#退回上一步位置

            self.grid[self.agentpos[i].x, self.agentpos[i].y] = self.ag_num[i]#智能体位置放入地图
            #print(self.agentpos[i].number,self.agentpos[i].x,self.agentpos[i].y)
            
        ro=self._oeil()#给出覆盖新区域的奖励,更新可见图
        rc,ang=self._tar_update()

        #判断距离, 小于100m给惩罚
        dis=self._get_closest_agent_distances()
        #print(dis)
        for i in range(self.num_agents):#距离过近
            if dis[i]<=2.12:
                reward[i]-=35
                self.dangereux_fois+=0.5#计一次协同失败

        myssurface=np.count_nonzero(self.obs == self.mys)#覆盖率计算
        rate=1-myssurface/self.surface

        spd=[20]*(self.num_agents)#航速设置
        course=[-1]*(self.num_agents)
        for i in range(self.num_agents):
            if action[f"agent_{i}"] == 0:
                course[i]=0
            if action[f"agent_{i}"] == 1:
                course[i]=180
            if action[f"agent_{i}"] == 2:
                course[i]=270
            if action[f"agent_{i}"] == 3:
                course[i]=90
            
        for i in range(self.num_agents):#奖励计算
            reward[i]+=ro[i]/1+rc[i]/1#*(2-myssurface/19200)
            if rc[i]>0:
                spd[i]=10
                course[i]=ang[i]
    
        allsavoir=0;finicheck=0
        for tar in self.target_dict: #已探查目标数
            if tar["savoir"]==True:
                allsavoir+=1
            if tar["controler_fois"]>=tar["restCheck"]:
                finicheck+=1

        
        for i in range(self.num_agents):#判断结束
            if ((not np.isin(self.obs, self.mys).any() )or(rate>self.endrate)) and self.droitefois>=self.endfois and allsavoir>=self.finisavoirnum:#and #finicheck>=self.finichecknum :#(not np.isin(self.obs, self.check).any() ) :#完成指标，给大奖励
                done[f"agent_{i}"] = True
                done['__all__'] = True
                reward[i] += self.droitefois*2
            if self.step_count>=self.maxstp:#self.width*self.height*1.6:#超时了
                done[f"agent_{i}"] = True
                done['__all__'] = True
                reward[i]-=200-self.droitefois*2#self.step_count/5#-np.count_nonzero(self.obs == self.mys)/200

        observation=self._get_observation()

        info = {'global':{"rate":rate,"right_probtimes":self.droitefois,"find_tars:":allsavoir,"fine_tars":finicheck},
                **{
                f"agent_{i}":{"pos":[pre_pos[i].x,pre_pos[i].y],"road":[self.agentpos[i].x,self.agentpos[i].y],"spd":spd[i],"course":course[i]} for i in range(self.num_agents)
                    }
                }#输出全局信息，覆盖率，已经进行的正确探测次数，已找到目标个数，完成探查的目标数，还有每个智能体位置与航路
        truncated=terminated=done
        if done["__all__"]:
            self.step_count=0
            if self.rendermode=="human":
                print("self.dangereux_fois="+str(self.dangereux_fois))
                print("allsavoir="+str(allsavoir)+"  "+"droitefois="+str(self.droitefois)+"  "+"toutfois="+str(self.toutfois))
                for target in self.target_dict:
                    print(target)
                    

        observation = {f"agent_{i}": space for i, space in enumerate(observation)}
        reward = {f"agent_{i}": space for i, space in enumerate(reward)}

        return  observation, reward, terminated, truncated, info



    def render(self, mode="human"):#可视化
        pygame.event.get()#刷新窗口
        line_thickness = 1 # 线条宽度
        #buffer = pygame.Surface((self.screen_width, self.screen_height))
        #pygame.display.set_caption("Grid World")
        screen.fill((255, 255, 255));
        #print(self.grid.max(),self.grid.min())
        #画出地图
        for x in range(self.width):
            for y in range(self.height):
                color = tuple(0 for _ in range(3))#根据地图编码选择待画像素颜色
                rect = pygame.Rect(x*self.scale, y*self.scale, self.scale, self.scale)
                if self.obs[x,y]==self.free:
                    color = tuple(255 for _ in range(3))
                elif self.obs[x,y]==self.obst:
                    color = tuple(0 for _ in range(3))
                elif self.obs[x,y]==self.pied:
                    color = tuple(100 for _ in range(3))
                elif self.obs[x,y]==self.check:
                    color = (0,255,255)
                elif self.obs[x,y]==self.mys:
                    color = (100,0,100)
                elif self.obs[x,y]==self.tarj_0:
                    color = (150,0,0)
                elif self.obs[x,y]==self.tarj_1:
                    color = (0,150,0)
                pygame.draw.rect(screen, color, rect)#画智能体，一个圆
                if self.grid[x,y] in self.ag_num:
                    x_pos = x*self.scale + int(self.scale/2) # 圆心横坐标
                    y_pos = y*self.scale + int(self.scale/2) # 圆心纵坐标
                    color = (255,255,0)
                    pygame.draw.circle(screen, color, (x_pos, y_pos),int(self.scale/2))
                   


        # 在栅格之间添加一个像素的间隔并画一条线
        for x in range(self.width):
            pygame.draw.line(screen, (0, 0, 0), (x*self.scale, 0), (x*self.scale, self.height*self.scale), line_thickness)
        for y in range(self.height):
            pygame.draw.line(screen, (0, 0, 0), (0, y*self.scale), (self.width*self.scale, y*self.scale), line_thickness)
        for i in range(self.num_agents): # 在圆形智能体的中心位置显示智能体编号
            x=self.agentpos[i].x;y=self.agentpos[i].y;ag_idx=self.agentpos[i].number
            x_pos = x*self.scale + int(self.scale/2)# 圆心横坐标
            y_pos = y*self.scale + int(self.scale/2) # 圆心纵坐标
            text_surface = font.render(str(ag_idx), True, (0, 0, 255))
            text_rect = text_surface.get_rect(center=(x_pos, y_pos))
            screen.blit(text_surface, text_rect)

        #screen.blit(buffer, (0, 0))#将buffer更新到screen中
        #pygame.display.flip()#渲染窗口
        pygame.display.update()


    def close(self):
        pygame.display.quit()
        pygame.quit()