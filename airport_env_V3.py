from gymnasium import Env, spaces
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
import math
import random
import pandas as pd

class AirEnv(Env):
    def __init__(self):
        super(AirEnv, self).__init__() 
        self.state_space_1=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_2=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_3=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_4=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_5=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_6=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_7=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_8=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_9=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_10=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_11=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_12=spaces.Box(low=np.array([0,-1,-2]),high=np.array([100,1,10]),shape=(3,),dtype=np.int32)
        self.state_space_13 = spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0]), high=np.array([2,2,3,3,3,3,3,3,3,3,3,3]),shape=(12,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            'ev_1': self.state_space_1,
            'ev_2': self.state_space_2,
            'ev_3': self.state_space_3,
            'ev_4': self.state_space_4,
            'cv_1': self.state_space_5,
            'cv_2': self.state_space_6,
            'cv_3': self.state_space_7,
            'cv_4': self.state_space_8,
            'ba_1': self.state_space_9,
            'ba_2': self.state_space_10,
            'ba_3': self.state_space_11,
            'ba_4': self.state_space_12,
            'station': self.state_space_13
        })
        self.service_demand=np.zeros(12,dtype=np.int32)
        self.service_demand[2+np.random.randint(10)]=1
        self.service_demand[0],self.service_demand[1]=2,2
        self.state=(np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),self.service_demand)
        self.step_num = 0
        self.max_step=90
        self.charge_rate=0.4
        self.work_rate=0.5
        self.travelling_rate_towering=0.00223
        self.travelling_rate_service=0.00183
        self.travelling_rate_fuel=0.00091
        self.work_rate_fuel=0.0005
        #当前所在节点
        self.current_vertice=np.zeros(12)
        #目标前往节点
        self.next_vertice=np.zeros(12)
        #标记车辆类型
        self.ev_mark=0
        self.cv_mark=1
        self.ba_mark=2
        #充电车辆数,用于记录车辆充电轮数,10分钟为一轮
        self.charging_num=np.zeros(12)
        #工作车辆数,用于记录车辆工作轮数,10分钟为一轮
        self.working_num=np.zeros(12)
        self.distance_array=np.array([[0,565,745,935,1125,1265,1415,1555,1725,1835,1935,2005,2095],
                                        [565,0,180,370,560,700,850,990,1160,1270,1370,1440,1530],
                                        [745,180,0,190,380,520,670,810,980,1090,1190,1260,1350],
                                        [935,370,190,0,190,330,480,620,790,900,1000,1070,1160],
                                        [1125,560,380,190,0,140,290,430,600,710,810,880,970],
                                        [1265,700,520,330,140,0,150,290,460,570,670,740,830],
                                        [1415,850,670,480,290,150,0,140,310,420,520,590,680],
                                        [1555,990,810,620,430,290,140,0,170,280,380,450,540],
                                        [1725,1160,980,790,600,460,310,170,0,110,210,280,370],
                                        [1835,1270,1090,900,710,570,420,280,110,0,100,170,260],
                                        [1935,1370,1190,1000,810,670,520,380,210,100,0,70,160],
                                        [2005,1440,1260,1070,880,740,590,450,280,170,70,0,90],
                                        [2095,1530,1350,1160,970,830,680,540,370,260,160,90,0]])

    def step(self, action):
        distance=np.zeros(12)
        ev_action = action['ev']-1
        cv_action = action['cv']-1
        ba_action = action['ba']-1
        #矫正动作
        reset_ev_action,reset_cv_action,reset_ba_action=self.reset_action(ev_action,cv_action,ba_action,self.step_num,self.state)
        for i in range(0,4):
            if reset_ev_action[i]==1:
                #布置工作车辆(纯电车辆)
                self.state[i][1],self.state[i][2],new_demand,distance[i]=self.work_setup(self.ev_mark,self.state[i][1],self.state[i][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[i]=self.state[i][2]
                #表示该号车开始第一轮的工作
                self.working_num[i]=1
            if reset_ev_action[i]==-1:
                #布置充电车辆
                self.state[i][1],self.state[i][2],new_demand,distance[i]=self.charging_setup(self.state[i][1],self.state[i][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[i]=self.state[i][2]
                #表示该号车开始第一轮的充电
                self.charging_num[i]=1
        
        for j in range(0,4):
            if reset_cv_action[j]==1:
                #布置工作车辆(油车)
                self.state[j+4][1],self.state[j+4][2],new_demand,distance[j+4]=self.work_setup(self.cv_mark,self.state[j+4][1],self.state[j+4][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[j+4]=self.state[j+4][2]
                self.working_num[j+4]=1
            if reset_cv_action[j]==-1:
                #布置加油车辆(油车)
                self.state[j+4][1],self.state[j+4][2],new_demand,distance[j+4]=self.fueling_setup(self.state[j+4][1],self.state[j+4][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[j+4]=self.state[j+4][2]
                self.charging_num[j+4]=1
        
        for l in range(0,4):
            if reset_ba_action[l]==1:
                #布置工作车辆(供水车辆)
                self.state[l+8][1],self.state[l+8][2],new_demand,distance[l+8]=self.work_setup(self.ba_mark,self.state[l+8][1],self.state[l+8][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[l+8]=self.state[l+8][2]
                self.working_num[l+8]=1
            if reset_ba_action[l]==-1:
                #布置充电车辆（供水车辆）
                self.state[l+8][1],self.state[l+8][2],new_demand,distance[l+8]=self.charging_setup(self.state[l+8][1],self.state[l+8][2],self.state[12])
                self.state=self.state[:12]+(new_demand,)
                self.next_vertice[l+8]=self.state[l+8][2]
                self.charging_num[l+8]=1

        for i in range(0,4):
            #计算电量
            self.state[i][0]=self.power_calculation(self.ev_mark,self.state[i],distance[i],self.charging_num[i],self.working_num[i])
            self.state[i+4][0]=self.fuel_calculation(self.cv_mark,self.state[i+4],distance[i+4],self.charging_num[i+4],self.working_num[i+4])
            self.state[i+8][0]=self.power_calculation(self.ba_mark,self.state[i+8],distance[i+8],self.charging_num[i+8],self.working_num[i+8])
        #终止条件：电量低于30%
        terminated=self.end(self.state)
        dis=sum(distance)
        #计算奖励，以电量和移动距离为基础
        reward=self.get_reward(self.state,distance)
        #步数+1
        self.step_num += 1
        #更新任务，将站点状态更新，若这一轮中拖车能完成任务，则将需求改为2，不能则为3
        new_demand=self.task_update(self.state,self.current_vertice,self.next_vertice)
        self.state=self.state[:12]+(new_demand,)
        #更新车辆位置，检查这一轮是否能到达目标站点开始工作，或者能到达的最远的位置
        self.current_vertice=self.vertice_update(self.current_vertice,self.next_vertice)
        #更新工作需求，每6轮更新一次，随机挑选站点工作
        new_demand=self.demand_update(self.state[12],self.step_num)
        self.state=self.state[:12]+(new_demand,)

        for i in range(0,12):
            #更新车辆状态，如果该轮次能完成任务或充电，则将工作状态更改为0，否则为1或-1
            self.state[i][1]=self.state_update(i,self.state[i][1],self.charging_num[i],self.working_num[i],distance[i])
            #更新作业或充电计数轮次，因为作业和充电轮次上限都为2，故而当经上一步判断，若一轮内能够完成，则1变为0，否则1变2，两轮内一定能完成，故而2后变为0
            if self.state[i][1]==0 and self.working_num[i]==1:
                self.working_num[i]=0
            elif self.working_num[i]==2:
                self.working_num[i]=0
            elif self.working_num[i]==1:
                self.working_num[i]=2
            if self.charging_num[i]==2:
                self.charging_num[i]=0
            elif self.charging_num[i]==1:
                self.charging_num[i]=2
        
        if self.step_num%2==1:
            self.state[12][0],self.state[12][1]=2,2
        
        state=dict()
        state['ev_1']=self.state[0]
        state['ev_2']=self.state[1]
        state['ev_3']=self.state[2]
        state['ev_4']=self.state[3]
        state['cv_1']=self.state[4]
        state['cv_2']=self.state[5]
        state['cv_3']=self.state[6]
        state['cv_4']=self.state[7]
        state['ba_1']=self.state[8]
        state['ba_2']=self.state[9]
        state['ba_3']=self.state[10]
        state['ba_4']=self.state[11]
        state['station']=self.state[12]
        info=dict()
        info={'distance':dis}
        truncated = self.step_num >= self.max_step
        return state, reward, terminated, truncated, info
        
    def reset(self,seed=None):
        self.service_demand=np.zeros(12,dtype=np.int32)
        self.service_demand[2+np.random.randint(10)]=1
        self.service_demand[0],self.service_demand[1]=2,2
        self.state=(np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),np.array([100,0,0]),self.service_demand)
        #print("Resetting step number from", self.step_num, "to 0")
        self.step_num=0
        self.charging_num=np.zeros(12)
        self.working_num=np.zeros(12)
        self.current_vertice=np.zeros(12)
        self.next_vertice=np.zeros(12)
        state=dict()
        state['ev_1']=self.state[0]
        state['ev_2']=self.state[1]
        state['ev_3']=self.state[2]
        state['ev_4']=self.state[3]
        state['cv_1']=self.state[4]
        state['cv_2']=self.state[5]
        state['cv_3']=self.state[6]
        state['cv_4']=self.state[7]
        state['ba_1']=self.state[8]
        state['ba_2']=self.state[9]
        state['ba_3']=self.state[10]
        state['ba_4']=self.state[11]
        state['station']=self.state[12]
        info=dict()

        return state,info
        
    def reset_action(self, ev_action, cv_action, ba_action, step_num, state):
        #初始化空闲拖车与空闲供水车（空闲指的是拖车和供水车都没有工作且当前步骤action值为1）
        available_car_towering=8
        available_car_service=4
        #初始化需求的拖车与供水车
        demand_car_towering=0
        demand_car_service=0
        #拖车与供水车的充电需求
        power_demand=0
        #初始化拖车与供水车的电量
        car_id=np.array([100,100,100,100,100,100,100,100,100,100,100,100])
        #获取这一步时的各车辆电量
        car_power_towering=np.array([state[0][0],state[1][0],state[2][0],state[3][0],state[4][0],state[5][0],state[6][0],state[7][0]])
        car_power_serivice=np.array([state[8][0],state[9][0],state[10][0],state[11][0]])
        #计算state中这一步的机场工作需求
        for i in range(2,12):
            if state[12][i]==1:
                demand_car_towering+=1
            elif state[12][i]==2:
                demand_car_service+=1
        #充电设置为至少20分钟，故每两轮进行一次决策，计算充电需求
        if step_num%2==1:
            available_charging_station=4
            for i in range(0,4):
                if ev_action[i]==-1:
                    car_id[i]=state[i][0]
                    power_demand+=1
                if ba_action[i]==-1:
                    power_demand+=1
                    car_id[8+i]=state[i+8][0]
        else:
            #若非偶数轮次，则不允许充电操作
            available_charging_station=0
            for i in range(0,4):
                if ev_action[i]==-1:
                    ev_action[i]=0
                if ba_action[i]==-1:
                    ba_action[i]=0
        #如果充电需求大于空闲充电站，则根据电量排序先让电量较少的去进行充电
        if power_demand>available_charging_station:
            indices=np.argsort(car_id)
            chosen=indices[0:available_charging_station]
            for i in chosen:
                if i <5:
                    ev_action[i]=-2
                else:
                    ba_action[i-8]=-2
        for i in range(0,4):
            if ev_action[i]<0:
                ev_action[i]+=1
            if ba_action[i]<0:
                ba_action[i]+=1
        #如果车辆当前状态不为空闲而动作中有工作或充电需求，动作更改为0，并更新空闲车辆数
        for i in range(0,4):
            if state[i][1]!=0 and ev_action[i]!=0:
                ev_action[i]=0
                available_car_towering-=1
            elif ev_action[i]==0:
                available_car_towering-=1
            if state[i+4][1]!=0 and cv_action[i]!=0:
                cv_action[i]=0
                available_car_towering-=1
            elif cv_action[i]==0:
                available_car_towering-=1
            if state[i+8][1]!=0 and ba_action[i]!=0:
                ba_action[i]=0
                available_car_service-=1
            elif ba_action[i]==0:
                available_car_service-=1
        #计算当前步骤需求工作车辆数和当前步骤空闲车辆数之差
        d_a_t=demand_car_towering-available_car_towering
        d_a_s=demand_car_service-available_car_service
        #在第三轮进行计算，保证拖车任务的完成，如果有任务不能完成，派遣没有工作指令的空余车辆中电量最高的前去工作
        for i in range(0,4):
            if ev_action[i]==1:
                car_power_towering[i]=0
            if cv_action[i]==1:
                car_power_towering[i+4]=0
            if ba_action[i]==1:
                car_power_serivice[i]=0
        if step_num%6==2 and d_a_t>0:
            indices = np.argsort(car_power_towering)
            chosen=indices[8-d_a_t:8]
            for i in chosen:
                if i<4:
                    ev_action[i]=1
                else:
                    cv_action[i-4]=1
            d_a_t=0
        #在第五轮进行计算，保证供水车任务的完成，如果有任务不能完成，派遣没有工作指令的空余车辆中电量最高的前去工作
        if step_num%6==4 and d_a_s>0:
            indices = np.argsort(car_power_serivice)
            chosen=indices[4-d_a_s:4]
            for i in chosen:
                ba_action[i]=1
            d_a_s=0
        return ev_action,cv_action,ba_action
    
    def vertice_update(self, current_vertice, next_vertice):
        for i in range(0,12):
            a=self.convert_state(current_vertice[i])
            b=self.convert_state(next_vertice[i])
            if self.distance_array[int(a),int(b)]<=2000:
                current_vertice[i]=next_vertice[i]
            else:
                if a<1:
                    current_vertice[i]+=10
                elif a>10:
                    current_vertice[i]-=10
        return current_vertice
    
    def demand_update(self,state,step_num):
        demand=np.array([1,1,1,8,4,4,3,8,3,6,5,4,4,3,3])
        if step_num%6==5:
            indices_to_choose = range(2, 12)
            selected_indices = random.sample(indices_to_choose, k=demand[int(((step_num+1)//6)-1)])
            for index in selected_indices:
                state[index] = 1

        return state
    
    def work_setup(self,mark,old_state,old_vertice,demand):
        #布置拖车作业，按照编号顺序布置拖车作业
        if mark==0 or mark==1:
            a=self.convert_state(old_vertice)
            for i in range(2,12):
                if demand[i]==1:
                    old_state=1
                    old_vertice=i-1
                    #代表该站点被占用工作中
                    demand[i]=3
                    break
            b=self.convert_state(old_vertice)
            distance=self.distance_array[int(a),int(b)]
            #返回新的工作状态（如果有），节点状态以及需求更新和移动距离（指一次指令所需要移动的距离，而非10分钟内移动的距离）
            return old_state,old_vertice,demand,distance
        else:
            #供水车同理
            a=self.convert_state(old_vertice)
            for i in range(2,12):
                if demand[i]==2:
                    old_state=1
                    old_vertice=i-1
                    demand[i]=0
                    break
            b=self.convert_state(old_vertice)
            distance=self.distance_array[int(a),int(b)]
            return old_state,old_vertice,demand,distance
        
    def close(self):
        print("DONE")
        
    def render(self,mode='human'):
        pass
        
    def charging_setup(self,old_state,old_vertice,demand):
        #布置充电作业，同工作布置道理相同
        a=self.convert_state(old_vertice)
        for i in range(0,2):
            if demand[i]>0:
                old_state=-1
                old_vertice=-1*i-1
                demand[i]-=1
                break
        b=self.convert_state(old_vertice)
        distance=self.distance_array[int(a),int(b)]
        return old_state,old_vertice,demand,distance
    
    def fueling_setup(self,old_state,old_vertice,demand):
        #布置充电作业，同工作布置道理相同
        a=self.convert_state(old_vertice)
        old_state=-1
        if self.distance_array[int(a),self.convert_state(-1)]<=self.distance_array[int(a),self.convert_state(-2)]:
            old_vertice=-1
        else:
            old_vertice=-2
        b=self.convert_state(old_vertice)
        distance=self.distance_array[int(a),int(b)]
        return old_state,old_vertice,demand,distance


    def convert_state(self,x):
        #为方便计算，不列入距离矩阵，将原来的充电2，充电1，停车场，站点1...顺序改为停车场，站点1，站点2，充电1...（场景构建在PPt中）
        if x>=7:
            x=x+2
        elif x>2 and x<7:
            x=x+1
        elif x==-2:
            x=8
        elif x==-1:
            x=3
        return x
    
    def task_update(self,state,old_vertice,new_vertice):
        #如果拖车在10分钟内能够完成任务，则站点状态改为2表示需要供水车服务，否则改为3为锁定状态
        for i in range(0,8):
            a=self.convert_state(old_vertice[i])
            b=self.convert_state(new_vertice[i])
            if self.distance_array[int(a),int(b)]<=2000 and state[i][1]==1:
                state[12][int(new_vertice[i]+1)]=2
            elif self.distance_array[int(a),int(b)]>2000 and state[i][1]==1:
                state[12][int(new_vertice[i]+1)]=3
        return state[12]
    
    def power_calculation(self,mark,state,distance,charging_num,working_num):
        if mark==0:
            #计算电量
            energy=state[0]
            #计算移动中消耗电量
            energy=energy-math.ceil(distance*self.travelling_rate_towering)
            if state[1]==1 and working_num==1:
                energy=energy-math.ceil(5*self.work_rate)
            elif state[1]==-1 and charging_num==1:
                energy=energy+math.ceil((20-math.ceil(distance*12/2095))*self.charge_rate)
                if energy>100:
                    energy=100
            return energy
        elif mark==2:
            energy=state[0]
            energy=energy-math.ceil(distance*self.travelling_rate_service)
            if state[1]==1 and working_num==1:
                energy=energy-math.ceil(5*self.work_rate)
            elif state[1]==-1 and charging_num==1:
                energy=energy+math.ceil((20-math.ceil(distance*12/2095))*self.charge_rate)
                if energy>100:
                    energy=100
            return energy
        
    def fuel_calculation(self,mark,state,distance,charging_num,working_num):
        energy=state[0]
            #计算移动中消耗油量
        energy=energy-math.ceil(distance*self.travelling_rate_fuel)
        if state[1]==1 and working_num==1:
            energy=energy-math.ceil(5*self.work_rate_fuel)
        elif state[1]==-1 and charging_num==1:
            energy=100
        return energy
        
    def state_update(self,i,state,charging_num,working_num,distance):
        if i<=3 or i>7:
            if working_num==1:
                if distance<=2000:
                    return 0
                else:
                    return 1
            elif charging_num==2 or working_num==2:
                return 0
            else:
                return state
        else:
            if working_num==1:
                if distance<=2000:
                    return 0
                else:
                    return 1
            if charging_num==1:
                if distance<=2000:
                    return 0
                else:
                    return -1
            elif working_num==2:
                return 0
            else:
                return state
        
    def end(self,state):
        if state[0][0]>=30 and state[1][0]>=30 and state[2][0]>=30 and state[3][0]>=30 and state[4][0]>=30 and state[5][0]>=30 and state[6][0]>=30 and state[7][0]>=30 and state[8][0]>=30 and state[9][0]>=30 and state[10][0]>=30 and state[11][0]>=30:
            return False
        else:
            return True
    
    def get_reward(self,state,distance):
        e1,e2,e3,e4,c1,c2,c3,c4,b1,b2,b3,b4=state[0][0]/100,state[1][0]/100,state[2][0]/100,state[3][0]/100,state[4][0]/100,state[5][0]/100,state[6][0]/100,state[7][0]/100,state[8][0]/100,state[9][0]/100,state[10][0]/100,state[11][0]/100
        if state[0][0]<30 or state[1][0]<30 or state[2][0]<30 or state[3][0]<30 or state[4][0]<30 or state[5][0]<30 or state[6][0]<30 or state[7][0]<30 or state[8][0]<30 or state[9][0]<30 or state[10][0]<30 or state[11][0]<30:
            return -1000
        else:
            for i in range(0,4):
                distance[i]=distance[i]/(2095*2)
                distance[i+4]=distance[i+4]/2095
                distance[i+8]=distance[i+8]/(2095*2)
            if sum(distance)==0:
                return (e1+e2+e3+e4+c1+c2+c3+c4+b1+b2+b3+b4)**2
            else:
                return (e1+e2+e3+e4+c1+c2+c3+c4+b1+b2+b3+b4)**2/sum(distance)


class FlattenDictWrapper(gym.Wrapper):  
    def __init__(self, env):  
        super(FlattenDictWrapper, self).__init__(env)  
        self.action_space = spaces.MultiDiscrete([3]*12) 

    def step(self, action):    
        ev_action = action[0:4]  
        ba_action = action[4:8]
        cv_action = action[8:12]
        return self.env.step({"ev": ev_action, "cv": cv_action, "ba": ba_action})  

    def reset(self,seed=None):  
        return self.env.reset()
    
class DistanceLoggerCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=0):
        super(DistanceLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.data = []

    def _on_step(self) -> bool:
        # 获取当前距离信息并记录
        distance = self.locals['infos'][0].get('distance', None)
        if distance is not None:
            self.data.append({'step': self.num_timesteps, 'distance': distance})
        return True

    def _on_training_end(self):
        # 保存数据到 CSV 文件
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_dir + '/distance_log_V3_3.csv', index=False)