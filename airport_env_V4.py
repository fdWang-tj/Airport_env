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
        self.observation_space=spaces.Dict()
        #NB-airtug
        for i in range(0,39):
            self.observation_space['nb_airtug_'+str(i+1)]=spaces.Box(low=np.array([0,-2,-2]),high=np.array([125,1,17]),shape=(3,),dtype=np.int32)
        #WB-airtug
        for i in range(0,60):
            self.observation_space['wb_airtug_'+str(i+1)]=spaces.Box(low=np.array([0,-2,-2]),high=np.array([350,1,17]),shape=(3,),dtype=np.int32)
        
        self.observation_space['station']=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]),shape=(20,),dtype=np.int32)

        self.step_num=0
        self.max_step=288

        #一分钟120kW充电桩能充2kWh的电量
        self.charge_rate=2
        
        #工作功率kwh/task
        self.nb_pushback_rate=10
        self.nb_tow_rate=90
        self.wb_pushback_rate=18.33
        self.wb_tow_rate=165

        #行驶功耗kwh/min
        self.nb_travel_rate=0.5
        self.wb_travel_rate=0.92

        #行驶速度，km/min
        self.empty_speed=0.33
        self.tow_speed=0.083

        #记录充电时长与工作时长以及上限时间
        self.charge_record=np.zeros(99)
        self.charge_upbound=np.zeros(99)
        self.work_record=np.zeros(99)
        self.work_upbound=np.zeros(99)

        #step过程中的state记录
        self.service_demand=np.zeros(20,dtype=np.int32)
        self.service_demand[0],self.service_demand[1]=2,2
        self.state = (tuple(np.array([125, 0, 0]) for _ in range(39))+tuple(np.array([350, 0, 0]) for _ in range(60))+(self.service_demand,))  

        #创建距离矩阵,公里
        distances=[0.08,0.09,0.1,0.17,0.15,0.19,0.09,0.04,0.06,0.06,0.05,0.05,0.04,0.22,0.15,0.14,0.14]
        for i in range(17):
            distances[i]+=0.07
        self.distance_matrix=self.create_distance_matrix(distances)
        #F32充电桩至最近A18距离，公里
        self.distance_charge=2.49

        #使用次数记录
        self.usage=np.zeros(99)

        #分类标志
        self.nb_flag=0
        self.wb_flag=1

        #任务格式更改，方便后续计算
        self.df_pushback=pd.read_excel('D:/VSCodeData/airport/mixed_fleet/a_area_task.xlsx')
        self.df_tow=pd.read_excel('D:/VSCodeData/airport/mixed_fleet/a_area_towing_task.xlsx')
        self.df_pushback['Scheduled']=pd.to_datetime(self.df_pushback['Scheduled'])
        self.df_pushback['time_total']=self.df_pushback['Scheduled'].dt.hour*60+self.df_pushback['Scheduled'].dt.minute
        self.df_pushback['step']=self.df_pushback['time_total']//5
        self.df_pushback['Stand_num']=self.df_pushback['Stand'].str.extract('(\d+)').astype(int)
        for i in range(0,87):
            self.df_pushback.loc[i,'Stand_num']=self.stand_transition(self.df_pushback['Stand_num'][i])
        self.df_tow['Scheduled Tow Out DT']=pd.to_datetime(self.df_tow['Scheduled Tow Out DT'])
        self.df_tow['time_total']=self.df_tow['Scheduled Tow Out DT'].dt.hour*60+self.df_tow['Scheduled Tow Out DT'].dt.minute
        self.df_tow['step']=self.df_tow['time_total']//5
        self.df_tow['Stand_num']=self.df_tow['From Parking Stand'].str.extract('(\d+)').astype(int)
        for i in range(0,12):
            self.df_tow.loc[i,'Stand_num']=self.stand_transition(self.df_tow['Stand_num'][i])

        self.reward=0


    def step(self,action):
        #print(self.state[99])
        distance=np.zeros(99)
        use_times=np.zeros(99)
        nb_action=action['nb_airtug']-1
        wb_action=action['wb_airtug']-1
        reset_nb_action,reset_wb_action=self.reset_action(nb_action,wb_action,self.state)
        for i in range(0,39):
            if reset_nb_action[i]==1:
                self.state[i][1],self.state[i][2],new_demand,distance[i],flag=self.work_setup(self.nb_flag,reset_nb_action[i],self.state[i][1],self.state[i][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.work_record[i]=1
                    self.work_upbound[i]=math.ceil((distance[i]/self.nb_travel_rate+5)/5)
                    self.usage[i]+=1
                    use_times[i]+=1
            if reset_nb_action[i]==2:
                self.state[i][1],self.state[i][2],new_demand,distance[i],flag=self.work_setup(self.nb_flag,reset_nb_action[i],self.state[i][1],self.state[i][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.work_record[i]=1
                    self.work_upbound[i]=math.ceil((distance[i]/self.nb_travel_rate+45)/5)
                    self.usage[i]+=1
                    use_times[i]+=1
            if self.state[i][1]==-2:
                self.state[i][1],self.state[i][2],new_demand,distance[i],flag=self.charge_setup(self.nb_flag,self.state[i][1],self.state[i][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.charge_record[i]=1
                    self.charge_upbound[i]=math.ceil((distance[i]/self.nb_travel_rate+(125-self.state[i][0])/self.charge_rate)/5)
                    self.usage[i]+=1
                    use_times[i]+=1

        for i in range(0,60):
            if reset_wb_action[i]==1:
                self.state[i+39][1],self.state[i+39][2],new_demand,distance[i+39],flag=self.work_setup(self.wb_flag,reset_wb_action[i],self.state[i+39][1],self.state[i+39][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.work_record[i+39]=1
                    self.work_upbound[i+39]=math.ceil((distance[i+39]/self.wb_travel_rate+5)/5)
                    self.usage[i+39]+=1
                    use_times[i+39]+=1
            if reset_wb_action[i]==2:
                self.state[i+39][1],self.state[i+39][2],new_demand,distance[i+39],flag=self.work_setup(self.wb_flag,reset_wb_action[i],self.state[i+39][1],self.state[i+39][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.work_record[i+39]=1
                    self.work_upbound[i+39]=math.ceil((distance[i+39]/self.wb_travel_rate+45)/5)
                    self.usage[i+39]+=1
                    use_times[i+39]+=1
            if self.state[i+39][1]==-2:
                self.state[i+39][1],self.state[i+39][2],new_demand,distance[i+39],flag=self.charge_setup(self.wb_flag,self.state[i+39][1],self.state[i+39][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.charge_record[i+39]=1
                    self.charge_upbound[i+39]=math.ceil((distance[i+39]/self.wb_travel_rate+(125-self.state[i+39][0])/self.charge_rate)/5)
                    self.usage[i+39]+=1
                    use_times[i+39]+=1
        
        for i in range(0,39):
            if nb_action[i]==-1:
                self.state[i][1],self.state[i][2],new_demand,distance[i],flag=self.charge_setup(self.nb_flag,self.state[i][1],self.state[i][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.charge_record[i]=1
                    self.charge_upbound[i]=math.ceil((distance[i]/self.nb_travel_rate+(125-self.state[i][0])/self.charge_rate)/5)
                    self.usage[i]+=1
                    use_times[i]+=1
        
        for i  in range(0,60):
            if wb_action[i]==-1:
                self.state[i+39][1],self.state[i+39][2],new_demand,distance[i+39],flag=self.charge_setup(self.wb_flag,self.state[i+39][1],self.state[i+39][2],self.state[99])
                self.state=self.state[:99]+(new_demand,)
                if flag>0:
                    self.charge_record[i+39]=1  
                    self.charge_upbound[i+39]=math.ceil((distance[i+39]/self.wb_travel_rate+(350-self.state[i+39][0])/self.charge_rate)/5)
                    self.usage[i+39]+=1
                    use_times[i+39]+=1
        
        #print(self.state[99])
        
        for i in range(0,60):
            if i<39:
                self.state[i][0]=self.power_calculation(self.nb_flag,self.state[i],distance[i],self.charge_record[i],self.work_record[i],nb_action[i])
                if self.state[i][0]<35:
                    self.state[i][1]=-2
            self.state[i+39][0]=self.power_calculation(self.wb_flag,self.state[i+39],distance[i+39],self.charge_record[i+39],self.work_record[i+39],wb_action[i])
            if self.state[i+39][0]<100:
                self.state[i+39][1]=-2
        
        power=np.zeros(99)
        for i in range(0,99):
            power[i]=self.state[i][0]
        #print(power)
        terminated=self.end(self.step_num,self.max_step)
        reward=self.get_reward(self.usage,self.state,use_times,self.step_num,self.max_step)
        self.reward+=reward
        #print(reward)
        self.step_num+=1
        #print(self.state[99])
        self.service_demand=self.demand_update(self.step_num,self.service_demand,self.df_pushback,self.df_tow)
        #print(self.service_demand)
        self.state=self.state[:99]+(self.service_demand,)
        #print(self.state[99])
        #print(self.charge_upbound,self.work_upbound)
        for i in range(0,99):
            if self.work_upbound[i]>0:
                self.work_record[i]+=1
                if self.work_record[i]>=self.work_upbound[i]:
                    self.work_upbound[i]=0
                    self.work_record[i]=0
                    self.state[i][1]=0
            if self.charge_upbound[i]>0:
                self.charge_record[i]+=1
                if self.charge_record[i]>=self.charge_upbound[i]:
                    self.charge_upbound[i]=0
                    self.charge_record[i]=0
                    self.state[i][1]=0
                    self.state[99][int(-self.state[i][2])-1]+=1
        state=dict()
        for i in range(39):
            state['nb_airtug_'+str(i+1)]=self.state[i]
        for i in range(60):
            state['wb_airtug_'+str(i+1)]=self.state[i+39]
        state['station']=self.state[99]
        #print(state['station'])
        info=dict()
        truncated=self.step_num>=self.max_step
        '''if self.step_num>=self.max_step:
            a=0
            b=0
            for i in range(99):
                if self.usage[i]==0 and i<39:
                    a+=1
                elif self.usage[i]==0 and i>=39:
                    b+=1
            print('Cars not used and reward:',a,b,a+b,self.reward)'''
        return state,reward,terminated,truncated,info
    
    def reset(self,seed=None):
        self.step_num=0
        self.reward=0
        self.service_demand=np.zeros(20,dtype=np.int32)
        self.service_demand[0],self.service_demand[1]=2,2
        self.state = (tuple(np.array([125, 0, 0]) for _ in range(39))+tuple(np.array([350, 0, 0]) for _ in range(60))+(self.service_demand,))  
        self.usage=np.zeros(99)
        self.work_record=np.zeros(99)
        self.charge_record=np.zeros(99)
        self.work_upbound=np.zeros(99)
        self.charge_upbound=np.zeros(99)
        state=dict()
        for i in range(39):
            state['nb_airtug_'+str(i+1)]=self.state[i]
        for i in range(60):
            state['wb_airtug_'+str(i+1)]=self.state[i+39]
        state['station']=self.state[99]
        info=dict()
        return state,info
    
    def close(self):
        print("close")
    
    def render(self,mode="human"):
        pass
    
    def create_distance_matrix(self,distances):
        #创建距离矩阵  
        num_stations = len(distances) + 1 
        distance_matrix = np.zeros((num_stations, num_stations))  
        for i in range(num_stations):  
            for j in range(num_stations):  
                if i == j:  
                    distance_matrix[i][j] = 0 
                elif i < j:   
                    distance_matrix[i][j] = sum(distances[i:j])  
                else:  
                    distance_matrix[i][j] = distance_matrix[j][i]  
        return distance_matrix
    
    def stand_transition(self,x):
        #将站点名称按顺序编号
        if x<6:
            return x-1
        else:
            return x-4
    
    def demand_update(self,step,service,pushback,tow):
        #更新任务需求
        for i in range(0,87):
            if pushback['step'][i]==step:
                if pushback['NB/WB'][i]=='NB':
                    service[pushback['Stand_num'][i]]=1
                else:
                    service[pushback['Stand_num'][i]]=3
            if i <12:
                if tow['step'][i]==step:
                    if tow['NB/WB'][i]=='NB':
                        service[tow['Stand_num'][i]]=2
                    else:
                        service[tow['Stand_num'][i]]=4
        return service

    def reset_action(self,nb_action,wb_action,state):
        #towing task只进行一次就需要充电，故当电量较低时，不进行拖动任务
        for i in range(0,39):
            if state[i][0]<120 and nb_action[i]==2:
                nb_action[i]=0
        for i in range(0,60):
            if state[i+39][0]<330 and wb_action[i]==2:
                wb_action[i]=0
        nb_pushback_task=0
        nb_tow_task=0
        wb_pushback_task=0
        wb_tow_task=0
        for i in range(2,20):
            if state[99][i]==1:
                nb_pushback_task+=1
            elif state[99][i]==2:
                nb_tow_task+=1
            if state[99][i]==3:
                wb_pushback_task+=1
            elif state[99][i]==4:
                wb_tow_task+=1
        nb_car_pushback=0
        nb_car_towing=0
        wb_car_pushback=0
        wb_car_towing=0
        #当车辆正在工作或者电量过低时，不允许动作，以及电量满时，不进行充电，并计算此步计划中的车辆数
        for i in range(0,39):
            if (nb_action[i]==1 or nb_action[i]==2) and (state[i][1]==1 or state[i][1]==-2 or state[i][1]==-1):
                nb_action[i]=0
            if nb_action[i]==-1 and state[i][0]==125:
                nb_action[i]=0
            if nb_action[i]==1:
                nb_car_pushback+=1
            if nb_action[i]==2:
                nb_car_towing+=1
        for i in range(0,60):
            if (wb_action[i]==1 or wb_action[i]==2) and (state[i+39][1]==1 or state[i+39][1]==-2 or state[i+39][1]==-1):
                wb_action[i]=0
            if wb_action[i]==-1 and state[i+39][0]==350:
                wb_action[i]=0
            if wb_action[i]==1:
                wb_car_pushback+=1
            if wb_action[i]==2:
                wb_car_towing+=1
        #保证任务完成
        if nb_pushback_task>nb_car_pushback:
            for i in range (0,39):
                if state[i][1]==0 and nb_action[i]==0:
                    nb_action[i]=1
                    break
        if nb_tow_task>nb_car_towing:
            for i in range (0,39):
                if state[i][1]==0 and nb_action[i]==0 and state[i][0]==125:
                    nb_action[i]=2
                    break
        if wb_pushback_task>wb_car_pushback:
            for i in range (0,60):
                if state[i+39][1]==0 and wb_action[i]==0:
                    wb_action[i]=1
                    break
        if wb_tow_task>wb_car_towing:
            for i in range (0,60):
                if state[i+39][1]==0 and wb_action[i]==0 and state[i+39][0]>=320:
                    wb_action[i]=2
                    break
        charge_demand=0
        for i in range(0,39):
            if nb_action[i]==-1:
                charge_demand+=1
        for i in range(0,60):
            if wb_action[i]==-1:
                charge_demand+=1
        charge_available=state[99][0]+state[99][1]
        must_charge=0
        for i in range(0,39):
            if state[i][1]==-2:
                must_charge+=1
        for i in range(0,60):
            if state[i+39][1]==-2:
                must_charge+=1
        if charge_available<must_charge:
            for i in range(0,39):
                if nb_action[i]==-1:
                    nb_action[i]=0
            for i in range(0,60):
                if wb_action[i]==-1:
                    wb_action[i]=0
        return nb_action,wb_action
    
    def work_setup(self,flag,action,work_state,location,demand):
        if flag==0:
            flag2=0
            if action==1:
                a=location
                for i in range(2,20):
                    if demand[i]==1:
                        work_state=1
                        location=i-2
                        demand[i]=0
                        flag2=1
                        break
                b=location
                distance=self.distance_matrix[int(a),int(b)]
                return work_state,location,demand,distance,flag2
            elif action==2:
                a=location
                for i in range(2,20):
                    if demand[i]==2:
                        work_state=1
                        location=i-2
                        demand[i]=0
                        flag2=1
                        break
                b=location
                distance=self.distance_matrix[int(a),int(b)]
                return work_state,location,demand,distance,flag2
        if flag==1:
            flag2=0
            if action==1:
                a=location
                for i in range(2,20):
                    if demand[i]==3:
                        work_state=1
                        location=i-2
                        demand[i]=0
                        flag2=1
                        break
                b=location
                distance=self.distance_matrix[int(a),int(b)]
                return work_state,location,demand,distance,flag2
            elif action==2:
                a=location
                for i in range(2,20):
                    if demand[i]==4:
                        work_state=1
                        location=i-2
                        demand[i]=0
                        flag2=1
                        break
                b=location
                distance=self.distance_matrix[int(a),int(b)]
                return work_state,location,demand,distance,flag2
    
    def charge_setup(self,flag,work_state,location,demand):
        a=location
        flag2=0
        for i in range(0,2):
            if demand[i]>0:
                work_state=-1
                flag2=1
                location=-1*i-1
                demand[i]-=1
                break
        distance=self.distance_matrix[int(a),14]+self.distance_charge
        return work_state,location,demand,distance,flag2
    
    def power_calculation(self,flag,state,distance,charge,work,action):
        if flag==0:
            energy=state[0]
            if state[1]==1 and work==1 and action==1:
                energy=math.floor(energy-0.25*self.nb_travel_rate*distance/self.empty_speed-self.nb_pushback_rate)
            elif state[1]==1 and work==1 and action==2:
                energy=math.floor(energy-0.25*self.nb_travel_rate*distance/self.empty_speed-self.nb_tow_rate)
            elif state[1]==-1 and charge==1:
                energy=125
            return energy
        if flag==1:
            energy=state[0]
            if state[1]==1 and work==1 and action==1:
                energy=math.floor(energy-0.25*self.wb_travel_rate*distance/self.empty_speed-self.wb_pushback_rate)
            elif state[1]==1 and work==1 and action==2:
                energy=math.floor(energy-0.25*self.wb_travel_rate*distance/self.empty_speed-self.wb_tow_rate)
            elif state[1]==-1 and charge==1:
                energy=350
            return energy

    def end(self,num,max_step):
        if num<max_step:
            return False
        else:
            return True
    
    def get_reward(self,usage,state,use_times,step,max_step):
        reward=0
        for i in range(0,99):
            if i <39:
                if state[i][0]<30:
                    reward-=100
                elif use_times[i]>0:
                    reward-=10
            else:
                if state[i][0]<80:
                    reward-=100
                elif use_times[i]>0:
                    reward-=10
        if step==max_step-1:
            for i in range(0,99):
                if usage[i]==0:
                    reward+=50
        return reward
        


class FlattenDictWrapper(gym.Wrapper):  
    def __init__(self, env):  
        super(FlattenDictWrapper, self).__init__(env)  
        self.action_space = spaces.MultiDiscrete([4]*99) 

    def step(self, action):    
        nb_action = action[0:39]  
        wb_action = action[39:99]
        return self.env.step({"nb_airtug": nb_action, "wb_airtug": wb_action})  

    def reset(self,seed=None):  
        return self.env.reset()