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
        ##步长设置，一天的训练量
        self.step_num=0
        self.max_step=288

        ##车辆种类
        self.vehicle_type=['nb_airtug','wb_airtug','water_service','baggage_tractor','hi_lift_truck','cargo_tractor','belt_loader','jcpl','transporter']

        ##nb_airtug相关参数
        self.nb_airtug_param=dict()
        self.nb_airtug_param['name']='nb_airtug'
        #nb_airtug数量以及已排序车辆
        self.nb_airtug_param['num']=15
        self.nb_airtug_param['ahead_num']=0
        #nb_airtug充电功率及电量以及低电量阈值
        self.nb_airtug_param['charge_power']=125
        self.nb_airtug_param['charge_rate']=2
        self.nb_airtug_param['low_power']=30
        #nb_airtug工作功率kwh/task及时间，task1对应pushback，task2对应tow
        self.nb_airtug_param['task_1_cost']=10
        self.nb_airtug_param['task_1_time']=5
        self.nb_airtug_param['task_2_cost']=90
        self.nb_airtug_param['task_2_time']=45
        #nb_airtug行驶功耗kwh/min
        self.nb_airtug_param['travel_rate']=0.5
        #nb_airtug空载速度与任务时速度，km/min
        self.nb_airtug_param['travel_speed']=0.33
        self.nb_airtug_param['task_speed']=0.083
        
        ##wb_airtug相关参数
        self.wb_airtug_param=dict()
        self.wb_airtug_param['name']='wb_airtug'
        #wb_airtug数量以及已排序车辆
        self.wb_airtug_param['num']=20
        self.wb_airtug_param['ahead_num']=15
        #wb_airtug充电功率及电量以及低电量阈值
        self.wb_airtug_param['charge_power']=350
        self.wb_airtug_param['charge_rate']=2
        self.wb_airtug_param['low_power']=80
        #wb_airtug工作功率kwh/task及时间，task1对应pushback，task2对应tow
        self.wb_airtug_param['task_1_cost']=18.33
        self.wb_airtug_param['task_1_time']=5
        self.wb_airtug_param['task_2_cost']=165
        self.wb_airtug_param['task_2_time']=45
        #wb_airtug行驶功耗kwh/min
        self.wb_airtug_param['travel_rate']=0.92
        #wb_airtug空载速度与任务时速度，km/min
        self.wb_airtug_param['travel_speed']=0.33
        self.wb_airtug_param['task_speed']=0.083
        
        ##water_service相关参数
        self.water_service_param=dict()
        self.water_service_param['name']='water_service'
        #water_service数量以及已排序车辆
        self.water_service_param['num']=15
        self.water_service_param['ahead_num']=35
        #water_service充电功率及电量以及低电量阈值
        self.water_service_param['charge_power']=100
        self.water_service_param['charge_rate']=0.83
        self.water_service_param['low_power']=30
        #water_service工作功率kwh/task及时间，仅有task1
        self.water_service_param['task_1_cost']=7.33
        self.water_service_param['task_1_time']=20
        self.water_service_param['task_2_cost']=0
        self.water_service_param['task_2_time']=0
        #water_service行驶功耗kwh/min
        self.water_service_param['travel_rate']=0.375
        #water_service空载速度与任务时速度，km/min
        self.water_service_param['travel_speed']=0.4
        self.water_service_param['task_speed']=0

        ## baggage_tractor 相关参数
        self.baggage_tractor_param = dict()
        self.baggage_tractor_param['name'] = 'baggage_tractor'
        # baggage_tractor 数量以及已排序车辆
        self.baggage_tractor_param['num'] = 15
        self.baggage_tractor_param['ahead_num'] = 50
        # baggage_tractor 充电功率及电量以及低电量阈值
        self.baggage_tractor_param['charge_power'] = 50
        self.baggage_tractor_param['charge_rate'] = 0.67
        self.baggage_tractor_param['low_power'] = 15
        # baggage_tractor 工作功率 kwh/task 及时间，仅有 task1
        self.baggage_tractor_param['task_1_cost'] = 3
        self.baggage_tractor_param['task_1_time'] = 12
        self.baggage_tractor_param['task_2_cost'] = 0
        self.baggage_tractor_param['task_2_time'] = 0
        # baggage_tractor 行驶功耗 kwh/min
        self.baggage_tractor_param['travel_rate'] = 0.1
        # baggage_tractor 空载速度与任务时速度，km/min
        self.baggage_tractor_param['travel_speed'] = 0.3
        self.baggage_tractor_param['task_speed'] = 0

        ## hi_lift_truck 相关参数
        self.hi_lift_truck_param = dict()
        self.hi_lift_truck_param['name'] = 'hi_lift_truck'
        # hi_lift_truck 数量以及已排序车辆
        self.hi_lift_truck_param['num'] = 20
        self.hi_lift_truck_param['ahead_num'] = 65
        # hi_lift_truck 充电功率及电量以及低电量阈值
        self.hi_lift_truck_param['charge_power'] = 280
        self.hi_lift_truck_param['charge_rate'] = 0.83
        self.hi_lift_truck_param['low_power'] = 84
        # hi_lift_truck 工作功率 kwh/task 及时间，仅有 task1
        self.hi_lift_truck_param['task_1_cost'] = 20
        self.hi_lift_truck_param['task_1_time'] = 10
        self.hi_lift_truck_param['task_2_cost'] = 0
        self.hi_lift_truck_param['task_2_time'] = 0
        # hi_lift_truck 行驶功耗 kwh/min
        self.hi_lift_truck_param['travel_rate'] = 0.1
        # hi_lift_truck 空载速度与任务时速度，km/min
        self.hi_lift_truck_param['travel_speed'] = 0.4
        self.hi_lift_truck_param['task_speed'] = 0

        ##cargo_tractor相关参数
        self.cargo_tractor_param=dict()
        self.cargo_tractor_param['name']='cargo_tractor'
        #cargo_tractor数量以及已排序车辆
        self.cargo_tractor_param['num']=20
        self.cargo_tractor_param['ahead_num']=85
        #cargo_tractor充电功率及电量以及低电量阈值
        self.cargo_tractor_param['charge_power']=80
        self.cargo_tractor_param['charge_rate']=0.67
        self.cargo_tractor_param['low_power']=24
        #cargo_tractor工作功率kwh/task及时间，仅有task1
        self.cargo_tractor_param['task_1_cost']=7.5
        self.cargo_tractor_param['task_1_time']=24
        self.cargo_tractor_param['task_2_cost']=0
        self.cargo_tractor_param['task_2_time']=0
        #cargo_tractor行驶功耗kwh/min
        self.cargo_tractor_param['travel_rate']=1.67
        #cargo_tractor空载速度与任务时速度，km/min
        self.cargo_tractor_param['travel_speed']=0.4
        self.cargo_tractor_param['task_speed']=0

        ##belt_loader相关参数
        self.belt_loader_param=dict()
        self.belt_loader_param['name']='belt_loader'
        #belt_loader数量以及已排序车辆
        self.belt_loader_param['num']=50
        self.belt_loader_param['ahead_num']=105
        #belt_loader充电功率及电量以及低电量阈值
        self.belt_loader_param['charge_power']=70
        self.belt_loader_param['charge_rate']=0.83
        self.belt_loader_param['low_power']=21
        #belt_loader工作功率kwh/task及时间，仅有task1
        self.belt_loader_param['task_1_cost']=2.08
        self.belt_loader_param['task_1_time']=25
        self.belt_loader_param['task_2_cost']=0
        self.belt_loader_param['task_2_time']=0
        #belt_loader行驶功耗kwh/min
        self.belt_loader_param['travel_rate']=0.08
        #belt_loader空载速度与任务时速度，km/min    
        self.belt_loader_param['travel_speed']=0.4
        self.belt_loader_param['task_speed']=0

        ##jcpl相关参数
        self.jcpl_param=dict()
        self.jcpl_param['name']='jcpl'
        #jcpl数量以及已排序车辆
        self.jcpl_param['num']=25
        self.jcpl_param['ahead_num']=155
        #jcpl充电功率及电量以及低电量阈值
        self.jcpl_param['charge_power']=115
        self.jcpl_param['charge_rate']=0.83
        self.jcpl_param['low_power']=35
        #jcpl工作功率kwh/task及时间，仅有task1
        self.jcpl_param['task_1_cost']=10.33
        self.jcpl_param['task_1_time']=35
        self.jcpl_param['task_2_cost']=0
        self.jcpl_param['task_2_time']=0
        #jcpl行驶功耗kwh/min
        self.jcpl_param['travel_rate']=0.09
        #jcpl空载速度与任务时速度，km/min
        self.jcpl_param['travel_speed']=0.4
        self.jcpl_param['task_speed']=0

        ##transporter相关参数
        self.transporter_param=dict()
        self.transporter_param['name']='transporter'
        #transporter数量以及已排序车辆
        self.transporter_param['num']=15
        self.transporter_param['ahead_num']=180
        #transporter充电功率及电量以及低电量阈值
        self.transporter_param['charge_power']=75
        self.transporter_param['charge_rate']=0.83
        self.transporter_param['low_power']=23
        #transporter工作功率kwh/task及时间，仅有task1
        self.transporter_param['task_1_cost']=5.625
        self.transporter_param['task_1_time']=25
        self.transporter_param['task_2_cost']=0
        self.transporter_param['task_2_time']=0
        #transporter行驶功耗kwh/min
        self.transporter_param['travel_rate']=0.1
        #transporter空载速度与任务时速度，km/min
        self.transporter_param['travel_speed']=0.4
        self.transporter_param['task_speed']=0


        ##设置观测空间
        self.observation_space=spaces.Dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                self.observation_space[f"{vehicle}_{i+1}"] = spaces.Box(low=np.array([0,-2,-2]), high=np.array([param['charge_power'],1,17]), shape=(3,), dtype=np.int32)
            if vehicle=='nb_airtug' or vehicle=='wb_airtug':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),shape=(20,),dtype=np.int32)
            elif vehicle=='water_service':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='baggage_tractor':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='hi_lift_truck':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='cargo_tractor':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='belt_loader':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='jcpl':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
            elif vehicle=='transporter':
                self.observation_space[vehicle]=spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),high=np.array([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),shape=(20,),dtype=np.int32)
        #print("Observation space keys:", self.observation_space.spaces.keys())

        ##设置不同车辆对应的工作时长以及充电时长和使用次数
        self.wt=dict()
        self.wu=dict()
        self.ct=dict()
        self.cu=dict()
        self.use_num=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            self.wt[vehicle]=np.zeros(param['num'])
            self.wu[vehicle]=np.zeros(param['num'])
            self.ct[vehicle]=np.zeros(param['num'])
            self.cu[vehicle]=np.zeros(param['num'])
            self.use_num[vehicle]=np.zeros(param['num'])

        ##step过程中的state记录
        self.demand_state=dict()
        for vehicle in self.vehicle_type:
            self.demand_state[vehicle]=np.zeros(20,dtype=np.int32)
            self.demand_state[vehicle][0],self.demand_state[vehicle][1]=2,2
        self.state=(tuple(np.array([125, 0, 0]) for _ in range(15))+tuple(np.array([350, 0, 0]) for _ in range(20))
                    +tuple(np.array([100, 0, 0]) for _ in range(15))+tuple(np.array([50, 0, 0]) for _ in range(15))
                    +tuple(np.array([280, 0, 0]) for _ in range(20))+tuple(np.array([80, 0, 0]) for _ in range(20))
                    +tuple(np.array([70, 0, 0]) for _ in range(50))+tuple(np.array([115, 0, 0]) for _ in range(25))
                    +tuple(np.array([75,0,0]) for _ in range(15))
                    )
        
        ##创建距离矩阵,公里
        distances=[0.08,0.09,0.1,0.17,0.15,0.19,0.09,0.04,0.06,0.06,0.05,0.05,0.04,0.22,0.15,0.14,0.14]
        for i in range(17):
            distances[i]+=0.07
        self.distance_matrix=self.create_distance_matrix(distances)
        #F32充电桩至最近A18距离，公里
        self.airtug_charge_distance=2.49

        ##记录未完成任务
        self.task_uncompleted=dict()
        for vehicle in self.vehicle_type:
            self.task_uncompleted[vehicle]=0

        ##记录工作情况
        self.work_table=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                self.work_table[vehicle+'_'+str(i+1)]=np.ones(self.max_step,dtype=np.int32)
        
        ##任务格式更改
        self.df_arrival=pd.read_excel('D:/VsCodeData/airport/mixed_fleet/a_area_task_arrival.xlsx')
        self.df_pushback=pd.read_excel('D:/VsCodeData/airport/mixed_fleet/a_area_task_departure.xlsx')
        self.df_tow=pd.read_excel('D:/VsCodeData/airport/mixed_fleet/a_area_towing_task.xlsx')
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
        self.df_arrival['Scheduled']=pd.to_datetime(self.df_arrival['Scheduled'])
        self.df_arrival['time_total']=self.df_arrival['Scheduled'].dt.hour*60+self.df_arrival['Scheduled'].dt.minute
        self.df_arrival['step']=self.df_arrival['time_total']//5
        self.df_arrival['Stand_num']=self.df_arrival['Stand'].str.extract('(\d+)').astype(int)
        for i in range(0,83):
            self.df_arrival.loc[i,'Stand_num']=self.stand_transition(self.df_arrival['Stand_num'][i])

    def step(self, action):
        reward=0
        action=self.reset_action(action)
        distance=dict()
        use=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            distance[vehicle]=np.zeros(param['num'])
            use[vehicle]=np.zeros(param['num'])
            use[vehicle],distance[vehicle]=self.work_arrange(param,action[vehicle],distance[vehicle],use[vehicle])
            use[vehicle],distance[vehicle]=self.essential_charge(param,distance[vehicle],use[vehicle])
        

        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            use[vehicle],distance[vehicle]=self.charge_arrange(param,action[vehicle],distance[vehicle],use[vehicle])

        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            self.power_calculate(param,distance[vehicle],self.wt[vehicle],self.ct[vehicle],action[vehicle])

        self.step_num+=1
        terminated=self.end(self.step_num,self.max_step)

        self.demand_update()

        reward=self.get_reward(use)    

        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                if self.wu[vehicle][i]>0:
                    self.wt[vehicle][i]+=1
                    if self.wt[vehicle][i]>=self.wu[vehicle][i]:
                        self.wt[vehicle][i]=0
                        self.wu[vehicle][i]=0
                        self.state[i+int(param['ahead_num'])][1]=0
                if self.cu[vehicle][i]>0:
                    self.ct[vehicle][i]+=1
                    if self.ct[vehicle][i]>=self.cu[vehicle][i]:
                        self.ct[vehicle][i]=0
                        self.cu[vehicle][i]=0
                        self.state[i+int(param['ahead_num'])][1]=0
                        if vehicle=='nb_airtug' or vehicle=='wb_airtug':
                            self.demand_state['nb_airtug'][int(-self.state[i+int(param['ahead_num'])][2]-1)]+=1
                            self.demand_state['wb_airtug'][int(-self.state[i+int(param['ahead_num'])][2]-1)]+=1
                        if vehicle=='water_service' or vehicle=='belt_loader':
                            self.demand_state['water_service'][int(self.state[i+int(param['ahead_num'])][2]+2)]+=1
                            self.demand_state['belt_loader'][int(self.state[i+int(param['ahead_num'])][2]+2)]+=1
                        else:
                            self.demand_state[vehicle][int(self.state[i+int(param['ahead_num'])][2]+2)]+=1
                        

        state=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                state[vehicle+'_'+str(i+1)]=self.state[i+int(param['ahead_num'])]
            state[vehicle]=self.demand_state[vehicle]
        info=dict()
        truncated=self.step_num>=self.max_step

        #print(self.task_uncompleted)
        if self.step_num>=self.max_step:
            #print(self.work_table)
            df = pd.DataFrame(self.work_table)
            df.to_csv('output.csv', index=False)

        return state, reward, terminated,truncated,info

    def reset(self,seed=None):
        self.step_num=0
        ##设置不同车辆对应的工作时长以及充电时长和使用次数
        self.wt=dict()
        self.wu=dict()
        self.ct=dict()
        self.cu=dict()
        self.use_num=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            self.wt[vehicle]=np.zeros(param['num'])
            self.wu[vehicle]=np.zeros(param['num'])
            self.ct[vehicle]=np.zeros(param['num'])
            self.cu[vehicle]=np.zeros(param['num'])
            self.use_num[vehicle]=np.zeros(param['num'])

        ##step过程中的state记录
        self.demand_state=dict()
        for vehicle in self.vehicle_type:
            self.demand_state[vehicle]=np.zeros(20,dtype=np.int32)
            self.demand_state[vehicle][0],self.demand_state[vehicle][1]=2,2
        self.state=(tuple(np.array([125, 0, 0]) for _ in range(15))+tuple(np.array([350, 0, 0]) for _ in range(20))
                    +tuple(np.array([100, 0, 0]) for _ in range(15))+tuple(np.array([50, 0, 0]) for _ in range(15))
                    +tuple(np.array([280, 0, 0]) for _ in range(20))+tuple(np.array([80, 0, 0]) for _ in range(20))
                    +tuple(np.array([70, 0, 0]) for _ in range(50))+tuple(np.array([115, 0, 0]) for _ in range(25))
                    +tuple(np.array([75,0,0]) for _ in range(15))
                    )
        ##记录工作情况
        self.work_table=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                self.work_table[vehicle+'_'+str(i+1)]=np.ones(self.max_step,dtype=np.int32)

        state=dict()
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                state[vehicle+'_'+str(i+1)]=self.state[i+int(param['ahead_num'])]

            state[vehicle]=self.demand_state[vehicle]
        info=dict()
        return state,info

    def get_reward(self,use):
        reward=0
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            for i in range(0,param['num']):
                if self.state[i+int(param['ahead_num'])][0]<param['low_power']:
                    reward-=100
                if use[vehicle][i]>0:
                    reward-=10
                if self.step==self.max_step:
                    if self.use_num[vehicle][i]==0:
                        reward+=50
        return reward

    def demand_update(self):
        for i in range(0,87):
            if self.df_pushback['step'][i]==self.step_num:
                if self.df_pushback['NB/WB'][i]=='NB':
                    self.demand_state['nb_airtug'][self.df_pushback['Stand_num'][i]]=1
                    self.demand_state['water_service'][self.df_pushback['Stand_num'][i]]=1
                    self.demand_state['baggage_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['hi_lift_truck'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['cargo_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['belt_loader'][self.df_pushback['Stand_num'][i]] = 1
                else:
                    self.demand_state['wb_airtug'][self.df_pushback['Stand_num'][i]]=1
                    self.demand_state['water_service'][self.df_pushback['Stand_num'][i]]=1
                    self.demand_state['baggage_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['hi_lift_truck'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['cargo_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['belt_loader'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['jcpl'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['transporter'][self.df_pushback['Stand_num'][i]] = 1
        for i in range(0,12):
            if self.df_tow['step'][i]==self.step_num:
                if self.df_tow['NB/WB'][i]=='NB':
                    self.demand_state['nb_airtug'][self.df_tow['Stand_num'][i]]=2
                else:
                    self.demand_state['wb_airtug'][self.df_tow['Stand_num'][i]]=2
        for i in range(0,83):
            if self.df_arrival['step'][i]==self.step_num:
                if self.df_arrival['NB/WB'][i]=='NB':
                    self.demand_state['baggage_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['cargo_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['hi_lift_truck'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['belt_loader'][self.df_pushback['Stand_num'][i]] = 1
                else:
                    self.demand_state['baggage_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['cargo_tractor'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['hi_lift_truck'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['belt_loader'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['jcpl'][self.df_pushback['Stand_num'][i]] = 1
                    self.demand_state['transporter'][self.df_pushback['Stand_num'][i]] = 1

    def end(self,step,max):
        if step<=max:
            return False
        else:
            return True
    
    def power_calculate(self,param,distance,work,charge,action):
        for i in range(0,param['num']):
            energy=self.state[i+param['ahead_num']][0]
            if work[i]==1:
                energy=math.floor(energy-0.25*param['travel_rate']*distance[i]/param['travel_speed']-param['task_'+str(action[i]-1)+'_cost'])
            if charge[i]==1:
                energy=param['charge_power']
            self.state[i+param['ahead_num']][0]=energy
            if energy<param['low_power']:
                self.state[i+param['ahead_num']][1]=-2

    def work_arrange(self,param,action,distance,use):
        for i in range(0,param['num']):
            if action[i]>1:
                self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],distance[i],flag=self.work_setup(action[i],self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],param['name'])
                if flag==1:
                    self.wt[param['name']][i]=1
                    #print(action[i])
                    self.wu[param['name']][i]=math.ceil((distance[i]/param['travel_rate']+param['task_'+str(action[i]-1)+'_time'])/5)
                    self.use_num[param['name']][i]+=1
                    use[i]+=1
                    for j in range(0,int(self.wu[param['name']][i])):
                        if self.step_num+j<self.max_step:
                            self.work_table[param['name']+'_'+str(i+1)][self.step_num+j]=action[i]
        return use,distance

    def essential_charge(self,param,distance,use):
        for i in range(0,param['num']):
            if self.state[i+int(param['ahead_num'])][1]==-2:
                self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],distance[i],flag=self.charge_setup(self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],param['name'])
                if flag==1:
                    self.ct[param['name']][i]=1
                    self.cu[param['name']][i]=math.ceil((distance[i]/param['travel_rate']+(param['charge_power']-self.state[i+int(param['ahead_num'])][0])/param['charge_rate'])/5)
                    self.use_num[param['name']][i]+=1
                    use[i]+=1
                    for j in range(0,int(self.cu[param['name']][i])):
                        if self.step_num+j<self.max_step:
                            self.work_table[param['name']+'_'+str(i+1)][self.step_num+j]=0
        return use,distance
    
    def charge_arrange(self,param,action,distance,use):
        for i in range(0,param['num']):
            if action[i]==0:
                self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],distance[i],flag=self.charge_setup(self.state[i+int(param['ahead_num'])][1],self.state[i+int(param['ahead_num'])][2],param['name'])
                if flag==1:
                    self.ct[param['name']][i]=1
                    self.cu[param['name']][i]=math.ceil((distance[i]/param['travel_rate']+(param['charge_power']-self.state[i+int(param['ahead_num'])][0])/param['charge_rate'])/5)
                    self.use_num[param['name']][i]+=1
                    use[i]+=1
                    for j in range(0,int(self.cu[param['name']][i])):
                        if self.step_num+j<self.max_step:
                            self.work_table[param['name']+'_'+str(i+1)][self.step_num+j]=0
        return use,distance
    
    def charge_setup(self,work_state,location,name):
        a=location
        flag=0
        distance=0
        for i in range(0,2):
            if self.demand_state[name][i]>=1:
                work_state=-1
                location=-1*i-1
                self.demand_state[name][i]-=1
                flag=1
                if name=='nb_airtug' or name=='wb_airtug':
                    distance=self.distance_matrix[int(a),14]+self.airtug_charge_distance
                    if name=='nb_airtug':
                        self.demand_state['wb_airtug'][i]-=1
                    else:
                        self.demand_state['nb_airtug'][i]-=1
                elif name!='nb_airtug' and name!='wb_airtug':
                    #-1的状态对应A12，-2的状态对应A11
                    distance=self.distance_matrix[int(a),int(i+9)]
                break
        
        return work_state,location,distance,flag

    def work_setup(self,action,work_state,location,name):
        flag=0
        if action==2:
            a=location
            for i in range(2,20):
                if self.demand_state[name][i]==1:
                    work_state=1
                    location=i-2
                    self.demand_state[name][i]=0
                    flag=1
                    break
            b=location
            distance=self.distance_matrix[int(a),int(b)]
            return work_state,location,distance,flag
        elif action==3:
            a=location
            for i in range(2,20):
                if self.demand_state[name][i]==2:
                    work_state=1
                    location=i-2
                    self.demand_state[name][i]=0
                    flag=1
                    break
            b=location
            distance=self.distance_matrix[int(a),int(b)]
            return work_state,location,distance,flag
        else:
            return work_state,location,0,flag
        
    def stand_transition(self,x):
        #将站点名称按顺序编号
        if x<6:
            return x-1
        else:
            return x-4
        
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
    
    def reset_action(self,action):
        for i in range(0, self.water_service_param['num']):
            if action['water_service'][i]==3:
                action['water_service'][i]=2
        for i in range(0, self.baggage_tractor_param['num']):
            #print("action:", action)
            if action['baggage_tractor'][i]==3:
                action['baggage_tractor'][i]=2
        for i in range(0, self.hi_lift_truck_param['num']):
            #print("action:", action)
            if action['hi_lift_truck'][i]==3:
                action['hi_lift_truck'][i]=2
        for i in range(0, self.cargo_tractor_param['num']):
            if action['cargo_tractor'][i]==3:
                action['cargo_tractor'][i]=2
        for i in range(0, self.belt_loader_param['num']):
            if action['belt_loader'][i]==3:
                action['belt_loader'][i]=2
        for i in range(0, self.jcpl_param['num']):
            if action['jcpl'][i]==3:
                action['jcpl'][i]=2
        for i in range(0, self.transporter_param['num']):
            if action['transporter'][i]==3:
                action['transporter'][i]=2
        for vehicle in self.vehicle_type:
            param_name = f"{vehicle}_param"
            param = getattr(self, param_name)
            task_1=0
            task_2=0
            task_1_car=0
            task_2_car=0
            ##记录派遣充电或工作的车辆的电量，选取点亮最高的前往工作以及电量最低的前往充电
            available_charging_station=self.demand_state[vehicle][0]+self.demand_state[vehicle][1]
            charging_demand=0
            car_work_1_id=np.zeros(param['num'])
            car_work_2_id=np.zeros(param['num'])
            car_charge_id=np.zeros(param['num'])            
            for i in range(2,20):
                if self.demand_state[vehicle][i]==1:
                    task_1+=1
                if self.demand_state[vehicle][i]==2:
                    task_2+=1
            for i in range(0,param['num']):
                if self.state[i+int(param['ahead_num'])][0]==param['charge_power'] and action[vehicle][i]==0:
                    action[vehicle][i]=1
                if (vehicle=='nb_airtug' or vehicle=='wb_airtug') and self.state[i+int(param['ahead_num'])][0]<param['charge_power'] and action[vehicle][i]==3:
                    action[vehicle][i]=1
                if self.state[i+int(param['ahead_num'])][1]==-2:
                    action[vehicle][i]=1
                if action[vehicle][i]==0:
                    charging_demand+=1
                    car_charge_id[i]=self.state[i+int(param['ahead_num'])][0]
            ##规定电量较低的优先充电
            if charging_demand>available_charging_station:
                indices_charge = np.argsort(car_charge_id)
                chosen_charge=indices_charge[0:available_charging_station]
                for i in chosen_charge:
                    action[vehicle][i]=-1
            for i in range(0,param['num']):
                if action[vehicle][i]<=0:
                    action[vehicle][i]+=1
            for i in range(0,param['num']):
                if action[vehicle][i]==2:
                    task_1_car+=1
                    car_work_1_id[i]=self.state[i+int(param['ahead_num'])][0]
                if action[vehicle][i]==3:
                    task_2_car+=1
                    car_work_2_id[i]=self.state[i+int(param['ahead_num'])][0]
            if task_1_car>task_1:
                indices_work_1=np.argsort(car_work_1_id)
                chosen_work_1=indices_work_1[0:param['num']-task_1]
                for i in chosen_work_1:
                    if action[vehicle][i]==2:
                        action[vehicle][i]=1
            if task_2_car>task_2:
                indices_work_2=np.argsort(car_work_2_id)
                chosen_work_2=indices_work_2[0:param['num']-task_2]
                for i in chosen_work_2:
                    if action[vehicle][i]==3:
                        action[vehicle][i]=1
            if task_2>task_2_car:
                for i in range(0,param['num']):
                    if self.state[i+int(param['ahead_num'])][0]-param['task_2_cost']>param['low_power'] and action[vehicle][i]==1:
                        action[vehicle][i]=3
                        task_2_car+=1
                        if task_2_car==task_2:
                            break
            if task_2>task_2_car:
                self.task_uncompleted[vehicle]+=1
            if task_1>task_1_car:
                for i in range(0,param['num']):
                    if self.state[i+int(param['ahead_num'])][0]-param['task_1_cost']>param['low_power'] and action[vehicle][i]==1:
                        action[vehicle][i]=2
                        task_1_car+=1
                        if task_1_car==task_1:
                            break
            if task_1>task_1_car:
                self.task_uncompleted[vehicle]+=1

        return action

class FlattenDictWrapper(gym.Wrapper):  
    def __init__(self, env):  
        super(FlattenDictWrapper, self).__init__(env) 
        self.action_space = spaces.MultiDiscrete([4]*195)

    def step(self, action):   
        #action=action[0]
        na_action = action[0:15]  
        wa_action = action[15:35]
        ws_action = action[35:50]
        bt_action = action[50:65]
        hl_action = action[65:85]
        ct_action = action[85:105]
        bl_action = action[105:155]
        jp_action = action[155:180]
        tp_action = action[180:195]
        return self.env.step({"nb_airtug": na_action, "wb_airtug": wa_action, "water_service": ws_action, "baggage_tractor": bt_action, "hi_lift_truck": hl_action, "cargo_tractor": ct_action, "belt_loader": bl_action,'jcpl':jp_action,'transporter':tp_action})

    def reset(self,seed=None):  
        return self.env.reset()
        
        
        
        
        
        
        
        
        
        
        
        
        