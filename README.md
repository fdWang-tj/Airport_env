# Airport_env
## 简介
基于强化学习方法的机场地勤设备的充电以及工作规划，初步采取地勤设备15辆窄体电动拖车，20辆宽体电动拖车，15辆电动供水车，设置工作站点18个，拖车专用120W充电站点2个，供水车专用50W充电站2个。
## 基本思路
### state设置
选取各类车辆的工作状态、SOC以及所在工作站点还有各个站点的服务需求类型。
### action设置
充电\加油，待机，工作
### reward设置
基于行驶里程以及各个车辆的电池状态、油量设置。
## 算法
使用Stable_baseline3中的基础PPO算法。
## 配置
- python-3.10.14
- gymnasium-0.29.1
- stable_baseline3-2.4.0a5
