import copy
import random

from matplotlib import pyplot as plt

from MapGenerator import Agv_Length
from Map import dictionary_map, get_path, plot_route_map, NodeVector, Grid
from AGV import AGV,Task
import imageio
import os
import math
from tqdm import tqdm
from colorama import Fore




def create_gif(fps=24):
    """
    将多张图片合成一个GIF动图。

    :param image_paths: 图片文件路径列表。
    :param gif_path: 输出的GIF文件路径。
    :param fps: GIF的帧率。
    """
    # 指定图片所在的文件夹路径
    folder_path = f'imagedata'

    # 指定输出视频的路径和文件名
    gif_path = 'output_gif_2agv.gif'

    # 获取文件夹中所有图片文件的路径
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    import re



    # 定义一个函数来提取文件名中的最后一个数字
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    # 按最后一个数字排序
    sorted_list = sorted(image_files, key=lambda x: extract_number(x))
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for i,image_path in enumerate(tqdm(sorted_list,disable=False)):
            image = imageio.imread(image_path)
            writer.append_data(image)








def gengrate_video():


    # 指定图片所在的文件夹路径
    folder_path = f'imagedata'

    # 指定输出视频的路径和文件名
    video_path = 'output_video_5agv.mp4'

    # 获取文件夹中所有图片文件的路径
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    import re



    # 定义一个函数来提取文件名中的最后一个数字
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    # 按最后一个数字排序
    sorted_list = sorted(image_files, key=lambda x: extract_number(x))


    #print(sorted_list)
    # 读取图片并生成视频
    with imageio.get_writer(video_path, fps=17) as video:

        for i,image_file in enumerate(tqdm(sorted_list,disable=False)):
            image = imageio.imread(image_file)
            video.append_data(image)
            #print(image_file)

    print("Video have finished")







def calculate_cosine(x, y):
    """
    计算点 (x, y) 与x轴正方向夹角的余弦值。

    参数:
    x -- 点的x坐标
    y -- 点的y坐标

    返回:
    余弦值
    """
    r = math.sqrt(x**2 + y**2)
    return x / r if r != 0 else float('inf')  # 避免除以0

def calculate_sine(x, y):
    """
    计算点 (x, y) 与x轴正方向夹角的正弦值。

    参数:
    x -- 点的x坐标
    y -- 点的y坐标

    返回:
    正弦值
    """
    r = math.sqrt(x**2 + y**2)
    return y / r if r != 0 else float('inf')  # 避免除以0



def acos_full_range(x,y):
    # 计算主反余弦值（0 到 pi）
    theta = math.acos(calculate_cosine(x,y))

    # 检查cos_value的符号，确定正确的象限
    if (x < 0 and y<0) or (x>0 and y<0):
        # 如果余弦值是负的，那么角度可能在第二或第三象限
        theta = theta+math.pi

    return theta

def get_theta(x,y):
    # 计算弧度值
    theta = math.atan2(y, x)

    #返回0——2*pi
    if theta<0:
        return 2*math.pi+theta
    else:
        return theta


def sigmoid_probability(x,k=1,x0=0):
    return 1/(1+math.exp(-k*(x0-x)))

# def linear_probability(num):



class simulate:
    def __init__(self, map):
        self.map=map #地图
        self.time=0  #仿真时间
        self.step=0.1 #仿真时间步
        self.agvs= {} #AGV小车字典
        self.error=0.06
        #self.rotate_error=
        self.frame=0
        self.is_finished=False
        self.global_planning_time=0



    def creatAGVS(self,num,task):
        for i in range(1,num+1):
            agv=AGV(i,task[i].start,i,[])
            agv.route=get_path(self.map,agv.park_loc,task[i].start)+get_path(self.map,task[i].start,task[i].end)[1:]
            agv.status="busy"
            agv.task=task[i]
            print(agv.route)
            agv.location=agv.route[0]
            agv.next_loc=agv.route[1]
            #self.map[agv.location].reservation=True
            #self.map[agv.next_loc].reservation=True
            agv.x=self.map[agv.park_loc].x
            agv.y=self.map[agv.park_loc].y
            temp=copy.deepcopy(agv)
            self.agvs[i]=temp

    def plot(self):
        ax=plot_route_map(self.agvs)
        self.frame+=1
        plt.savefig(f'imagedata\\temp_frame_{self.frame}.png')


    def judje_cross_agv(self,agv):
        invetory=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        assert isinstance(agv,AGV),"obj is not an instance of AGV"
        midpoint=[self.map[agv.location].x+(self.map[agv.next_loc].x-self.map[agv.location].x)/2,
                  self.map[agv.location].y+(self.map[agv.next_loc].y-self.map[agv.location].y)/2]
        assert isinstance(self.map[agv.location],Grid),"self.map[agv.location] is not an instance of AGV"
        cross_neighbour=[point for point in self.map[agv.location].neighbor if
                         math.sqrt((self.map[point].x-midpoint[0])**2+(self.map[point].y-midpoint[1])**2)<Agv_Length and
                         point!=agv.next_loc and point not in invetory]
        if len(cross_neighbour)!=0:
            temp=True
            for point in cross_neighbour:
                    temp=temp and self.map[point].reservation
            return temp
        else :
            return False

    def re_plan_path(self,agv):
        assert isinstance(agv.task,Task),"agv.task is not an instance of Task"
        start=agv.task.start
        end=agv.task.end
        if not self.map[end].reservation and not self.map[self.map[end].neighbor[0]].reservation:
            if start in agv.route[agv.routeid+1:]:
                if not self.map[start].reservation and not self.map[self.map[start].neighbor[0]].reservation:
                    agv.route=agv.route[0:agv.routeid]+get_path(self.map,agv.location,start)+get_path(self.map,start,end)[1:]
                    agv.next_loc=agv.route[agv.routeid+1]
                    self.map[agv.next_loc].reservation=True
                    self.map[agv.next_loc].reserve_agv=agv.ID
                    print(Fore.GREEN+f"AGV:{agv.ID} have re-plan route {agv.route} \n"+Fore.BLACK)
                    agv.waiting_time_work=0
                else:
                    agv.waiting_time_work=0

            else:
                agv.route=agv.route[0:agv.routeid]+get_path(self.map,agv.location,end)
                agv.next_loc=agv.route[agv.routeid+1]
                self.map[agv.next_loc].reservation=True
                self.map[agv.next_loc].reserve_agv=agv.ID
                print(Fore.GREEN+f"AGV:{agv.ID} have re-plan route {agv.route} \n"+Fore.BLACK)
                agv.waiting_time_work=0
        else:
            print(Fore.RED+f"AGV:{agv.ID} because of end reserved have wait {agv.waiting_time_work}s \n"+Fore.BLACK)
            agv.waiting_time_work=0


    def global_planning(self,current_agv):
        #全局启发式路径规划
        other_agv_remain_points=0
        for Key,agv in self.agvs.items():
            assert isinstance(current_agv,AGV),"obj is not an instance of AGV"
            if agv.ID!=current_agv.ID:
                other_agv_remain_points+=len(agv.route[agv.routeid:])
        average_remain_points=other_agv_remain_points/(len(self.agvs)-1)
        replan_probability=sigmoid_probability(average_remain_points,1,5)
        #print(average_remain_points,replan_probability)
        probability=random.random()
        try:
            if probability<=replan_probability:
                assert isinstance(current_agv.task,Task),"agv.task is not an instance of Task"
                start=current_agv.task.start
                end=current_agv.task.end
                if not self.map[end].reservation and not self.map[self.map[end].neighbor[0]].reservation:
                    if start in current_agv.route[current_agv.routeid+1:]:
                        if not self.map[start].reservation and not self.map[self.map[start].neighbor[0]].reservation:
                            current_agv.route=current_agv.route[0:current_agv.routeid]+get_path(self.map,current_agv.location,start)+get_path(self.map,start,end)[1:]
                            current_agv.next_loc=current_agv.route[current_agv.routeid+1]
                            self.map[current_agv.next_loc].reservation=True
                            self.map[current_agv.next_loc].reserve_agv=current_agv.ID
                            print(Fore.GREEN+f"AGV:{current_agv.ID} have global_planning route {current_agv.route} \n"+Fore.BLACK)
                        else:
                            pass
                    else:
                        current_agv.route=current_agv.route[0:current_agv.routeid]+get_path(self.map,current_agv.location,end)
                        current_agv.next_loc=current_agv.route[current_agv.routeid+1]
                        self.map[current_agv.next_loc].reservation=True
                        self.map[current_agv.next_loc].reserve_agv=current_agv.ID
                        print(Fore.GREEN+f"AGV:{current_agv.ID} have global_planning route {current_agv.route} \n"+Fore.BLACK)
                else:
                    pass
        except TypeError as ty:
            print(Fore.RED+f"{TypeError}"+Fore.BLACK)








    def update_agvs(self):
        for Key,agv in self.agvs.items():
            assert isinstance(agv,AGV),"obj is not an instance of AGV"
            #先判断任务是否完成
            if agv.status=='finish':
                continue
            #在旋转到指定位置
            vector=[self.map[agv.next_loc].x-agv.x,self.map[agv.next_loc].y-agv.y]
            rotate_diff=get_theta(vector[0],vector[1])-agv.rotate
            if abs(rotate_diff)>0.08:
                #顺时针，逆时针旋转逻辑
                rotate_direct=1 if rotate_diff>0 else -1
                if rotate_diff>math.pi:
                    rotate_direct=-1
                if rotate_diff<-math.pi:
                    rotate_direct=1
                agv.rotate+=self.step*agv.rotatespeed*rotate_direct
                if agv.rotate>2*math.pi:
                    agv.rotate=agv.rotate%(2*math.pi)
                if agv.rotate<0:
                    agv.rotate=agv.rotate+2*math.pi
                continue
            else:
                agv.rotate=get_theta(vector[0],vector[1])


            assert isinstance(self.map[agv.location],Grid),"self.map[agv.location] is not an instance of Grid"
            assert isinstance(self.map[agv.next_loc],Grid),"self.map[agv.next_loc] is not an instance of Grid"

            # self.global_planning_time+=self.step
            # if self.global_planning_time>Agv_Length*4/agv.speed:
            #     #print("我")
            #     self.global_planning(agv)
            #     self.global_planning_time=0

            #判断下一个目的地是否被别的agv预定，若被预定着等待一定时间啊，若发生死锁则重新规划路径，否者就前进
            if  ((self.map[agv.next_loc].reservation==True and self.map[agv.next_loc].reserve_agv!=agv.ID)
                    or self.judje_cross_agv(agv)):
                try:
                    agv.waiting_time_work+=self.step

                    if agv.waiting_time_work>Agv_Length*2/agv.speed:
                        replan_proibility=random.random()
                        if replan_proibility>0.5:
                            self.re_plan_path(agv)

                        else:
                            print(Fore.RED+f"AGV:{agv.ID} because of random plan have wait {agv.waiting_time_work}s \n"+Fore.BLACK)
                            agv.waiting_time_work=0
                except TypeError as ty:
                    print(Fore.RED+f"{TypeError}"+Fore.BLACK)
                    agv.waiting_time_work=0
                #continue
            else:
                #开始前进
                agv.x+=agv.speed*self.step*calculate_cosine(vector[0],vector[1])
                agv.y+=agv.speed*self.step*calculate_sine(vector[0],vector[1])
                if agv.waiting_time_work!=0:
                    print(Fore.RED+f"AGV:{agv.ID} have wait {agv.waiting_time_work}s \n"+Fore.BLACK)
                    agv.waiting_time_work=0

            #判断是否到达下一个点
            if math.sqrt((self.map[agv.next_loc].x-agv.x)**2+(self.map[agv.next_loc].y-agv.y)**2)<self.error:
                if agv.routeid+1!=len(agv.route)-1: #判断是否到达终点
                    agv.x=self.map[agv.next_loc].x
                    agv.y=self.map[agv.next_loc].y
                    agv.last_loc=agv.location
                    agv.routeid=agv.routeid+1
                    agv.location=agv.next_loc
                    agv.next_loc=agv.route[agv.routeid+1]
                    self.map[agv.last_loc].reservation=False
                    self.map[agv.last_loc].reserve_agv=0  #沒有人預定
                    print(f"AGV:{agv.ID}")
                    print(f"NEXT NODE:{agv.next_loc} x:{self.map[agv.next_loc].x}   y:{self.map[agv.next_loc].y}")
                    print(f"NOW NODE:{agv.location} x:{self.map[agv.location].x}   y:{self.map[agv.location].y}")
                    print(f"NOW LOCATION: x:{agv.x}   y:{agv.y}")
                    print(f"NOW ROTATION: {agv.rotate} {agv.rotate/math.pi}\u03c0 ")
                    print(f"REMAIN ROAD: "+"——>".join(str(point) for point in agv.route[agv.routeid:])+"\n")
                    self.map[agv.location].reservation=True
                    self.map[agv.location].reserve_agv=agv.ID
                    if self.map[agv.next_loc].reservation!=True:
                        self.map[agv.next_loc].reservation=True
                        self.map[agv.next_loc].reserve_agv=agv.ID
                else:
                    agv.status="finish"
                    agv.last_loc= agv.location
                    agv.location=agv.next_loc
                    self.map[agv.last_loc].reservation=False
                    self.map[agv.last_loc].reserve_agv=0
                    print(f"AGV:{agv.ID} have finished \n")













    def run(self):
        while not self.is_finished:
            self.time+=self.step

            self.update_agvs()

            # if (self.time//self.step)%4==0:
            #     self.plot()
           # print(f"simulate time: {self.time}")

            tempbool=True
            for Key,agv in self.agvs.items():
                assert isinstance(agv,AGV),"obj is not an instance of AGV"
                tempbool=tempbool and (agv.status=="finish")

            self.is_finished=tempbool

        print(f"Simulate Finished! Time:{self.time}\n")
       # gengrate_video()









if __name__=="__main__":
    SM=simulate(dictionary_map)
    tasks={}

    task1=Task(1,1,1,25,9,1,1)
    task2=Task(1,1,1,39,10,1,1)
    task3=Task(1,1,1,38,11,1,1)
    task4=Task(1,1,1,35,12,1,1)
    task5=Task(1,1,1,37,13,1,1)
    task6=Task(1,1,1,30,16,1,1)
    task7=Task(1,1,1,41,15,1,1)
    task8=Task(1,1,1,26,177,1,1)
    task9=Task(1,1,1,25,179,1,1)
    task10=Task(1,1,1,27,181,1,1)

    tasks[1]=task1
    tasks[2]=task2
    tasks[3]=task3
    tasks[4]=task4
    tasks[5]=task5
    tasks[6]=task6
    tasks[7]=task7
    tasks[8]=task8
    tasks[9]=task9
    tasks[10]=task10

    SM.creatAGVS(10,tasks)
    SM.run()
    #gengrate_video()
   # create_gif()
    #print(sigmoid_probability(15,1,15))





