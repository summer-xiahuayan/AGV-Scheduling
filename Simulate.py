import copy

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

    return theta



class simulate:
    def __init__(self, map):
        self.map=map #地图
        self.time=0  #仿真时间
        self.step=0.1 #仿真时间步
        self.agvs= {} #AGV小车字典
        self.error=0.06
        self.frame=0
        self.is_finished=False



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


    def update_agvs(self):
        for Key,agv in self.agvs.items():
            assert isinstance(agv,AGV),"obj is not an instance of AGV"
            if agv.status=='finish':
                continue
            vector=[self.map[agv.next_loc].x-agv.x,self.map[agv.next_loc].y-agv.y]
            #旋转逻辑
            rotate_diff=get_theta(vector[0],vector[1])-agv.rotate
            if abs(rotate_diff)>0.08:
                rotate_direct=1 if rotate_diff>0 else -1
                agv.rotate+=self.step*agv.rotatespeed*rotate_direct
                continue
            else:
                agv.rotate=get_theta(vector[0],vector[1])

            assert isinstance(self.map[agv.location],Grid),"self.map[agv.location] is not an instance of Grid"
            assert isinstance(self.map[agv.next_loc],Grid),"self.map[agv.next_loc] is not an instance of Grid"

            if  self.map[agv.next_loc].reservation==True and self.map[agv.next_loc].reserve_agv!=agv.ID:
                agv.waiting_time_work+=self.step
                if agv.waiting_time_work>Agv_Length*2/agv.speed/self.step:
                    assert isinstance(agv.task,Task),"agv.task is not an instance of Task"
                    start=agv.task.start
                    end=agv.task.end
                    if start in agv.route:
                        agv.route=agv.route[0:agv.routeid]+get_path(self.map,agv.location,start)+get_path(self.map,start,end)[1:]
                    else:
                        agv.route=agv.route[0:agv.routeid]+get_path(self.map,agv.location,end)
                    print(f"AGV:{agv.ID} path have re-plan {agv.route}")
                continue
            else:
                if agv.waiting_time_work!=0:
                    print(Fore.RED+f"AGV:{agv.ID} have wait {agv.waiting_time_work}s"+Fore.BLACK)
                    agv.waiting_time_work=0


            agv.x+=agv.speed*self.step*calculate_cosine(vector[0],vector[1])
            agv.y+=agv.speed*self.step*calculate_sine(vector[0],vector[1])
            if math.sqrt((self.map[agv.next_loc].x-agv.x)**2+(self.map[agv.next_loc].y-agv.y)**2)<self.error:
                if agv.routeid+1!=len(agv.route)-1: #判断是否到达终点
                    agv.x=self.map[agv.next_loc].x
                    agv.y=self.map[agv.next_loc].y
                    agv.last_loc=agv.location
                    agv.routeid=agv.routeid+1
                    agv.location=agv.next_loc
                    agv.next_loc=agv.route[agv.routeid+1]

                    self.map[agv.location].reservation=True
                    self.map[agv.next_loc].reservation=True
                    self.map[agv.location].reserve_agv=agv.ID
                    self.map[agv.next_loc].reserve_agv=agv.ID

                    self.map[agv.last_loc].reservation=False
                    self.map[agv.last_loc].reserve_agv=0  #沒有人預定


                    print(f"AGV:{agv.ID}")
                    print(f"next node:{agv.next_loc} x:{self.map[agv.next_loc].x}   y:{self.map[agv.next_loc].y}")
                    print(f"now node:{agv.location} x:{self.map[agv.location].x}   y:{self.map[agv.location].y}")
                else:
                    agv.status="finish"
                    print(f"AGV:{agv.ID} have finished")

    def run(self):
        while not self.is_finished:
            self.time+=self.step

            self.update_agvs()

            if (self.time//self.step)%4==0:
                self.plot()
           # print(f"simulate time: {self.time}")

            tempbool=True
            for Key,agv in self.agvs.items():
                assert isinstance(agv,AGV),"obj is not an instance of AGV"
                tempbool=tempbool and (agv.status=="finish")

            self.is_finished=tempbool

        print("Simulate Finished")
        gengrate_video()









if __name__=="__main__":
    SM=simulate(dictionary_map)
    tasks={}

    task1=Task(1,1,1,6,15,1,1)
    task2=Task(1,1,1,8,13,1,1)
    task3=Task(1,1,1,10,14,1,1)
    task4=Task(1,1,1,12,17,1,1)
    task5=Task(1,1,1,14,6,1,1)

    tasks[1]=task1
    tasks[2]=task2
    tasks[3]=task3
    tasks[4]=task4
    tasks[5]=task5

    SM.creatAGVS(5,tasks)
    SM.run()
    #gengrate_video()
   # create_gif()





