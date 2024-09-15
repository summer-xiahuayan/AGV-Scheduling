import copy

from matplotlib import pyplot as plt

from MapGenerator import Agv_Length
from Map import dictionary_map,get_path,plot_route_map
from AGV import AGV,Task
import imageio
import os
import math
from tqdm import tqdm




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
    video_path = 'output_video_2agv.mp4'

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
    with imageio.get_writer(video_path, fps=24) as video:

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
            agv=AGV(i,task[i].start,1,[])
            agv.route=get_path(self.map,agv.park_loc,task[i].start)+get_path(self.map,task[i].start,task[i].end)[1:]
            agv.status="busy"
            print(agv.route)
            agv.location=agv.route[0]
            agv.next_loc=agv.route[1]
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

            #if (self.time//self.step)%8==0:
            self.plot()
            print(f"simulate time: {self.time}")

            tempbool=True
            for Key,agv in self.agvs.items():
                assert isinstance(agv,AGV),"obj is not an instance of AGV"
                tempbool=tempbool and (agv.status=="finish")

            self.is_finished=tempbool

        print("Simulate Finished")
        gengrate_video()









if __name__=="__main__":
    # SM=simulate(dictionary_map)
    # tasks={}
    # task1=Task(1,1,1,2,3,1,1)
    # task2=Task(1,1,1,16,14,1,1)
    # tasks[1]=task1
    # tasks[2]=task2
    # SM.creatAGVS(2,tasks)
    # SM.run()
    # #gengrate_video()
    create_gif()





