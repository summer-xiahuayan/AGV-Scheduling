import copy
from MapGenerator import Agv_Length
from Map import dictionary_map,get_path,plot_route_map
from AGV import AGV,Task
import imageio
import os
import math


def gengrate_video():


    # 指定图片所在的文件夹路径
    folder_path = f'imagedata'

    # 指定输出视频的路径和文件名
    video_path = 'output_video.mp4'

    # 获取文件夹中所有图片文件的路径
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

    # 按文件名排序，确保图片按正确的顺序添加到视频中
    image_files.sort()  # 可以根据需要添加更复杂的排序逻辑

    # 读取图片并生成视频
    with imageio.get_writer(video_path, fps=24) as video:
        for image_file in image_files:
            image = imageio.imread(image_file)
            video.append_data(image)

    print("视频已生成")







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
            agv=AGV(i,task.start,1,[])
            agv.route=get_path(self.map,agv.park_loc,task.start)+get_path(self.map,task.start,task.end)[1:]
            print(agv.route)
            agv.location=agv.route[0]
            agv.next_loc=agv.route[1]
            agv.x=self.map[agv.park_loc].x
            agv.y=self.map[agv.park_loc].y
            temp=copy.deepcopy(agv)
            self.agvs[i]=temp

    def plot(self):
        for Key,agv in self.agvs.items():
            assert isinstance(agv,AGV),"obj is not an instance of AGV"
            plt=plot_route_map(self.map,agv.route)
            plt.plot(agv.x, agv.y,'.', markersize=10, color="red")
            plt.savefig(f'imagedata\\temp_frame_{self.frame}.png')
            self.frame+=1



    def run(self):
        while not self.is_finished:
            self.time+=self.step
            for Key,agv in self.agvs.items():
                assert isinstance(agv,AGV),"obj is not an instance of AGV"
                direct_vectory=[agv.x-self.map[agv.location].x,
                                agv.y-self.map[agv.location].y]

                vector=[self.map[agv.next_loc].x-agv.x,self.map[agv.next_loc].y-agv.y]

                #print(vector)
                agv.x+=agv.speed*self.step*calculate_cosine(vector[0],vector[1])
                agv.y+=agv.speed*self.step*calculate_sine(vector[0],vector[1])
                if math.sqrt((self.map[agv.next_loc].x-agv.x)**2+(self.map[agv.next_loc].y-agv.y)**2)<self.error:
                    agv.x=self.map[agv.next_loc].x
                    agv.y=self.map[agv.next_loc].y
                    agv.last_loc=agv.location
                    agv.routeid=agv.routeid+1
                    agv.location=agv.next_loc
                    agv.next_loc=agv.route[agv.routeid+1]
                    print(f"next node:{agv.next_loc} x:{self.map[agv.next_loc].x}   y:{self.map[agv.next_loc].y}")
                    print(f"now node:{agv.location} x:{self.map[agv.location].x}   y:{self.map[agv.location].y}")
                    if agv.routeid==len(agv.route)-1:
                        self.is_finished=True
                if (self.time//self.step)%2==0:
                    self.plot()
                    print(f"simulate time: {self.time}")
        print("Simulate Finished")









if __name__=="__main__":
    SM=simulate(dictionary_map)
    task=Task(1,1,1,3,8,1,1)
    SM.creatAGVS(1,task)
    SM.run()







