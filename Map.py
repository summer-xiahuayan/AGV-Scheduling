import copy

import pandas as pd
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque
import logging
import math
from MapGenerator import Agv_Length,Agv_Width
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

from AGV import AGV
from logger import logger

# 统一读入配置文件
# df_Grid = pd.read_csv(f"FactoryMap.csv")
# df_Inventory = pd.read_csv(f"FactoryInventory.csv")

df_Grid = pd.read_csv(f"data.csv")
df_Inventory = pd.read_csv(f"datainventory.csv")

COLOR = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
     'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
     'navajowhite','navy', 'sandybrown']

# 节点类的形式储存地图数据
class Grid:
    def __init__(self, grid_id, x, y, grid_type, neighbor, state, conflict_list, entrance):
        self.id = grid_id  # 地图节点的编号
        self.x = x  # 节点横坐标
        self.y = y  # 节点纵坐标
        self.type = grid_type  # 节点的类型 道路/停靠点为1，货位为2
        self.neighbor = neighbor
        self.entrance = entrance  # 库位的入口节点（道路节点则为其本身）
        self.reservation = False
        self.reserve_agv = 0
        if state == -1:
            self.state = None
        else:
            self.state = state  # 节点的库存配置状态，None代表道路节点，0 代表无货，>= 1 代表有货，-1 代表被预约
        if conflict_list == [-1]:
            self.conflict = []
        else:
            self.conflict = conflict_list  # 每个节点的冲突节点列表，None为路径节点，列表为库位节点

# 树结构，用于A-star回溯路径
class NodeVector:
    node = None  # 当前的节点
    frontNode = None  # 当前节点的前置节点
    childNodes = None  # 当前节点的后置节点们
    g = 0  # 起点到当前节点所经过的距离
    h = 0  # 启发值

    def __init__(self, node):
        self.node = node
        self.childNodes = []

    @property
    def f(self):
        return self.h
        # return self.g*0.2 + self.h*0.8

    def calcGH(self, target):
        """target代表目标节点，Grid类"""
        self.g = self.frontNode.g + \
                 abs(self.node.x - self.frontNode.node.x) + \
                 abs(self.node.y - self.frontNode.node.y)
        dx = abs(target.x - self.node.x)
        dy = abs(target.y - self.node.y)
        self.h = dx + dy

    def eulerDS(self, targetstart,targetend):
        """target代表目标节点，Grid类"""
        self.g = math.sqrt((self.node.x-targetstart.x)**2+(self.node.y-targetstart.y)**2)
        self.h = math.sqrt((self.node.x-targetend.x)**2+(self.node.y-targetend.y)**2)

    def __lt__(self, other):
        return self.f < other.f



class A_star:
    def __init__(self, grids, start_point, end_point):
        self.grids = grids
        self.start_point = start_point
        self.end_point = end_point

        self.open_set = PriorityQueue()
        self.closed_set = deque()

        self.found_end_node = None  # 寻找到的终点，用于判断算法结束

    def is_closed(self, grid_id):
        """判断id节点是否在closed_set中，在返回True，不在返回False"""
        return grid_id in [vector.node.id for vector in self.closed_set]

    def get_route(self):
        """输出最优路径"""
        route = [self.found_end_node.node.id]
        current = self.found_end_node
        while True:
            current = current.frontNode
            route.append(current.node.id)
            if current.node.id == self.start_point.id:
                break
        return list(reversed(route))

    def process(self):
        # 初始化open集合，并把起始点放入
        self.open_set.put((0, NodeVector(self.start_point)))

        # 开始迭代，直到找到终点，或找完了所有能找的点
        while self.found_end_node is None and not self.open_set.empty():
            # 选出下一个合适的点
            vector = self.popLowGHNode()  # vector: NodeVector
            # 获取合适点周围所有的邻居
            neighbors = [self.grids[i] for i in vector.node.neighbor if not self.is_closed(i)]
            for neighbor in neighbors:
                # 初始化邻居，并计算g和h
                child = NodeVector(neighbor)
                child.frontNode = vector
                #曼哈顿距离
                #child.calcGH(self.end_point)
                #欧拉距离
                child.eulerDS(self.start_point,self.end_point)
                if not child.node.reservation:
                    vector.childNodes.append(child)

                    # 添加到open集合中
                    assert isinstance(child.f, float), f'{child.f}'
                    # self.open_set.put((child, child.f))
                    self.open_set.put([child.f, child])


                # 找到终点
                if neighbor == self.end_point:
                    self.found_end_node = child

        if self.found_end_node is None:
            logger.warning(f'无法找到从{self.start_point.id}到{self.end_point.id}的路')
            return None
        else:
            route = self.get_route()
            return route

    # A*，寻找f = g + h最小的节点
    def popLowGHNode(self):
        found_node = self.open_set.get()[1]
        self.closed_set.append(found_node)
        return found_node

# 初始化函数读入配置文件
def create_map(df_grid, df_inventory):
    dict_map = {}
    for i in range(df_grid.shape[0]):
        neighbour = df_grid.iloc[i, 4].split(',')
        neighbour = [int(x) for x in neighbour]
        # conflict = df_grid.iloc[i, 5].split(',')
        # conflict = [int(x) for x in conflict]
        grid_tmp = Grid(df_grid.iloc[i, 0], df_grid.iloc[i, 1], df_grid.iloc[i, 2], df_grid.iloc[i, 3],
                        neighbour, df_inventory.iloc[i, 1], [-1], df_grid.iloc[i, 6])
        dict_map[i+1] = grid_tmp
    return dict_map

def get_path(map,start,end):
    return A_star(grids=map, start_point=map[start], end_point=map[end]).process()

def plot_map(dictionary_map,start,end):
    #fig = plt.figure()
    fig,ax= plt.subplots(figsize=(30, 10))  # 设置DPI为100
    for i in range(1,len(dictionary_map)+1):
        #print(i)
        neighbour = dictionary_map[i].neighbor
        neighbour = [int(x) for x in neighbour]
        if dictionary_map[i].type==1:
            color="black"
        elif dictionary_map[i].type==2:
            color="red"
        else:
            color="yellow"

        ax.plot(dictionary_map[i].x, dictionary_map[i].y,'.', markersize=10, color=color)
        ax.text(dictionary_map[i].x, dictionary_map[i].y, str(i), ha='right', va='bottom')
        for neighbouri in neighbour:
            x = [dictionary_map[i].x,dictionary_map[neighbouri].x]
            y = [dictionary_map[i].y,dictionary_map[neighbouri].y]
            # 使用plot函数画线
            ax.plot(x, y, '-k')  # '-r' 表示红色的实线
            ax.arrow(dictionary_map[i].x, dictionary_map[i].y,
                      (dictionary_map[neighbouri].x-dictionary_map[i].x)*0.3,
                      (dictionary_map[neighbouri].y-dictionary_map[i].y)*0.3, head_width=0.2, head_length=0.2, fc='lightblue', ec='black')


   #  task_1_get =get_path(dictionary_map,start,end)
   # # print(task_1_get)
   #  i=1
   #  for point in task_1_get:
   #
   #      if i==len(task_1_get):
   #          break
   #      #print(point)
   #      print(task_1_get[i])
   #      x = [dictionary_map[point].x, dictionary_map[task_1_get[i]].x]
   #      y = [dictionary_map[point].y, dictionary_map[task_1_get[i]].y]
   #      # 使用plot函数画线
   #      plt.plot(x, y, '-b',linewidth=3)  # '-r' 表示红色的实线
   #      i+=1
   #  plt.yticks(size=40, fontproperties='Times New Roman')
   #  plt.xticks(size=40, fontproperties='Times New Roman')
   #  plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    #plt.savefig(f'map.png')

    return fig,ax




def plot_route_map(agvs):
    # 创建一个图形和坐标轴
    #fig,ax= plt.subplots()
    fig,ax=plot_map(dictionary_map,0,0)

    for Key,agv in agvs.items():
        assert isinstance(agv,AGV),"obj is not an instance of AGV"
        crossx=math.sqrt((Agv_Length/2)**2+(Agv_Width/2)**2)*math.cos(agv.rotate+math.atan(Agv_Width/Agv_Length))
        crossy=math.sqrt((Agv_Length/2)**2+(Agv_Width/2)**2)*math.sin(agv.rotate+math.atan(Agv_Width/Agv_Length))
        crossx_=math.sqrt((Agv_Length/2)**2+(Agv_Width/2)**2)*math.cos(agv.rotate-math.atan(Agv_Width/Agv_Length))
        crossy_=math.sqrt((Agv_Length/2)**2+(Agv_Width/2)**2)*math.sin(agv.rotate-math.atan(Agv_Width/Agv_Length))
        forward_left=(crossx+agv.x,crossy+agv.y)
        backword_right=(agv.x-crossx,agv.y-crossy)
        forward_right=(crossx_+agv.x,crossy_+agv.y)
        backword_left=(agv.x-crossx_,agv.y-crossy_)

        head_backword_right=(forward_right[0]+(backword_right[0]-forward_right[0])*0.3,forward_right[1]+(backword_right[1]-forward_right[1])*0.3)
        head_backword_left=(forward_left[0]+(backword_left[0]-forward_left[0])*0.3,forward_left[1]+(backword_left[1]-forward_left[1])*0.3)



        x = [agv.x, dictionary_map[agv.next_loc].x]
        y = [agv.y, dictionary_map[agv.next_loc].y]
        # 使用plot函数画线
        ax.plot(x, y, COLOR[Key],linewidth=3)

        task_1_get=agv.route[agv.routeid+1:]
        i=1
        for point in task_1_get:

            if i==len(task_1_get):
                break
             #print(point)
            #print(task_1_get[i])
            x = [dictionary_map[point].x, dictionary_map[task_1_get[i]].x]
            y = [dictionary_map[point].y, dictionary_map[task_1_get[i]].y]
             # 使用plot函数画线
            ax.plot(x, y, COLOR[Key],linewidth=3)  # '-r' 表示红色的实线
            i+=1

        # 定义矩形四个角的坐标
        # 假设这四个点分别是矩形的左上角、右上角、右下角、左下角
        points = [forward_left, forward_right,backword_right ,backword_left]
        head_point=[forward_left,forward_right,head_backword_right,head_backword_left]
        # print(points)
        # print(head_point)
        # 创建一个多边形
        polygon = patches.Polygon(points, closed=True, edgecolor="black", facecolor=COLOR[Key])
        # 将多边形添加到坐标轴
        ax.add_patch(polygon)

        # 创建一个多边形
        polygon = patches.Polygon(head_point, closed=True, edgecolor="black", facecolor=COLOR[0])
        # 将多边形添加到坐标轴
        ax.add_patch(polygon)

    # # 设置y轴的范围
    ax.set_ylim(-24, -8)
    # # 设置y轴的范围
    ax.set_xlim(15, 65)
    return ax
        #plt.show()








dictionary_map = create_map(df_Grid, df_Inventory)
#mapimg=plot_map(dictionary_map,0,0)

if __name__=="__main__":

   # plot_map(dictionary_map,3,8)
   plot_route_map(dictionary_map,"agvs")










