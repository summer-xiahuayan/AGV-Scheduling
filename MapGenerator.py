import copy
import csv
from matplotlib import pyplot as plt
import math
Agv_Length=1.8
Factory_Edge_X=[17.4,17.4,34.1887,42.3887,48.5887,55.6387,62.9045,62.9045,55.6387,48.5887,42.3887,34.1887,28.5,27.5,26.5,25.5,24.5]
Factory_Edge_Y=[-14.26,-18.75,-23.2283,-23.2283,-23.2283,-23.2283,-23.2283,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9]
Error=0.1


def get_edge_neighbour(x,y,innerxlist,innerylist):
    dx=[int(abs(x-xi)) for xi in innerxlist]
    dy=[int(abs(y-yi)) for yi in innerylist]
    manhattan_distance=[dx[i]+dy[i] for i in range(len(dx))]
    return manhattan_distance.index(min(manhattan_distance))


def get_8_neighbour(innerxlist,innerylist):
    neighbourlist=[]
    for i in range(len(innerxlist)):
        point_neighbourlist=[]
        for j in range(len(innerxlist)):
            distance=math.sqrt((innerxlist[j]-innerxlist[i])**2+(innerylist[j]-innerylist[i])**2)
            if distance>0 and distance <=math.sqrt(2)*Agv_Length+Error:
                point_neighbourlist.append(j)
        temp=copy.deepcopy(point_neighbourlist)
        neighbourlist.append(temp)
    return neighbourlist


def Writecsv(Factory_Edge_X,Factory_Edge_Y,innerxlist,innerylist):
    headers = ['id', 'x', 'y', 'type', 'neighbour', 'clashed', 'entrance']
    innertoedge= {}
    # 定义CSV文件的数据
    rows = []
    for i in range(len(Factory_Edge_X)):
        row=[]
        row.append(i+1)
        row.append(Factory_Edge_X[i])
        row.append(Factory_Edge_Y[i])
        row.append(1)
        id=get_edge_neighbour(Factory_Edge_X[i],Factory_Edge_Y[i],innerxlist,innerylist)+len(Factory_Edge_X)+1
        row.append(id)
        #用于存储inner节点到edge的邻居信息
        innertoedge[id]=i+1
        row.append(-1)
        row.append(i+1)
        temp=copy.deepcopy(row)
        rows.append(temp)

    neighbourlist=get_8_neighbour(innerxlist,innerylist)
    for i in range(len(Factory_Edge_X),len(Factory_Edge_X)+len(innerxlist)):
        id=i-len(Factory_Edge_X)
        row=[]
        row.append(i+1)
        row.append(innerxlist[id])
        row.append(innerylist[id])
        row.append(1)
        neighbour=','.join(str(num+len(Factory_Edge_X)+1) for num in neighbourlist[id])
        if i+1 in innertoedge.keys():
            neighbour+=','+str(innertoedge[i+1])
        row.append(neighbour)
        row.append(-1)
        row.append(i+1)
        temp=copy.deepcopy(row)
        rows.append(temp)


    # 指定CSV文件的名称
    filename = 'data.csv'
    # 打开文件并写入数据
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入头部
        writer.writerows(rows)    # 写入数据行

    headers = ['id','inventory']
    # 定义CSV文件的数据
    rows = []
    for i in range(len(Factory_Edge_X)+len(innerxlist)):
        row=[]
        row.append(i+1)
        row.append(-1)
        temp=copy.deepcopy(row)
        rows.append(temp)
    # 指定CSV文件的名称
    filename = 'datainventory.csv'
    # 打开文件并写入数据
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入头部
        writer.writerows(rows)    # 写入数据行


def Generator(Agv_Length,xlist,ylist):
    leftupx=min(xlist)
    leftupy=max(ylist)
    rightdownx=max(xlist)
    rightdowny=min(ylist)
    gridx=leftupx
    gridxlist=[]
    gridylist=[]
    while(gridx<rightdownx+Agv_Length):
        tempx=copy.deepcopy(gridx)
        gridy=leftupy
        while(gridy>rightdowny-Agv_Length):
            tempy=copy.deepcopy(gridy)
            gridxlist.append(tempx)
            gridylist.append(tempy)
            gridy-=Agv_Length
        gridx+=Agv_Length

    # 将顶点坐标转换为多边形格式
    polygon = list(zip(Factory_Edge_X, Factory_Edge_Y))
    innerxlist=[]
    innerylist=[]
    for i in range(len(gridxlist)):
        x=gridxlist[i]
        y=gridylist[i]
        if is_point_in_polygon(x, y, polygon):
            innerxlist.append(x)
            innerylist.append(y)


    return innerxlist,innerylist



def is_point_in_polygon(x, y, polygon):
    """
    判断点是否在多边形内
    :param x: 点的x坐标
    :param y: 点的y坐标
    :param polygon: 多边形顶点坐标列表，格式为[(x1, y1), (x2, y2), ...]
    :return: 点在多边形内返回True，否则返回False
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y < max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x < xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside









if __name__=="__main__":
    x,y=Generator(Agv_Length,Factory_Edge_X,Factory_Edge_Y)
    print(x,y)

    for i in range(len(x)):
        xpoint=x[i]
        ypoint=y[i]
        plt.plot(xpoint, ypoint,'.', markersize=10, color="black")
    for i in range(len(Factory_Edge_X)):
        xpoint=Factory_Edge_X[i]
        ypoint=Factory_Edge_Y[i]
        plt.plot(xpoint, ypoint,'.', markersize=10, color="blue")
    plt.show()

    Writecsv(Factory_Edge_X,Factory_Edge_Y,x,y)



