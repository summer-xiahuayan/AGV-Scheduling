import math
import Map
from logger import logger
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

class AGV:
    def __init__(self, num, park_grid,map):
        self.ID = num
        self.status = 'idle'  # idle, busy, put,down, block
        self.task = None  # 实时任务编号task_list[0]
        self.tasklist = []  # 已规划任务清单
        self.route = []  # 从 任务库 中导出对应任务编号的任务路径序列，并将其赋予叉车route_list[self.task]
        self.routeid=0
        self.location = park_grid  # 位置节点编号self.route[2]
        self.last_loc = None  # 序列中上一目标位置编号self.route[0]
        self.next_loc = None  # 序列中下一目标位置编号self.route[2]
        self.park_loc = park_grid
        self.is_load = 0  # 0表示车上空载，1表示车上有货
        self.speed=1
        self.x=0
        self.y=0
        self.rotatespeed=2*math.pi/8
        self.rotate=0
        self.waiting_time_work=0


    def add_task_to_tasklist(self,task):
        self.tasklist.append(task)

    def set_task_from_tasklist(self):
        if len(self.tasklist)!=0:
            self.task=self.tasklist.pop(0)
            logger.info(f"AGV{self.ID} set task success")
        else:
            logger.warning(f"AGV{self.ID} do not have tasks")

    def set_route_from_task(self,map):
        assert isinstance(map,dict),"agv.task is not an instance of Task"
        start=self.task.start
        end=self.task.end
        self.x=map[self.park_loc].x
        self.y=map[self.park_loc].y
        self.routeid=0
        self.route=[]
        if not map[end].reservation and not map[map[end].neighbor[0]].reservation:
            if start!=self.location:
                if not map[start].reservation and not map[map[start].neighbor[0]].reservation:
                    self.route= Map.get_path(map, self.location, start) + Map.get_path(map, start, end)[1:]
                    self.location=self.route[self.routeid]
                    self.last_loc = self.route[self.routeid]
                    self.next_loc=self.route[self.routeid+1]
                    map[self.next_loc].reservation=True
                    map[self.next_loc].reserve_agv=self.ID
                    logger.info(f"AGV:{self.ID} have get route {self.route} \n")
                    return True
                else:
                    logger.warning(f"AGV:{self.ID} start node {start} was reserved,reserved people:{map[start].reserve_agv}\n")
                    logger.warning(f"AGV:{self.ID} start neighbor node {map[start].neighbor[0]} "
                                   f"was reserved,reserved people:{map[map[start].neighbor[0]].reserve_agv}\n")
                    return False
            else:
                self.route=Map.get_path(map, start, end)
                self.location=self.route[self.routeid]
                self.last_loc = self.route[self.routeid]
                self.next_loc=self.route[self.routeid+1]
                map[self.next_loc].reservation=True
                map[self.next_loc].reserve_agv=self.ID
                logger.info(f"AGV:{self.ID} have get route {self.route} \n")
                return True
        else:
            logger.warning(f"AGV:{self.ID} end node {end} was reserved \n")
            return False


    def set_route_to_task_start(self):
        assert isinstance(map,dict),"agv.task is not an instance of Task"
        start=self.task.start
        if start!=self.location:
            if not map[start].reservation and not map[map[start].neighbor[0]].reservation:
                self.route= self.route[0:self.routeid] + Map.get_path(map, self.location, start)
                self.next_loc=self.route[self.routeid+1]
                map[self.next_loc].reservation=True
                map[self.next_loc].reserve_agv=self.ID
                logger.info(f"AGV:{self.ID} have get route to task start:{self.route} \n")
            else:
                logger.warning(f"AGV:{self.ID} start node {start} was reserved \n")
        else:
            self.route=self.route[0:self.routeid] + Map.get_path(map, self.location, start)
            self.next_loc=self.route[self.routeid+1]
            map[self.next_loc].reservation=True
            map[self.next_loc].reserve_agv=self.ID
            logger.info(f"AGV:{self.ID} have get to task start:{self.route} \n")


    def set_route_to_task_end(self):
        assert isinstance(map,dict),"agv.task is not an instance of Task"
        end=self.task.end
        if end!=self.location:
            if not map[end].reservation and not map[map[end].neighbor[0]].reservation:
                self.route= self.route[0:self.routeid] + Map.get_path(map, self.location, end)
                self.next_loc=self.route[self.routeid+1]
                map[self.next_loc].reservation=True
                map[self.next_loc].reserve_agv=self.ID
                logger.info(f"AGV:{self.ID} have get route to task end:{self.route} \n")
            else:
                logger.warning(f"AGV:{self.ID} end node {end} was reserved \n")
        else:
            self.route=self.route[0:self.routeid] + Map.get_path(map, self.location, end)
            self.next_loc=self.route[self.routeid+1]
            map[self.next_loc].reservation=True
            map[self.next_loc].reserve_agv=self.ID
            logger.info(f"AGV:{self.ID} have get to task end:{self.route} \n")

    def rotate_step(self,step,rotate_direct):
        self.rotate+=step*self.rotatespeed*rotate_direct
        if self.rotate>2*math.pi:
            self.rotate=self.rotate%(2*math.pi)
        if self.rotate<0:
            self.rotate=self.rotate+2*math.pi

    def go_forward_step(self,step,direct_vector):
        self.x+=self.speed*step*calculate_cosine(direct_vector[0],direct_vector[1])
        self.y+=self.speed*step*calculate_sine(direct_vector[0],direct_vector[1])


    # def put_goods(self,step):
    #     assert isinstance(self.task,Task),"agv.task is not an instance of Task"
    #     put_time=self.task.put_time
    #     while put_time>0:





class Task:
    def __init__(self, num, arrival, task_type, start, end, state, car):
        self.num = num
        self.arrival = arrival  # 任务到达时间
        self.type = task_type  # streamline(online)记为 1 & warehouse(offline)记为 0
        self.start = start  # 任务取货位置
        self.end = end  # 任务卸货位置，None表示未分配库位
        self.state = state  # 任务状态，0表示未分配，1表示被分配但未执行，2表示正在执行，3表示已完成
        self.car = car  # 表示分配的叉车序号，None表示未被分配
        self.route_seq = None  # 储存任务对应的路径序列，从起点到取货点到卸货点为完整序列
        self.put_time=0
        self.down_time=0