import math


class AGV:
    def __init__(self, num, simul_loc, park_grid, task_list):
        self.ID = num
        self.status = 'idle'  # idle, busy, down, block具体状态代表的含义还需要统一
        self.task = None  # 实时任务编号task_list[0]
        self.tasklist = task_list  # 已规划任务清单
        self.route = []  # 从 任务库 中导出对应任务编号的任务路径序列，并将其赋予叉车route_list[self.task]
        self.routeid=0
        self.location = simul_loc  # 位置节点编号self.route[2]
        self.last_loc = None  # 序列中上一目标位置编号self.route[0]
        self.next_loc = None  # 序列中下一目标位置编号self.route[2]
        self.park_loc = park_grid
        self.task_status = None  # 'get', 'put', 'return', None
        self.is_load = 0  # 0表示车上空载，1表示车上有货
        self.end_state = (0, False)  # 车辆任务结束状态，第一项表示结束时间，第二项表示车辆是否已完成所有任务
        self.waiting_time_park = 0  # 表示车辆因为控制策略而等待的时间（在停靠点等待）
        self.waiting_time_work = 0  # 表示车辆因为控制策略而等待的时间（行进过程中）
        self.speed=1
        self.x=0
        self.y=0
        self.rotatespeed=2*math.pi/8
        self.rotate=0
        self.rotation=0
        self.neighbour=-1
        self.lastneighbour=None

    def get_process(self, task_list: dict):
        if self.task is None:
            return 0
        else:
            task_end = task_list[self.task].end
            end_idx = self.route[1:].index(task_end)
            return round(((len(task_list[self.task].route_seq) - end_idx) / len(task_list[self.task].route_seq))*100)/100

    @property
    def waiting_time(self):
        return self.waiting_time_park + self.waiting_time_work

    def get_location(self):  # 考虑可以放在Problem类，依据AGV编号来获取实时位置
        return self.location

    def get_status(self):  # 获取AGV的实时状态
        return self.status


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