import csv
import numpy as np
import math

class Coordinate:

    # PHI       翻滚角
    # THETA     俯仰角
    # PSI       偏航角
    # ALTITUDE  海拔高度
    # JA        原点经度
    # WA        原点纬度

    F = 35.00000            # 焦距
    ER = 6378137.00000      # 赤道半径
    EJ = 6356725.00000      # 极半径
    PIXEL_SIZE = 0.00451    # 像元尺寸


    def __init__(self, PIC_NO):

        with open('new.csv', 'r') as f:
            _csv_result = list(csv.reader(f))
        print(_csv_result)
        for row in _csv_result:
            if row[0] == PIC_NO:
                self.WA = float(row[2])
                self.JA = float(row[3])
                self.ALTITUDE = float(row[4])
                self.PHI = float(row[5])
                self.THETA = float(row[6])
                self.PSI = float(row[7])

    def resetCoordinate(self, WA, JA, ALTITUDE, PHI, THETA, PSI):
        self.PHI = PHI
        self.THETA = THETA
        self.PSI = PSI
        self.ALTITUDE = ALTITUDE
        self.JA = JA
        self.WA = WA

    def XiangSuCoordinate(self, w, h, x, y):
        '''
            把以左上角为原点的像素坐标系 转换为以图像中心为原点的像素坐标系
            :param w: 图像宽
            :param h: 图像高
            :param x: 左上原点坐标系下x轴坐标
            :param y: 左上原点坐标系下y轴坐标
            :return:  中心原点坐标系下的x y 坐标
        '''
        w = w / 2
        h = h / 2
        x = math.floor(x - w)
        y = math.floor(h - y)
        return x, y

    def getLongitudeAndLatitude(self, X, Y):
        '''
                这个函数用来求像素点(X,Y)对应的经纬度
                :param X: 像素坐标系坐标X
                :param Y: 像素坐标系坐标Y
                :return:  经过转换之后的像素坐标坐标
                    坐标转换过程如下:
                    像素坐标系       机体坐标系        大地坐标系               像素坐标系
                    x    y          x    y           x    y      回到图像    x    y
                    3285,1870 ----> 1870 3285 ----> 3609 -1117  ---------> -1117 3609
                    406  1179 ----> 1179 406  ----> 651  -1063  ---------> -1063 651
            '''
        X, Y = Y, X  # 像素坐标系--->机体坐标系 需要交换X Y 的值
        R1 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(self.PHI)), -np.sin(np.deg2rad(self.PHI))],
                       [0, np.sin(np.deg2rad(self.PHI)), np.cos(np.deg2rad(self.PHI))]])

        R2 = np.array([[np.cos(np.deg2rad(self.THETA)), 0, np.sin(np.deg2rad(self.THETA))],
                       [0, 1, 0],
                       [-np.sin(np.deg2rad(self.THETA)), 0, np.cos(np.deg2rad(self.THETA))]])

        R3 = np.array([[np.cos(np.deg2rad(self.PSI)), -np.sin(np.deg2rad(self.PSI)), 0],
                       [np.sin(np.deg2rad(self.PSI)), np.cos(np.deg2rad(self.PSI)), 0],
                       [0, 0, 1]])

        R_temp = R1.dot(R2)
        R = R_temp.dot(R3)
        #R = np.linalg.inv(R)

        # 机体坐标系----> 大地坐标系
        Alter = R.dot(np.array([[X], [Y], [0]]))
        # 大地坐标系----> 像素坐标系 交换 X Y 即可
        Xb, Yb = Alter[1], Alter[0]

        # Xb Yb则为最终的像素点坐标
        # L 为原点与Xb Yb之间的距离
        L = (Coordinate.PIXEL_SIZE / Coordinate.F) * self.ALTITUDE * np.sqrt((np.power(Xb, 2) + np.power(Yb, 2)))
        # angle 为原点与Xb Yb形成的方位角
        angle = 0.0
        x1, y1, x2, y2 = 0, 0, Xb, Yb
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        angle = angle * 180 / math.pi

        # dx: L向经度方向的投影
        # dy: L向纬度方向的投影
        dx = L * np.sin(np.deg2rad(angle))
        dy = L * np.cos(np.deg2rad(angle))
        ex = Coordinate.EJ + (Coordinate.ER - Coordinate.EJ) * (90 - self.WA) / 90
        ed = ex * np.cos((self.WA * np.pi / 180))
        Jb = dx / ed * 180 / np.pi + self.JA
        Wb = dy / ex * 180 / np.pi + self.WA
        print(Jb, Wb)
        return Jb, Wb




# coordinate1 = Coordinate("06020009")