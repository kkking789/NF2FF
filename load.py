import numpy as np
from scipy.io import loadmat
import warnings
from math import e
import re

path1 = r"nf_H.fld"
path2 = r"nf_E.fld"
MATLAB = "matlab"
HFSS = "hfss"



class DataLoader:
    def __init__(self, path1_=path1, path2_=path2, model=MATLAB):
        self.data1 = []
        self.data2 = []
        if model == MATLAB:
            self.orgdata1 = loadmat(path1_)["sdata"]
            self.orgdata2 = loadmat(path2_)["sdata"]
            if self.orgdata1["fstart"][0][0] == self.orgdata2["fstart"][0][0]\
                and self.orgdata1["fstop"][0][0] == self.orgdata2["fstop"][0][0]\
                and self.orgdata1["fpoints"][0][0] == self.orgdata2["fpoints"][0][0]\
                and self.orgdata1["xpoints"][0][0] == self.orgdata2["xpoints"][0][0]\
                and self.orgdata1["ypoints"][0][0] == self.orgdata2["ypoints"][0][0]:

                self.data1 = self.orgdata1["s21"][0][0]
                self.data2 = self.orgdata2["s21"][0][0]
                self.datasize = (self.orgdata1["xpoints"][0][0][0][0],self.orgdata1["ypoints"][0][0][0][0])
                self.xstep = self.orgdata1["x_step"][0][0][0][0]
                self.ystep = self.orgdata1["y_step"][0][0][0][0]
                self.xpoints = self.orgdata1["xpoints"][0][0][0][0]
                self.ypoints = self.orgdata1["ypoints"][0][0][0][0]

                self.frequency = self.orgdata1["freq"][0][0][0]
                self.fpoints = self.orgdata1["fpoints"][0][0][0][0]
                self.fstart = self.orgdata1["fstart"][0][0][0][0]
                self.fstop = self.orgdata1["fstop"][0][0][0][0]
                self.cnt = 0
            else:
                raise ValueError("(╬▔皿▔)╯Data source is different")
        elif model == HFSS:
            with open(path1_, "r") as file1:
                lines = file1.readlines()
                for line in lines:
                    if line[0] == "G":
                        data = [float(mid) for mid in re.findall(r'-?\d+\.?\d*', line)]
                        self.xpoints = int((data[3]-data[0])/data[6]+1)
                        self.xstep = int(data[6])/1000
                        self.ypoints = int((data[4]-data[1])/data[7]+1)
                        self.ystep = int(data[7])/1000
                        continue
                    elif line[0] == "X":
                        continue
                    self.data1.append([float(mid) for mid in line.split()])
            with open(path2_, "r") as file2:
                lines = file2.readlines()
                for line in lines:
                    if line[0] == "G" or line[0] == "X":
                        continue
                    self.data2.append([float(mid) for mid in line.split()])

        self.model = model


    def get_data(self, freq=-1, index=-1):
        """
        input:
            freq/index: 输出指定频点, 仅对matlab数据文件有效
        output:
            matlab数据文件: 文件1数据, 文件2数据, 数据所属的频点
            HFSS数据文件: 磁场数据, 电场数据
        """
        if self.model == MATLAB:
            pol1_data = np.zeros(self.datasize, dtype=complex)
            pol2_data = np.zeros(self.datasize, dtype=complex)
            if freq == -1:
                if index == -1:
                    fpoint = self.cnt
                    self.cnt += 1
                elif -1 < index < self.fpoints:
                    fpoint = index
                else:
                    fpoint = 0
                    warnings.warn("φ(*￣0￣) Index is out of range", Warning)
                if self.cnt == self.fpoints:
                    self.cnt = 0
            elif self.fstart < freq < self.fstop:
                fpoint = int((freq-self.fstart)/(self.fstop-self.fstart)*self.fpoints)
            else:
                warnings.warn("φ(*￣0￣) Frequency is out of range", Warning)
                fpoint = self.cnt

            for i in range(0,self.datasize[0]):
                for j in range(0,self.datasize[1]):
                    pol1_data[i][j] = self.data1[i,j][0][fpoint]
                    pol2_data[i][j] = self.data2[i,j][0][fpoint]

            return np.array(pol1_data), np.array(pol2_data), self.frequency[fpoint]
        elif self.model == HFSS:
            comH = [(data[3]*e**(data[4]*1j),data[5]*e**(data[6]*1j)) for data in self.data1]
            comE = [(data[3]*e**(data[4]*1j),data[5]*e**(data[6]*1j)) for data in self.data2]

            return comH,comE, self.data1[0][2]






if __name__ == "__main__":
    dataloader = DataLoader(model=HFSS)
    H,E=dataloader.get_data()
    print(H[0])


