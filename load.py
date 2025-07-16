import numpy as np
from scipy.io import loadmat
import warnings

path1 = r"scanarray_pol1_h6mm-10_2_2009.mat"
path2 = r"scanarray_pol2_h6mm-10_2_2009.mat"


class DataLoader:
    def __init__(self, path1_=path1, path2_=path2):
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



    def get_data(self, freq=-1, index=-1):
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






if __name__ == "__main__":
    dataloader = DataLoader()
    a,b,c = dataloader.get_data(index=100)

