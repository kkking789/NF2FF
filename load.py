import numpy as np
from scipy.io import loadmat
import warnings
from math import e
import re



class DataLoader:
    def __init__(self, path):
        self.data = []
        with open(path, "r") as file1:
            lines = file1.readlines()
            for line in lines:
                if line[0] == "G":
                    data = [float(mid) for mid in re.findall(r'-?\d+\.?\d*', line)]
                    self.xpoints = int((data[3]-data[0])/data[6]+1)
                    self.xstep = float(data[6])/1000
                    self.ypoints = int((data[4]-data[1])/data[7]+1)
                    self.ystep = float(data[7])/1000
                    continue
                elif line[0] == "X":
                    continue
                self.data.append([float(mid) for mid in line.split()])

    def get_data(self):
        output = [(data[3]*e**(data[4]*1j),data[5]*e**(data[6]*1j),data[7]*e**(data[8]*1j)) for data in self.data]
        h = self.data[0][2]
        return output, h, self.xpoints, self.xstep, self.ypoints, self.ystep






if __name__ == "__main__":
    dataloader = DataLoader(model=HFSS)
    H,E=dataloader.get_data()
    print(H[0])


