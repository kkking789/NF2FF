import load
import numpy as np
from rebuild import GBuilder,PCG
from scipy.sparse.linalg import bicgstab


if __name__ == "__main__":
    dataloader = load.DataLoader()
    pol1_data,pol2_data,freq = dataloader.get_data()

    Gbuilder = GBuilder(dataloader.xstep, dataloader.ystep, dataloader.xpoints, dataloader.ypoints,6)
    G = Gbuilder.build(freq)

    pol1_M = np.linalg.solve(-G, pol1_data.flatten())
    pol2_M = np.linalg.solve(G, pol1_data.flatten())

    FGbuilder = GBuilder(dataloader.xstep, dataloader.ystep, dataloader.xpoints, dataloader.ypoints,1e10)
    G = Gbuilder.build(freq)

