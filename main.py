import load
from load import HFSS,MATLAB
import numpy as np
from rebuild import GBuilder,PCG, SourceRebuilder
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataloader = load.DataLoader(model=HFSS)
    H,E,h = dataloader.get_data()
    rebuilder = SourceRebuilder(E,H,h,2.5e9, dataloader.xstep,dataloader.ystep,dataloader.xpoints,dataloader.ypoints)
    rebuilder.solve(E,H)




    # Gbuilder = GBuilder(dataloader.xstep, dataloader.ystep, dataloader.xpoints, dataloader.ypoints,6)
    # G = Gbuilder.build(freq)
    #
    # pol1_M = np.linalg.solve(-G, pol1_data.flatten())
    # pol2_M = np.linalg.solve(G, pol1_data.flatten())
    #
    # pol1_M = abs(pol1_M.imag)
    # sns.heatmap(pol1_M.reshape((51,51)), cmap="YlOrRd")
    # plt.show()
    #
    # # FGbuilder = GBuilder(dataloader.xstep, dataloader.ystep, dataloader.xpoints, dataloader.ypoints, 1e6)
    # # FG = FGbuilder.build(freq)
    # #


