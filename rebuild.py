import os.path
import sys
import math
from math import e
import numpy as np
import pyswarms as ps

class ProgressBar:
	def __init__(self, total, length=50, title='building'):
		self.total = total
		self.length = length
		self.title = title
		self.progress = 0

	def update(self):
		progress = self.progress
		percent = "{0:.1f}".format(100 * (progress / float(self.total)))
		filled_length = int(self.length * progress // self.total)+1

		GREEN = '\033[92m'
		RED = '\033[91m'
		RESET = '\033[0m'

		filled = GREEN + '-' * filled_length + RESET
		unfilled = RED + '-' * (self.length - filled_length) + RESET
		bar = filled + unfilled

		sys.stdout.write(f'\r{self.title} {bar} {percent}% (￣o￣) . z Z')
		if percent == "100.0":
			sys.stdout.write(f'\r{self.title} {bar} {percent}% ╰(*°▽°*)╯')
		sys.stdout.flush()
		self.progress+=1

class Grid:
	def __init__(self,xstep,ystep,xpoints,ypoints):
		self.xstep = xstep
		self.ystep = ystep
		self.xpoints = xpoints
		self.ypoints = ypoints
		self.total = xpoints*ypoints

class Rebuilder:
	def __init__(self,E,H,h,xstep,ystep,xpoints,ypoints, dxpoints, dypoints,f,hg=0.002):
		self.k = 2 * math.pi * f / 3e8
		self.K_h = self.k / (4 * math.pi)
		self.measure = Grid(xstep,ystep,xpoints,ypoints)
		self.rebuild = Grid(xstep*xpoints/dxpoints,ystep*ypoints/dypoints,dxpoints,dypoints)
		self.A = np.zeros((3 * self.measure.total, 2 * self.rebuild.total), dtype=complex)
		self.h = hg #重构点离地高度
		self.E = E
		self.H = H

		self.x = 0
		self.y = 0
		self.z = h
		self.r1 = 0
		self.r2 = 0
		self.G1r1 = 0
		self.G1r2 = 0
		self.G2r1 = 0
		self.G2r2 = 0

	def r(self):
		x = self.x
		y = self.y
		z = self.z
		h = self.h
		self.r1 = math.sqrt(x ** 2 + y ** 2 + (z - h) ** 2)
		self.r2 = math.sqrt(x ** 2 + y ** 2 + (z + h) ** 2)

	def G(self):
		k = self.k
		r1 = self.r1
		r2 = self.r2

		def f(r):
			return e**(-1j * k * r) / r

		def G1(r):
			return (3 / ((k * r) ** 2) + 1j * 3 / (k * r) - 1) * f(r)

		def G2(r):
			return (2 / ((k * r) ** 2 + 1j * 2 / (k * r))) * f(r)

		self.G1r1 = G1(r1)
		self.G1r2 = G1(r2)
		self.G2r1 = G2(r1)
		self.G2r2 = G2(r2)


	def A_build(self):
		measure = self.measure  # 测量点网格
		rebuild = self.rebuild  # 重建点网格
		K_h = self.K_h
		k = self.k
		A = np.zeros((3 * measure.total, 2 * rebuild.total), dtype=complex)

		if os.path.isfile("tmp\A.npy"):
			self.A = np.load("tmp\A.npy")
			if self.A.shape[1] == 2 * rebuild.total:
				return
		progressbar = ProgressBar(measure.total*rebuild.total)
		for M in range(measure.ypoints):
			for N in range(measure.xpoints):
				for m in range(rebuild.ypoints):
					for n in range(rebuild.xpoints):
						self.x = M * measure.xstep - n * rebuild.xstep
						self.y = N * measure.ystep - m * rebuild.ystep
						self.r()
						self.G()
						x = self.x
						y = self.y
						z = self.z
						h = self.h
						r1 = self.r1
						r2 = self.r2
						G1r1 = self.G1r1
						G2r2 = self.G2r2
						G1r2 = self.G1r2
						G2r1 = self.G2r1
						A_HxMx = K_h * k * (-1 * (y ** 2 + (z - h) ** 2) / (r1 ** 2) * G1r1 + G2r1
											- 1 * (y ** 2 + (z + h) ** 2) / (r2 ** 2) * G1r2 + G2r2)
						A_HxMy = K_h * k * ((x * y) / (r1 ** 2) * G1r1
											+ (x * y) / (r2 ** 2) * G1r2)
						A_HyMx = K_h * k * ((x * y) / (r1 ** 2) * G1r1
											+ (x * y) / (r2 ** 2) * G1r2)
						A_HyMy = K_h * k * (-1 * ((z - h) ** 2 + x ** 2) / (r1 ** 2) * G1r1 + G2r1
											- 1 * ((z + h) ** 2 + x ** 2) / (r2 ** 2) * G1r2 + G2r2)
						A_HzMx = K_h * k * ((z - h) * x / r1 ** 2 * G1r1 + (z + h) * x / r2 ** 2 * G1r2)
						A_HzMy = K_h * k * ((z - h) * y / r1 ** 2 * G1r1 + (z + h) * y / r2 ** 2 * G1r2)

						A[M * measure.xpoints + N][m * rebuild.xpoints + n] = A_HxMx
						A[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total] = A_HxMy
						A[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n] = A_HyMx
						A[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total] = A_HyMy
						A[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n] = A_HzMx
						A[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n + rebuild.total] = A_HzMy

						progressbar.update()

		self.A = A
		np.save("tmp\A.npy",A)


	def solve(self, swarm_size = 100, options = {'c1': 1.2, 'c2': 1.2, 'w': 0.9}):
		self.A_build()
		dim = 4*self.rebuild.total
		A = self.A
		H = self.H
		# def func(X):
		# 	F = np.zeros(swarm_size)
		# 	for i in range(swarm_size):
		# 		D = np.zeros(2*self.rebuild.total,dtype=complex)
		# 		for j in range(2*self.rebuild.total):
		# 			D[j]=X[i][j*2]+1j*X[i][j*2+1]
		# 		H_dipole = A @ D
		# 		f = np.mean(abs(H - H_dipole)/abs(H))
		# 		F[i] = f
		#
		# 	return F

		from sko.GA import GA
		def func(p):
			D = np.zeros(2 * self.rebuild.total, dtype=complex)

			for j in range(2*self.rebuild.total):
				D[j]=p[j*2]+1j*p[j*2+1]
			H_dipole = A @ D
			f = np.mean(abs(H - H_dipole)/abs(H))
			return  f

		ga = GA(func=func, n_dim=dim, size_pop=50, max_iter=800, prob_mut=0.001,precision=1e-7)
		x,y=ga.run()
		print(x)

		# optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
		# 									dimensions=dim,
		# 									options=options)
		# cost, dipole = optimizer.optimize(func, iters=2000)
		# return cost, dipole




if __name__ == "__main__":
	print(e)


