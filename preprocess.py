import pdb
import numpy as np
from iohandler.spectral_reader import find_files

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

directory = '/home/jrm/proj/vc2016b/TR_log_SP_Z_LT8000'
pattern = '.+(150|[01][0-4]\d|0\d\d)\.bin'
mFea = 513 + 1

files = find_files(directory, pattern=pattern)

print(len(files))
xmax = -9999 * np.ones([1, mFea - 1])
xmin = 99999 * np.ones([1, mFea - 1])
# x_all = np.ones([0, 513])
x_all = list()
for f in files:
	x = np.fromfile(f, dtype=np.float32)
	x = np.reshape(x, [-1, mFea])
	x = x[:, 1:]
	xmax = np.concatenate([x, xmax], axis=0).max(0).reshape([1, mFea -1])
	xmin = np.concatenate([x, xmin], axis=0).min(0).reshape([1, mFea -1])

	x_all.append(x)

x_all = np.concatenate(x_all, axis=0)

xmu = x_all.mean(0).reshape([1, mFea -1])
xsd = x_all.std(0).reshape([1, mFea -1])

(xmax - xmu) / xsd
(xmu- xmin) / xsd

print(xmax)
# print()
print(xmin)

np.save('xmax.npf', xmax)
np.save('xmin.npf', xmin)



for i in range(513):
	x_ = x_all[:, i]
	bins = np.linspace(xmin[0, i], xmax[0, i], 256)

	plt.figure()
	plt.hist(x_, bins)
	plt.savefig('img/{:03d}'.format(i))
	plt.close()

