from pysptk import mcep
import numpy as np 

class MelCepstralProcessing(object):
    ''' 
    Specify input dimension (1 or 2) and input type (wav, mag spec) 
    *Requires 'PYSPTK'
    '''
    def __init__(self, itype=3, order=24, isMatrix=True, drop_zeroth=False):
        self.itype = itype
        self.order = order
        self.isMatrix = isMatrix
        self.drop_zeroth = drop_zeroth
        self.a = 10 / np.log(10) * np.sqrt(2)
    def x2mcep(self, x):
        x = x.astype(np.float64)
        # [TODO] to avoid ValueError: ndarray is not C-contiguous
        if not x.flags['C_CONTIGUOUS']:
            x = x.copy(order='C')
        # if self.itype == 3:
        #     etype = 2
        #     eps = 1e-10
        # else:
        #     etype = 0
        #     eps = 0.0
        etype = 0
        eps = 0.0
        if self.isMatrix:
            return np.asarray(
                [mcep(xi, order=self.order, itype=self.itype, etype=etype, eps=eps) for xi in x])
        else:
            return mcep(x, order=self.order, itype=self.itype, etype=etype, eps=eps)
    def mmcd(self, x, y):
        ''' 
        Mean Mel-Cepstral Distortion 
        Usually the very is around 7-8 (dB)
        '''
        x = self.x2mcep(x)
        y = self.x2mcep(y)
        if self.drop_zeroth:
            x = x[:, 1:]
            y = y[:, 1:]
        z = np.square(x - y)
        z = np.sum(z, 1)
        z = np.sqrt(z)
        z = np.mean(z)
        return self.a * z
