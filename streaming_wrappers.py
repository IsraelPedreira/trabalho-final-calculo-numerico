from benchmark import StreamingInterpolator
from newton import NewtonIncremental, newton_parallel
from lagrange import lagrange, lagrange_parallel
import numpy as np

class StreamNewtonCPU(StreamingInterpolator):
    def __init__(self):
        self.engine = NewtonIncremental()
    
    def add_share(self, share):
        self.engine.adicionar_share(share)
        
    def get_secret(self):
        return self.engine.obter_segredo()
    
    def reset(self):
        self.engine.reset()

class StreamLagrangeCPU(StreamingInterpolator):
    def __init__(self):
        self.history = []
        
    def add_share(self, share):
        self.history.append(share)
        
    def get_secret(self):
        if len(self.history) < 2:
            return 0.0
        return lagrange(self.history)
    
    def reset(self):
        self.history = []

class StreamLagrangeGPU(StreamingInterpolator):
    def __init__(self):
        self.history = []
        
    def add_share(self, share):
        self.history.append(share)
        
    def get_secret(self):
        if len(self.history) < 2:
            return 0.0
        return lagrange_parallel(self.history)
    
    def reset(self):
        self.history = []

class StreamNewtonGPU(StreamingInterpolator):
    def __init__(self):
        self.history_np = np.empty((0, 2))
        
    def add_share(self, share):
        new_row = np.array([[share[0], share[1]]])
        self.history_np = np.vstack([self.history_np, new_row])
        
    def get_secret(self):
        return newton_parallel(self.history_np)
    
    def reset(self):
        self.history_np = np.empty((0, 2))