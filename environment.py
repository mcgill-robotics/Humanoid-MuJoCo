
# class for running parallel instances of the simulation at once

class Environment:
    def __init__(self, sim_class, run_on_gpu, thread_count, step_size):
        # , xml_path, timestep=0.001, randomization_factor=0, run_on_gpu=True
        if run_on_gpu:
            self.step = self._step_gpu
            self.reset = self._reset_gpu
            self.getObs = self._getObs_gpu
            self.getReward = self._getReward_gpu
        else:
            self.step = self._step_cpu
            self.reset = self._reset_cpu
            self.getObs = self._getObs_cpu
            self.getReward = self._getReward_cpu
        
    def step(self): pass
        
    def _step_gpu(self):
        
    def _step_cpu(self):
    
    def reset(self): pass
        
    def _reset_gpu(self):
        
    def _reset_cpu(self):
        
    def getObs(self): pass
        
    def _getObs_gpu(self):
        
    def _getObs_cpu(self):
        
    def getReward(self): pass
        
    def _getReward_gpu(self):
        
    def _getReward_cpu(self):
        
    