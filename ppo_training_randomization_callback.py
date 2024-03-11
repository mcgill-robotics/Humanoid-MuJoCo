from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np

class IncreaseRandomizationOnNoModelImprovement(BaseCallback):
    parent: EvalCallback

    def __init__(self, envs, randomization_increment):
        super().__init__(verbose=0)
        self.envs = envs
        self.randomization_increment = randomization_increment

    def _on_step(self):
        assert self.parent is not None, "``IncreaseRandomizationOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        if self.parent.last_mean_reward > 900: increment_randomization = True
        else: increment_randomization = False

        current_randomization_factor = self.envs[0].get_attr("randomization_factor")
        
        if increment_randomization and current_randomization_factor < 1:
            print(
                f"Increasing randomization factor to {current_randomization_factor + self.randomization_increment}."
            )
            for env in self.envs: env.set_attr("randomization_factor", current_randomization_factor + self.randomization_increment)

        return True