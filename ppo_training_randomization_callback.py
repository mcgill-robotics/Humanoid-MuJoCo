from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np

class IncreaseRandomizationOnNoModelImprovement(BaseCallback):
    parent: EvalCallback

    def __init__(self, max_no_improvement_evals, envs, randomization_increment, min_evals):
        super().__init__(verbose=0)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.envs = envs
        self.randomization_increment = randomization_increment

    def _on_step(self):
        assert self.parent is not None, "``IncreaseRandomizationOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        increment_randomization = False

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    increment_randomization = True

        self.last_best_mean_reward = self.parent.best_mean_reward

        current_randomization_factor = self.envs[0].get_attr("randomization_factor")
        
        if increment_randomization and current_randomization_factor < 1:
            print(
                f"Increasing randomization factor to {current_randomization_factor + self.randomization_increment}."
            )
            for env in self.envs: env.set_attr("randomization_factor", current_randomization_factor + self.randomization_increment)

        return True