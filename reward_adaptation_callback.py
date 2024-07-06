from stable_baselines3.common.callbacks import BaseCallback


class RewardAdaptationCallback(BaseCallback):
    def __init__(
        self,
        envs,
        eval_freq,
        eval_cb,
        success_threshold,
        max_evals_at_max_reward,
        initial_randomization_factor,
        randomization_increment,
        verbose,
        log_dir,
    ):
        super().__init__(verbose)
        self.envs = envs
        self.eval_freq = eval_freq
        self.eval_cb = eval_cb
        self.success_threshold = success_threshold
        self.max_evals_at_max_reward = max_evals_at_max_reward
        self.initial_randomization_factor = initial_randomization_factor
        self.current_randomization_factor = initial_randomization_factor
        self.current_evals_at_max_reward = 0
        self.randomization_increment = randomization_increment
        self.log_dir = log_dir
        self._log_randomization_factor()

    def _log_randomization_factor(self):
        with open(self.log_dir + "/randomization_factors.csv", "a") as f:
            f.write(str(self.current_randomization_factor) + "\n")

    def _update_randomization_factor(self, new_randomization_factor):
        if new_randomization_factor > 1:
            new_randomization_factor = 1
        elif new_randomization_factor < 0:
            new_randomization_factor = 0
        if self.current_randomization_factor == new_randomization_factor:
            return
        print(
            "{} randomization factor to",
            (
                "Increasing"
                if new_randomization_factor > self.current_randomization_factor
                else "Decreasing"
            ),
            new_randomization_factor,
        )
        self.current_randomization_factor = new_randomization_factor
        self.current_evals_at_max_reward = 0
        for env in self.envs:
            env.set_attr("randomization_factor", self.current_randomization_factor)
            env.reset()

    def _on_step(self):
        continue_training = True
        if self.n_calls % self.eval_freq == 0:
            if (
                self.eval_cb.logger.name_to_value["eval/success_rate"]
                > self.success_threshold
            ):
                if self.current_randomization_factor == 1:
                    self.current_evals_at_max_reward += 1
                self._update_randomization_factor(
                    self.current_randomization_factor + self.randomization_increment
                )
            else:
                self._update_randomization_factor(
                    self.current_randomization_factor - self.randomization_increment
                )

            self._log_randomization_factor()
            if self.current_evals_at_max_reward >= self.max_evals_at_max_reward:
                continue_training = False

        return continue_training
