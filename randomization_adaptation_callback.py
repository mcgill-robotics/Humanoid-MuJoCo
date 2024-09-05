from stable_baselines3.common.callbacks import BaseCallback


class RewardAdaptationCallback(BaseCallback):
    def __init__(
        self,
        envs,
        eval_cb,
        success_reward_threshold,
        max_evals_at_max_reward,
        initial_randomization_factor,
        randomization_increment,
        log_dir,
    ):
        super().__init__(0)
        self.envs = envs
        self.eval_cb = eval_cb
        self.eval_freq = eval_cb.eval_freq
        self.success_reward_threshold = success_reward_threshold
        self.max_evals_at_max_reward = max_evals_at_max_reward
        self.initial_randomization_factor = initial_randomization_factor
        self.current_randomization_factor = initial_randomization_factor
        self.current_evals_at_max_reward = 0
        self.randomization_increment = randomization_increment
        self.log_dir = log_dir
        with open(self.log_dir + "/randomization_factors.csv", "w+") as f:
            f.write("")

    def _log_randomization_factor(self):
        with open(self.log_dir + "/randomization_factors.csv", "a+") as f:
            f.write(str(self.current_randomization_factor) + "\n")

    def _update_randomization_factor(self, new_randomization_factor):
        if new_randomization_factor > 1:
            new_randomization_factor = 1
        elif new_randomization_factor < 0:
            new_randomization_factor = 0
        if self.current_randomization_factor == new_randomization_factor:
            return
        print(
            "{} randomization factor to {}".format(
                (
                    "Increasing"
                    if new_randomization_factor > self.current_randomization_factor
                    else "Decreasing"
                ),
                new_randomization_factor,
            )
        )
        self.current_randomization_factor = new_randomization_factor
        self.current_evals_at_max_reward = 0
        for env in self.envs:
            env.set_attr("randomization_factor", self.current_randomization_factor)
            env.reset()

    def _on_step(self):
        continue_training = True
        if self.n_calls % self.eval_freq == 0:
            if float(self.eval_cb.last_mean_reward) > self.success_reward_threshold:
                if self.current_randomization_factor == 1:
                    self.current_evals_at_max_reward += 1
                self._update_randomization_factor(
                    self.current_randomization_factor + self.randomization_increment
                )
            else:
                self._update_randomization_factor(
                    self.current_randomization_factor - self.randomization_increment
                )

            if self.current_evals_at_max_reward >= self.max_evals_at_max_reward:
                continue_training = False

            self._log_randomization_factor()

        return continue_training
