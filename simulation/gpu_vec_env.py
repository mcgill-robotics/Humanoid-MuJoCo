import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from simulation.reward_functions import *
from simulation.simulation_parameters import *
import gc
import random
from simulation.gpu_vec_env_utils import *
from simulation import SIM_XML_PATH, reward_functions


class GPUVecEnv(VecEnv):
    def __init__(
        self,
        num_envs,
        reward_fn,
        xml_path=SIM_XML_PATH,
        randomization_factor=0,
        use_potential_rewards=USE_POTENTIAL_REWARDS,
        max_simulation_time_override=None,
    ):
        if jax.default_backend() != "gpu":
            print("ERROR: Failed to find GPU device.")
            exit()
        print("INFO: Running MuJoCo on GPU.")

        self.platform = "GPU"
        self.xml_path = xml_path
        self.randomization_factor = randomization_factor
        self.timestep = TIMESTEP
        self.num_envs = num_envs
        self.use_potential_rewards = bool(use_potential_rewards)
        if type(reward_fn) == str:
            reward_fn = getattr(reward_functions, reward_fn)
        self.physics_steps_per_control_step = PHYSICS_STEPS_PER_CONTROL_STEP
        self.rng_key = jax.random.PRNGKey(42)
        self.rng = jax.random.split(self.rng_key, self.num_envs)
        self.render_mode = [[None]] * self.num_envs
        self.max_simulation_time = (
            max_simulation_time_override
            if max_simulation_time_override is not None
            else MAX_SIM_TIME
        )
        if self.max_simulation_time < 0:
            self.max_simulation_time = np.inf
        self.gravity_vector_batch = jp.array([jp.array([0, 0, -1])] * self.num_envs)

        # DEFINE JAX JITTED FUNCTIONS
        def rollout(m, d):
            for _ in range(self.physics_steps_per_control_step):
                d = mjx.step(m, d)
            return d

        self.jax_step = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
        self.getContactSensorData = jax.jit(
            jax.vmap(getContactSensorData, in_axes=(None, 0))
        )
        self.reward_fn = jax.jit(
            jax.vmap(
                lambda velocity, target_velocity, torso_quat, target_yaw, z_pos, joint_torques, ctrl_change, isSelfColliding: reward_fn(
                    velocity,
                    target_velocity,
                    torso_quat,
                    target_yaw,
                    z_pos,
                    joint_torques,
                    ctrl_change,
                    isSelfColliding,
                )
            )
        )
        self.isSelfColliding = jax.jit(jax.vmap(checkSelfCollision, in_axes=(None, 0)))

        # DEFINE ACTION AND OBSERVATION SPACES
        self.action_space = spaces.Box(
            -1, 1, shape=(len(JOINT_NAMES),), dtype=np.float32
        )
        observation_size = len(JOINT_NAMES) + len(JOINT_NAMES) + 3 + 3 + 3 + 2 + 3 + 3
        self.observation_space = spaces.Box(
            -10, 10, shape=(observation_size,), dtype=np.float32
        )

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    ###### METHODS REQUIRED BY STABLE BASELINES VECENV ######
    def close(self):
        try:
            del self.model
        except:
            pass
        try:
            del self.cpu_model
        except:
            pass
        try:
            del self.base_mjx_data
        except:
            pass
        try:
            del self.data_batch
        except:
            pass
        # clean up any unreferenced variables
        gc.collect()

    def render(self, mode=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def step_async(self, actions):
        self.async_result = self.step(actions)

    def step_wait(self):
        return self.async_result

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            return getattr(self, attr_name)
        else:
            return [getattr(self, attr_name)] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "reset":
            print("Attempted reset on envs [{}]".format(indices))
        elif method_name == "step":
            print("Attempted step on envs [{}]".format(indices))
        else:
            func = getattr(self, method_name)
            if indices is not None:
                return [func(*method_args)]
            else:
                return func(*method_args)

    def seed(self, seed):
        self.rng_key = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(self.rng_key, self.num_envs)

    #########################################################

    def _init_model(self):
        # load model from XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.timestep
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 15
        self.model.opt.ls_iterations = 15
        # save joint addresses
        self.joint_qpos_idx = []
        self.joint_dof_idx = []
        for joint in JOINT_NAMES:
            self.joint_dof_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
            self.joint_qpos_idx.append(
                self.model.jnt_qposadr[self.model.joint(joint).id]
            )
        self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
        self.joint_dof_idx = jp.array(self.joint_dof_idx)
        # save gravity vector
        # save torso body index
        self.torso_idx = self.model.body(TORSO_BODY_NAME).id
        # get pressure sensor geom ids
        self.pressure_sensor_ids = [
            self.model.geom(pressure_sensor_geom).id
            for pressure_sensor_geom in PRESSURE_GEOM_NAMES
        ]
        self.non_robot_geom_ids = [self.model.geom(geom).id for geom in NON_ROBOT_GEOMS]

    def _randomize_control_inputs(self):
        # initialize control inputs for each env
        if USE_CONTROL_INPUTS:
            self.control_inputs_velocity = jax.random.uniform(
                self.rng_key,
                (self.num_envs, 2),
                minval=-1 * CONTROL_INPUT_MAX_VELOCITY,
                maxval=CONTROL_INPUT_MAX_VELOCITY,
            )
            self.control_inputs_yaw = jax.random.uniform(
                self.rng_key,
                (self.num_envs, 1),
                minval=-1 * CONTROL_INPUT_MAX_YAW,
                maxval=CONTROL_INPUT_MAX_YAW,
            )
            if RANDOMIZATION_FACTOR_AFFECTS_CONTROL_INPUT:
                self.control_inputs_yaw = (
                    self.control_inputs_yaw * self.randomization_factor
                )
                self.control_inputs_velocity = (
                    self.control_inputs_velocity * self.randomization_factor
                )
        else:
            self.control_inputs_velocity = jp.zeros((self.num_envs, 2))
            self.control_inputs_yaw = jp.zeros((self.num_envs, 1))

    def _init_sim_trackers(self):
        # initialize instance parameters
        self.next_force_start_times = jp.zeros((self.num_envs))
        self.next_force_durations = jp.zeros((self.num_envs))
        self.next_force_magnitudes = jp.zeros((self.num_envs))
        self.next_force_bodies = jp.zeros((self.num_envs))
        self.next_force_directions = jp.zeros((self.num_envs, 2))
        self.last_actions = self.data_batch.ctrl
        self.action_change = jp.zeros(self.data_batch.ctrl.shape)
        self.previous_rewards, _ = self._get_rewards(override_potential=True)

    def _randomize_dynamics(self):
        # floor friction (0.5 to 1.0)
        self.model.geom("floor").friction = [
            coef
            * (
                1.0 * (1.0 - self.randomization_factor)
                + random.uniform(
                    FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER
                )
                * self.randomization_factor
            )
            for coef in self.model.geom("floor").friction
        ]
        # vary the mass of all limbs randomly
        for i in range(self.model.nbody - 1):
            self.model.body(i + 1).mass[0] = max(
                0.00001,
                self.model.body(i + 1).mass[0]
                + random.uniform(
                    -MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                    MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                ),
            )
        # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
        self.model.body(random.randint(1, self.model.nbody - 1)).mass[
            0
        ] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED * self.randomization_factor)

    def _randomize_joint_properties(self):
        # randomize joint properties
        for joint in JOINT_NAMES:
            self.model.joint(joint).armature[0] += (
                random.uniform(0, JOINT_ARMATURE_MAX_CHANGE) * self.randomization_factor
            )
            self.model.joint(joint).stiffness[0] += (
                random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)
                * self.randomization_factor
            )
            self.model.joint(joint).margin[0] += (
                random.uniform(0, JOINT_MARGIN_MAX_CHANGE) * self.randomization_factor
            )
            self.model.joint(joint).range[0] += (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
            self.model.joint(joint).range[1] += (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
        for joint in JOINT_ACTUATOR_NAMES:
            self.model.actuator(joint).forcerange[0] += (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )
            self.model.actuator(joint).forcerange[1] += (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )

    def _init_GPU_model(self):
        # create MJX model/data from CPU model/data
        self.cpu_model = self.model
        self.model = mjx.put_model(self.cpu_model)
        mj_data = mujoco.MjData(self.cpu_model)
        mujoco.mj_kinematics(self.cpu_model, mj_data)
        self.base_mjx_data = mjx.put_data(self.cpu_model, mj_data)
        self.data_batch = jax.vmap(lambda rng: self.base_mjx_data)(self.rng)

    def _randomize_joint_positions(self):
        # randomize joint initial states (GPU)
        joint_pos_range = JOINT_INITIAL_OFFSET_MIN + self.randomization_factor * (
            JOINT_INITIAL_OFFSET_MAX - JOINT_INITIAL_OFFSET_MIN
        )
        joint_mask = jp.zeros(self.base_mjx_data.qpos.shape, dtype=jp.float32)
        joint_mask = joint_mask.at[self.joint_qpos_idx].set(1.0)
        joint_pos_range = joint_pos_range * joint_mask

        self.data_batch = jax.vmap(
            lambda rng: self.base_mjx_data.replace(
                qpos=self.base_mjx_data.qpos
                + jax.random.uniform(
                    rng,
                    self.base_mjx_data.qpos.shape,
                    minval=-joint_pos_range,
                    maxval=joint_pos_range,
                )
            )
        )(self.rng)

    def _randomize_delays(self):

        # delays in actions and observations (10ms to 50ms)
        # round delays to be multiples of the timestep
        self.actual_timestep = self.timestep * self.physics_steps_per_control_step
        self.action_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.action_delay = round(self.action_delay / self.actual_timestep)
        self.joint_observation_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.joint_observation_delay = round(
            self.joint_observation_delay / self.actual_timestep
        )
        self.local_ang_vel_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.local_ang_vel_delay = round(
            self.local_ang_vel_delay / self.actual_timestep
        )
        self.torso_local_velocity_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.torso_local_velocity_delay = round(
            self.torso_local_velocity_delay / self.actual_timestep
        )
        self.local_gravity_vector_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.local_gravity_vector_delay = round(
            self.local_gravity_vector_delay / self.actual_timestep
        )
        self.contact_sensor_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.contact_sensor_delay = round(
            self.contact_sensor_delay / self.actual_timestep
        )

        # make buffers for observations and actions
        self.action_buffer = [self.data_batch.ctrl] * (int)(self.action_delay)
        self.joint_angles_buffer = [
            jp.array([[0] * len(JOINT_NAMES)] * self.num_envs)
        ] * self.joint_observation_delay
        self.joint_velocities_buffer = [
            jp.array([[0] * len(JOINT_NAMES)] * self.num_envs)
        ] * self.joint_observation_delay
        self.local_ang_vel_buffer = [
            jp.array([[0] * 3] * self.num_envs)
        ] * self.local_ang_vel_delay
        self.torso_local_velocity_buffer = [
            jp.array([[0] * 3] * self.num_envs)
        ] * self.torso_local_velocity_delay
        self.local_gravity_vector_buffer = [
            jp.array([[0, 0, -1]] * self.num_envs)
        ] * self.local_gravity_vector_delay
        self.contact_sensor_buffer = [
            jp.array([[0, 0]] * self.num_envs)
        ] * self.contact_sensor_delay

    def reset(self, seed=None, options=None):
        self.close()

        # APPLY SEED
        if seed is not None:
            self.seed(seed)

        self._randomize_control_inputs()

        self._init_model()
        self._randomize_dynamics()
        self._randomize_joint_properties()

        self._init_GPU_model()

        self._randomize_joint_positions()
        self._randomize_delays()

        self._init_sim_trackers()

        return self._get_obs()

    def _get_torso_velocity(self):  # NOTE: in global reference frame (NWU)
        return self.data_batch.cvel[:, self.torso_idx][:, 3:]

    def _get_torso_angular_velocity(self):  # NOTE: in global reference frame (NWU)
        return self.data_batch.cvel[:, self.torso_idx][:, 0:3]

    def _get_joint_torques(self):
        return (
            self.data_batch.qfrc_constraint[:, self.joint_dof_idx]
            + self.data_batch.qfrc_smooth[:, self.joint_dof_idx]
        )

    def _get_joint_positions(self):
        return self.data_batch.qpos[:, self.joint_qpos_idx]

    def _get_joint_velocities(self):
        return self.data_batch.qvel[:, self.joint_dof_idx]

    def _get_torso_quaternion(self):  # in NWU
        return self.data_batch.xquat[:, self.torso_idx]

    def _get_torso_z_pos(self):
        return self.data_batch.xpos[:, self.torso_idx, 2]

    def _get_rewards(self, override_potential=False):

        torso_global_velocity = self._get_torso_velocity()
        torso_z_pos = self._get_torso_z_pos()
        torso_quat = self._get_torso_quaternion()
        joint_torques = self._get_joint_torques()
        is_self_colliding = self.isSelfColliding(
            self.non_robot_geom_ids, self.data_batch
        )

        rewards, areTerminal = self.reward_fn(
            torso_global_velocity,
            self.control_inputs_velocity,
            torso_quat,
            self.control_inputs_yaw,
            torso_z_pos,
            joint_torques,
            self.action_change,
            is_self_colliding,
        )

        if self.use_potential_rewards and not override_potential:
            _rewards = rewards - self.previous_rewards
            self.previous_rewards = rewards
            rewards = _rewards

        return np.array(rewards), np.array(areTerminal)

    def _get_obs(self):

        # joint positions
        joint_pos_noise = (
            self.randomization_factor
            * (JOINT_ANGLE_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(
                key=self.rng_key, shape=(self.num_envs, len(self.joint_qpos_idx))
            )
        )
        joint_angles = self._get_joint_positions() + joint_pos_noise

        # joint velocities
        joint_vel_noise = (
            self.randomization_factor
            * (JOINT_VELOCITY_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(
                key=self.rng_key, shape=(self.num_envs, len(self.joint_qpos_idx))
            )
        )
        joint_velocities = self._get_joint_velocities() + joint_vel_noise

        # local angular velocity
        ang_vel_noise = (
            self.randomization_factor
            * (GYRO_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3))
        )
        torso_quat = self._get_torso_quaternion()
        local_ang_vel = (
            inverseRotateVectors(torso_quat, self._get_torso_angular_velocity())
            + ang_vel_noise
        )

        # local velocity
        local_vel_noise = (
            self.randomization_factor
            * VELOCIMETER_NOISE_STDDEV
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3))
        )
        torso_local_velocity = (
            inverseRotateVectors(torso_quat, self._get_torso_velocity())
            + local_vel_noise
        )

        # gravity direction (local)
        quaternion_noise = (
            self.randomization_factor
            * (IMU_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 4))
        )
        noisy_torso_quat = torso_quat + quaternion_noise
        local_gravity_vector = inverseRotateVectors(
            noisy_torso_quat, self.gravity_vector_batch
        )

        # foot contact states
        contact_states = self.getContactSensorData(
            self.pressure_sensor_ids, self.data_batch
        )

        # cycle observations through observation buffers
        self.joint_angles_buffer.append(joint_angles)
        self.joint_velocities_buffer.append(joint_velocities)
        self.local_ang_vel_buffer.append(local_ang_vel)
        self.torso_local_velocity_buffer.append(torso_local_velocity)
        self.local_gravity_vector_buffer.append(local_gravity_vector)
        self.contact_sensor_buffer.append(contact_states)

        joint_angles = self.joint_angles_buffer.pop(0)
        joint_velocities = self.joint_velocities_buffer.pop(0)
        local_ang_vel = self.local_ang_vel_buffer.pop(0)
        torso_local_velocity = self.torso_local_velocity_buffer.pop(0)
        local_gravity_vector = self.local_gravity_vector_buffer.pop(0)
        contact_states = self.contact_sensor_buffer.pop(0)

        # calculate clock phase observations (no delay on these)
        clock_phase_sin = jp.sin(self.data_batch.time).reshape(-1, 1)
        clock_phase_cos = jp.cos(self.data_batch.time).reshape(-1, 1)
        clock_phase_complex = (
            (clock_phase_sin)
            / (2 * jp.sqrt((clock_phase_sin * clock_phase_sin) + 0.04))
        ) + 0.5

        delayed_observations = jp.hstack(
            (
                joint_angles,  # rad
                joint_velocities,  # rad / s
                local_ang_vel,  # rad / s
                torso_local_velocity,  # m/s
                local_gravity_vector,  # unit vector
                contact_states[:, 0].reshape(
                    -1, 1
                ),  # for left foot (is it touching the ground?)
                contact_states[:, 1].reshape(-1, 1),  # for right foot
                self.control_inputs_velocity,  # as defined in reset
                self.control_inputs_yaw,  # as defined in reset
                clock_phase_sin,  # as defined in paper on potential rewards
                clock_phase_cos,  # as defined in paper on potential rewards
                clock_phase_complex,  # as defined in paper on potential rewards
            )
        )

        return np.array(delayed_observations)

    def _apply_action(self, actions):
        # cycle action through action buffer
        self.action_buffer.append(actions)
        action_to_take = self.action_buffer.pop(0)
        action_to_take = jp.clip(jp.array(action_to_take), -1.0, 1.0)
        action_to_take = action_to_take * (jp.pi / 2)
        self.data_batch = self.data_batch.replace(ctrl=action_to_take)

        self.action_change = action_to_take - self.last_actions
        self.last_actions = action_to_take

    def step(self, actions=None):

        # apply inputted actions
        if actions is not None:
            self._apply_action(actions)

        # apply forces to the robot to destabilise it
        xfrc_applied = applyExternalForces(self)
        self.data_batch = self.data_batch.replace(xfrc_applied=xfrc_applied)

        # step sims, update data batch
        self.data_batch = self.jax_step(self.model, self.data_batch)

        obs = self._get_obs()
        rewards, terminals = self._get_rewards()

        truncated = np.any(self.data_batch.time >= self.max_simulation_time)
        fraction_of_terminated = np.sum(terminals) / np.sum(np.ones(terminals.shape))
        done = truncated or fraction_of_terminated > TERMINAL_FRACTION_RESET_THRESHOLD
        dones = np.full(terminals.shape, done)
        infos = [{}] * self.num_envs

        # automatically reset environment if any instance terminates
        if done:
            for env_idx in range(self.num_envs):
                infos[env_idx]["terminal_observation"] = obs[env_idx]
                infos[env_idx]["TimeLimit.truncated"] = not terminals[env_idx]
            obs = self.reset()

        return obs, rewards, dones, infos


if __name__ == "__main__":
    sim_batch = GPUVecEnv(
        num_envs=1,
        xml_path=SIM_XML_PATH,
        reward_fn=controlInputRewardFn,
        randomization_factor=1,
    )

    obs = sim_batch.reset()
    rewards = []
    terminals = []

    while True:
        actions = np.random.uniform(-1, 1, (sim_batch.num_envs, len(JOINT_NAMES)))
        actions = np.zeros((sim_batch.num_envs, len(JOINT_NAMES)))
        actions = None

        obs, rewards, terminals, _ = sim_batch.step(actions)
        # print(rewards[0])
        if np.isnan(obs).any() or np.isnan(rewards).any() or np.isnan(terminals).any():
            print("ERROR: NaN value in observations/rewards/terminals.")
