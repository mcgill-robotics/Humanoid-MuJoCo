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
import time


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
        self.initialized_model_info = False
        self.control_inputs_velocity = jp.zeros((self.num_envs, 2))
        self.control_inputs_yaw = jp.zeros((self.num_envs, 1))

        self.jax_step = jax.jit(
            jax.vmap(
                lambda m, d: jax.lax.fori_loop(
                    jp.array(0),
                    jp.array(self.physics_steps_per_control_step),
                    lambda _, x: mjx.step(m, x),
                    d,
                )
            )
        )

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

        # save the important info of the mujoco model (indices for joints, torso, etc.)
        self._init_model_info()
        # fill model and data batches such that they can be properly indexed
        self._init_batches()
        # create and populate observation and action buffers
        self._init_buffers()
        # populate sim trackers
        self._init_sim_trackers()

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    #########################################################
    ###### METHODS REQUIRED BY STABLE BASELINES VECENV ######
    #########################################################

    def close(self):
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
        if method_name == "step":
            print("Attempted step on envs [{}]".format(indices))
        elif method_name == "reset":
            return self.reset(idx=indices, *method_args)
        else:
            func = getattr(self, method_name, **method_kwargs)
            if indices is not None:
                return [func(*method_args)]
            else:
                return func(*method_args)

    def seed(self, seed):
        self.rng_key = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(self.rng_key, self.num_envs)

    #########################################################
    ################ INITIALIZATION METHODS #################
    #########################################################

    def _make_model(self):
        # load model from XML
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        model.opt.timestep = self.timestep
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 15
        model.opt.ls_iterations = 15
        return model

    def _init_model_info(self):
        # load model from XML
        model = self._make_model()
        # save joint addresses
        self.joint_qpos_idx = []
        self.joint_dof_idx = []
        for joint in JOINT_NAMES:
            self.joint_dof_idx.append(model.jnt_dofadr[model.joint(joint).id])
            self.joint_qpos_idx.append(model.jnt_qposadr[model.joint(joint).id])
        self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
        self.joint_dof_idx = jp.array(self.joint_dof_idx)
        # save gravity vector
        # save torso body index
        self.torso_idx = model.body(TORSO_BODY_NAME).id
        # get pressure sensor geom ids
        self.pressure_sensor_ids = [
            model.geom(pressure_sensor_geom).id
            for pressure_sensor_geom in PRESSURE_GEOM_NAMES
        ]
        self.non_robot_geom_ids = [model.geom(geom).id for geom in NON_ROBOT_GEOMS]

    def _init_batches(self):
        cpu_model = self._make_model()
        gpu_model = mjx.put_model(cpu_model)
        mj_data = mujoco.MjData(cpu_model)
        mujoco.mj_kinematics(cpu_model, mj_data)
        mjx_data = mjx.put_data(cpu_model, mj_data)
        self.data_batch = jax.vmap(lambda _: mjx_data)(self.rng)
        self.model_batch = jax.vmap(lambda _: gpu_model)(self.rng)

    def _init_buffers(self):
        # delays in actions and observations (10ms to 50ms)
        # round delays to be multiples of the timestep
        self.actual_timestep = self.timestep * self.physics_steps_per_control_step
        max_timestep_delay = round(MAX_DELAY / self.actual_timestep)
        # make buffers for observations and actions
        self.action_buffer = jp.array([self.data_batch.ctrl] * (max_timestep_delay))
        self.action_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.joint_angles_buffer = jp.array(
            [jp.array([[0] * len(JOINT_NAMES)] * self.num_envs)] * max_timestep_delay
        )
        self.joint_angles_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.joint_velocities_buffer = jp.array(
            [jp.array([[0] * len(JOINT_NAMES)] * self.num_envs)] * max_timestep_delay
        )
        self.joint_velocities_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.local_ang_vel_buffer = jp.array(
            [jp.array([[0] * 3] * self.num_envs)] * max_timestep_delay
        )
        self.local_ang_vel_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.torso_local_velocity_buffer = jp.array(
            [jp.array([[0] * 3] * self.num_envs)] * max_timestep_delay
        )
        self.torso_local_velocity_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.local_gravity_vector_buffer = jp.array(
            [jp.array([[0, 0, 0]] * self.num_envs)] * max_timestep_delay
        )
        self.local_gravity_vector_delay_indices = [(0, i) for i in range(self.num_envs)]
        self.contact_sensor_buffer = jp.array(
            [jp.array([[0, 0]] * self.num_envs)] * max_timestep_delay
        )
        self.contact_sensor_delay_indices = [(0, i) for i in range(self.num_envs)]

    def _init_sim_trackers(self):
        # initialize instance parameters
        self.next_force_start_times = jp.zeros((self.num_envs))
        self.next_force_durations = jp.zeros((self.num_envs))
        self.next_force_magnitudes = jp.zeros((self.num_envs))
        self.next_force_bodies = jp.zeros((self.num_envs))
        self.next_force_directions = jp.zeros((self.num_envs, 2))
        self.last_actions = jp.zeros((self.num_envs, len(self.joint_dof_idx)))
        self.action_change = jp.zeros((self.num_envs, len(self.joint_dof_idx)))
        self.previous_rewards, _ = self._get_rewards(override_potential=True)

    #########################################################
    ############## RESET/RANDOMIZATION METHODS ##############
    #########################################################

    def _replace_data_index(self, idx, data_to_insert):
        self.data_batch = jax.vmap(lambda i, d: d if i == idx else data_to_insert)(
            jp.arange(self.num_envs), self.data_batch
        )

    def _replace_model_index(self, idx, model_to_insert):
        # '__class__', '__delattr__', '__dir__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'fields', 'replace', 'tree_replace
        for model in self.model_batch:
            print(model)
        # for field_name in [f.name for f in self.model_batch.fields()]:
        #     print(field_name)
        #     print(getattr(self.model_batch, field_name))
        #     update_field_value = getattr(self.model_batch, field_name).at[idx].set(getattr(model_to_insert, field_name))
        #     self.model_batch = self.model_batch.replace({field_name: getattr(model_to_insert, field_name)})
        
        # unstacked_model_batch = tree_unstack(self.model_batch)
        # unstacked_model_batch[idx] = model_to_insert
        
        # self.model_batch = jax.vmap(lambda i, m: model_to_insert if i == idx else m)(
        #     jp.arange(self.num_envs), self.model_batch
        # )
        # self.model_batch = tree_stack(unstacked_model_batch)

    def _get_random_control_inputs(self):
        # initialize control inputs for each env
        control_inputs_velocity = jax.random.uniform(
            self.rng_key,
            (self.num_envs, 2),
            minval=-1 * CONTROL_INPUT_MAX_VELOCITY,
            maxval=CONTROL_INPUT_MAX_VELOCITY,
        )
        control_inputs_yaw = jax.random.uniform(
            self.rng_key,
            (self.num_envs, 1),
            minval=-1 * CONTROL_INPUT_MAX_YAW,
            maxval=CONTROL_INPUT_MAX_YAW,
        )
        if RANDOMIZATION_FACTOR_AFFECTS_CONTROL_INPUT:
            control_inputs_yaw = self.control_inputs_yaw * self.randomization_factor
            control_inputs_velocity = (
                self.control_inputs_velocity * self.randomization_factor
            )
        return control_inputs_velocity, control_inputs_yaw

    def _update_sim_trackers(self, idx):
        # initialize instance parameters
        self.next_force_start_times = self.next_force_start_times.at[idx].set(
            jp.zeros((self.num_envs))[idx]
        )
        self.next_force_durations = self.next_force_durations.at[idx].set(
            jp.zeros((self.num_envs))[idx]
        )
        self.next_force_magnitudes = self.next_force_magnitudes.at[idx].set(
            jp.zeros((self.num_envs))[idx]
        )
        self.next_force_bodies = self.next_force_bodies.at[idx].set(
            jp.zeros((self.num_envs))[idx]
        )
        self.next_force_directions = self.next_force_directions.at[idx].set(
            jp.zeros((self.num_envs, 2))[idx]
        )
        self.last_actions = self.last_actions.at[idx].set(
            jp.zeros((self.num_envs, len(self.joint_dof_idx)))[idx]
        )
        self.action_change = self.action_change.at[idx].set(
            jp.zeros((self.num_envs, len(self.joint_dof_idx)))[idx]
        )
        _previous_rewards, _ = self._get_rewards(override_potential=True)
        self.previous_rewards[idx] = _previous_rewards[idx]

    def _randomize_dynamics(self, model):
        # floor friction (0.5 to 1.0)
        model.geom("floor").friction = [
            coef
            * (
                1.0 * (1.0 - self.randomization_factor)
                + random.uniform(
                    FLOOR_FRICTION_MIN_MULTIPLIER, FLOOR_FRICTION_MAX_MULTIPLIER
                )
                * self.randomization_factor
            )
            for coef in model.geom("floor").friction
        ]
        # vary the mass of all limbs randomly
        for i in range(model.nbody - 1):
            model.body(i + 1).mass[0] = max(
                0.00001,
                model.body(i + 1).mass[0]
                + random.uniform(
                    -MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                    MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                ),
            )
        # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
        model.body(random.randint(1, model.nbody - 1)).mass[0] += random.uniform(
            0, MAX_EXTERNAL_MASS_ADDED * self.randomization_factor
        )

        return model

    def _randomize_joint_properties(self, model):
        # randomize joint properties
        for joint in JOINT_NAMES:
            model.joint(joint).armature[0] += (
                random.uniform(0, JOINT_ARMATURE_MAX_CHANGE) * self.randomization_factor
            )
            model.joint(joint).stiffness[0] += (
                random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)
                * self.randomization_factor
            )
            model.joint(joint).margin[0] += (
                random.uniform(0, JOINT_MARGIN_MAX_CHANGE) * self.randomization_factor
            )
            model.joint(joint).range[0] += (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
            model.joint(joint).range[1] += (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
        for joint in JOINT_ACTUATOR_NAMES:
            model.actuator(joint).forcerange[0] += (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )
            model.actuator(joint).forcerange[1] += (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )
        return model

    def _randomize_joint_positions(self, data):
        # randomize joint initial states (GPU)
        joint_pos_range = JOINT_INITIAL_OFFSET_MIN + self.randomization_factor * (
            JOINT_INITIAL_OFFSET_MAX - JOINT_INITIAL_OFFSET_MIN
        )
        joint_mask = jp.zeros(data.qpos.shape, dtype=jp.float32)
        joint_mask = joint_mask.at[self.joint_qpos_idx].set(1.0)
        joint_pos_range = joint_pos_range * joint_mask

        return data.replace(
            qpos=data.qpos
            + jax.random.uniform(
                self.rng_key,
                data.qpos.shape,
                minval=-joint_pos_range,
                maxval=joint_pos_range,
            )
        )

    def _randomize_delays(self, idx):
        self.max_timestep_delay = round(
            MAX_DELAY * self.randomization_factor / self.actual_timestep
        )
        self.min_timestep_delay = round(
            MIN_DELAY * self.randomization_factor / self.actual_timestep
        )
        # generate random delay indexes
        self.action_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.joint_angles_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.joint_velocities_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.local_ang_vel_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.torso_local_velocity_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.local_gravity_vector_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        self.contact_sensor_delay_indices[idx] = (
            random.randint(self.min_timestep_delay, self.max_timestep_delay),
            idx,
        )
        # reset buffers to be only zeros for this env
        self.action_buffer[:, idx] = self.action_buffer[:, idx] * 0
        self.joint_angles_buffer[:, idx] = self.joint_angles_buffer[:, idx] * 0
        self.joint_velocities_buffer[:, idx] = self.joint_velocities_buffer[:, idx] * 0
        self.local_ang_vel_buffer[:, idx] = self.local_ang_vel_buffer[:, idx] * 0
        self.torso_local_velocity_buffer[:, idx] = (
            self.torso_local_velocity_buffer[:, idx] * 0
        )
        self.local_gravity_vector_buffer[:, idx] = (
            self.local_gravity_vector_buffer[:, idx] * 0
        )
        self.contact_sensor_buffer[:, idx] = self.contact_sensor_buffer[:, idx] * 0

    #########################################################
    ############### MUJOCO API HELPER METHODS ###############
    #########################################################

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

    def _apply_action(self, actions):
        # convert actions to jax array clipped from -1 to 1
        actions = jp.clip(jp.array(actions), -1.0, 1.0) * (jp.pi / 2)
        # cycle action through action buffer
        action_to_take = self.action_buffer[0]
        self.action_buffer[0:-1] = self.action_buffer[1:]
        self.action_buffer[self.action_delay_indices] = actions
        # apply actions (convert to radians first)
        self.data_batch = self.data_batch.replace(ctrl=action_to_take)

        self.action_change = action_to_take - self.last_actions
        self.last_actions = action_to_take

    #########################################################
    ################ MAIN PUBLIC API METHODS ################
    #########################################################

    def reset(self, seed=None, options=None, idx=None):
        self.close()

        # APPLY SEED
        if seed is not None:
            self.seed(seed)
        # POPULATE INDICES TO RESET
        if idx is None:
            idx = np.full(self.num_envs, fill_value=True)

        # randomize control inputs
        if USE_CONTROL_INPUTS:
            control_inputs_velocity, control_inputs_yaw = (
                self._get_random_control_inputs()
            )
            self.control_inputs_velocity[idx] = control_inputs_velocity[idx]
            self.control_inputs_yaw[idx] = control_inputs_yaw[idx]

        # randomize models and data for requested envs
        for i in np.arange(self.num_envs)[idx]:
            # make model
            cpu_model = self._make_model()
            # randomize model
            cpu_model = self._randomize_dynamics(cpu_model)
            cpu_model = self._randomize_joint_properties(cpu_model)
            # convert model to GPU and make data from it
            gpu_model = mjx.put_model(cpu_model)
            cpu_data = mujoco.MjData(cpu_model)
            mujoco.mj_kinematics(cpu_model, cpu_data)
            gpu_data = mjx.put_data(cpu_model, cpu_data)
            # randomize GPU data
            gpu_data = self._randomize_joint_positions(gpu_data)

            # insert data and model into batches
            self._replace_model_index(i, gpu_model)
            self._replace_data_index(i, gpu_data)

            # randomize delay for obs/action buffers
            self._randomize_delays(i)

        # update variables keeping track of simulation properties
        self._update_sim_trackers(idx)

        return self._get_obs(idx)

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

    def _get_obs(self, idx=None):

        if idx is None:
            idx = np.full(self.num_envs, fill_value=True)

        # joint positions
        joint_pos_noise = (
            self.randomization_factor
            * (JOINT_ANGLE_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(
                key=self.rng_key, shape=(self.num_envs, len(self.joint_qpos_idx))
            )
        )
        _joint_angles = self._get_joint_positions()[idx] + joint_pos_noise[idx]

        # joint velocities
        joint_vel_noise = (
            self.randomization_factor
            * (JOINT_VELOCITY_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(
                key=self.rng_key, shape=(self.num_envs, len(self.joint_qpos_idx))
            )
        )
        _joint_velocities = self._get_joint_velocities()[idx] + joint_vel_noise[idx]

        # local angular velocity
        ang_vel_noise = (
            self.randomization_factor
            * (GYRO_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3))
        )
        torso_quat = self._get_torso_quaternion()[idx]
        _local_ang_vel = (
            inverseRotateVectors(torso_quat, self._get_torso_angular_velocity()[idx])
            + ang_vel_noise[idx]
        )

        # local velocity
        local_vel_noise = (
            self.randomization_factor
            * VELOCIMETER_NOISE_STDDEV
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 3))
        )
        _torso_local_velocity = (
            inverseRotateVectors(torso_quat, self._get_torso_velocity()[idx])
            + local_vel_noise[idx]
        )

        # gravity direction (local)
        quaternion_noise = (
            self.randomization_factor
            * (IMU_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(self.num_envs, 4))
        )
        noisy_torso_quat = torso_quat + quaternion_noise[idx]
        _local_gravity_vector = inverseRotateVectors(
            noisy_torso_quat, self.gravity_vector_batch[idx]
        )

        # foot contact states
        _contact_states = self.getContactSensorData(
            self.pressure_sensor_ids, self.data_batch[idx]
        )

        # cycle observations through observation buffers
        # get the first (oldest) observation in the buffer
        joint_angles = self.joint_angles_buffer[0]
        joint_velocities = self.joint_velocities_buffer[0]
        local_ang_vel = self.local_ang_vel_buffer[0]
        torso_local_velocity = self.torso_local_velocity_buffer[0]
        local_gravity_vector = self.local_gravity_vector_buffer[0]
        contact_states = self.contact_sensor_buffer[0]
        # shift all observations in the buffers
        self.joint_angles_buffer[0:-1] = self.joint_angles_buffer[1:]
        self.joint_velocities_buffer[0:-1] = self.joint_velocities_buffer[1:]
        self.local_ang_vel_buffer[0:-1] = self.local_ang_vel_buffer[1:]
        self.torso_local_velocity_buffer[0:-1] = self.torso_local_velocity_buffer[1:]
        self.local_gravity_vector_buffer[0:-1] = self.local_gravity_vector_buffer[1:]
        self.contact_sensor_buffer[0:-1] = self.contact_sensor_buffer[1:]
        # insert latest observations in the buffers
        self.joint_angles_buffer[self.joint_angles_delay_indices[idx]] = _joint_angles
        self.joint_velocities_buffer[self.joint_velocities_delay_indices[idx]] = (
            _joint_velocities
        )
        self.local_ang_vel_buffer[self.local_ang_vel_delay_indices[idx]] = (
            _local_ang_vel
        )
        self.torso_local_velocity_buffer[
            self.torso_local_velocity_delay_indices[idx]
        ] = _torso_local_velocity
        self.local_gravity_vector_buffer[
            self.local_gravity_vector_delay_indices[idx]
        ] = _local_gravity_vector
        self.contact_sensor_buffer[self.contact_sensor_delay_indices[idx]] = (
            _contact_states
        )

        # calculate clock phase observations (no delay on these)
        clock_phase_sin = jp.sin(self.data_batch.time).reshape(-1, 1)[idx]
        clock_phase_cos = jp.cos(self.data_batch.time).reshape(-1, 1)[idx]
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

    def step(self, actions=None):
        # apply inputted actions
        if actions is not None:
            self._apply_action(actions)

        # apply forces to the robot to destabilise it
        xfrc_applied = applyExternalForces(self)
        self.data_batch = self.data_batch.replace(xfrc_applied=xfrc_applied)

        # step sims, update data batch
        self.data_batch = self.jax_step(self.model_batch, self.data_batch)

        obs = self._get_obs()
        reward, terminal = self._get_rewards()

        truncated = self.data_batch.time >= self.max_simulation_time
        done = np.logical_or(truncated, terminal)

        info = [{}] * self.num_envs
        # automatically reset environment if any instance terminates
        for env_idx in range(self.num_envs):
            if done[env_idx]:
                info[env_idx]["terminal_observation"] = obs[env_idx]
                info[env_idx]["TimeLimit.truncated"] = truncated[env_idx]

        obs[done] = self.reset(idx=done)

        return obs, reward, done, info


if __name__ == "__main__":
    sim_batch = GPUVecEnv(
        num_envs=2,
        xml_path=SIM_XML_PATH,
        reward_fn=controlInputRewardFn,
        randomization_factor=1,
    )

    obs = sim_batch.reset()
    sim_batch.step()

    total_step_time = 0
    total_step_calls = 0

    total_reward = 0
    n_steps = 0

    while True:
        actions = np.random.uniform(-1, 1, (sim_batch.num_envs, len(JOINT_NAMES)))
        actions = np.zeros((sim_batch.num_envs, len(JOINT_NAMES)))

        start_time = time.time()
        obs, rewards, terminals, _ = sim_batch.step(actions)
        end_time = time.time()
        total_step_time += end_time - start_time
        total_step_calls += 1

        # print(
        #     f"{sim_batch.num_envs} Step Time: {total_step_time / (sim_batch.num_envs*total_step_calls)}"
        # )

        rewards[rewards == -100] = 0
        total_reward += np.mean(rewards)
        n_steps += 1

        if terminals.any():
            print("Avg. Total Reward: ", total_reward, "Episode Length: ", n_steps)
            total_reward = 0
            n_steps = 0
        # print(rewards[0])
        if np.isnan(obs).any() or np.isnan(rewards).any() or np.isnan(terminals).any():
            print("ERROR: NaN value in observations/rewards/terminals.")
