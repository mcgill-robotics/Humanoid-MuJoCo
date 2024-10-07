import jax

import numpy as jp

# if jax.default_backend() != "gpu":
#     print(" >> Using CPU backend for NumPy.")
#     import numpy as jp
# else:
#     print(" >> Using GPU backend for NumPy.")
#     from jax import numpy as jp

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
import cv2
import random
from simulation.reward_functions import *
from simulation.simulation_parameters import *
from jax.scipy.spatial.transform import Rotation
from simulation import SIM_XML_PATH, reward_functions
import gc
import time
from perlin_noise import PerlinNoise

inverseRotateVectors = (
    lambda q, v: Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(v)
)


class CPUEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        reward_fn=SELECTED_REWARD_FUNCTION,
        xml_path=SIM_XML_PATH,
        randomization_factor=0,
        use_potential_rewards=USE_POTENTIAL_REWARDS,
        max_simulation_time_override=None,
        enable_rendering=False,
        reward_override=None,
    ):
        self.xml_path = xml_path
        self.randomization_factor = randomization_factor
        self.timestep = TIMESTEP
        self.use_potential_rewards = bool(use_potential_rewards)
        if type(reward_fn) == str:
            reward_fn = getattr(reward_functions, reward_fn)
        self.reward_fn = reward_fn
        self.physics_steps_per_control_step = PHYSICS_STEPS_PER_CONTROL_STEP
        self.num_envs = 1
        self.rng_key = jax.random.PRNGKey(0)
        self.enable_rendering = enable_rendering

        self.action_space = spaces.Box(
            -1, 1, shape=(len(JOINT_NAMES),), dtype=np.float64
        )
        observation_size = len(JOINT_NAMES) * 2 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            -10, 10, shape=(observation_size,), dtype=np.float64
        )
        if self.reward_fn == standupReward:
            self.IS_STANDUP = True
        else:
            self.IS_STANDUP = False
        if self.IS_STANDUP:
            _MAX_SIM_TIME = MAX_SIM_TIME_STANDUP
        else:
            _MAX_SIM_TIME = MAX_SIM_TIME
        self.max_simulation_time = (
            max_simulation_time_override
            if max_simulation_time_override is not None
            else _MAX_SIM_TIME
        )
        if self.max_simulation_time < 0:
            self.max_simulation_time = np.inf

        self.reward_override = reward_override

        self._init_models()

    def _init_models(self):
        # load model from XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.timestep

        # Visualization Options:
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.model.vis.scale.contactwidth = 0.1
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.scale.forcewidth = 0.05
        self.model.vis.map.force = 0.3

        # LOAD BASE MODEL
        self.base_model = mujoco.MjModel.from_xml_path(self.xml_path)

        ### SAVE MODEL IDs
        # save torso body index
        self.torso_idx = self.model.body(TORSO_BODY_NAME).id
        self.free_joint_qpos_idx = self.model.jnt_qposadr[
            self.model.joint(FREE_JOINT_NAME).id
        ]
        # save joint addresses
        self.joint_qpos_idx = []
        self.joint_dof_idx = []
        self.joint_actuator_idx = []
        for joint in JOINT_NAMES:
            self.joint_dof_idx.append(self.model.jnt_dofadr[self.model.joint(joint).id])
            self.joint_qpos_idx.append(
                self.model.jnt_qposadr[self.model.joint(joint).id]
            )
            self.joint_actuator_idx.append(self.model.actuator(joint).id)
        self.joint_qpos_idx = jp.array(self.joint_qpos_idx)
        self.joint_dof_idx = jp.array(self.joint_dof_idx)
        self.joint_actuator_idx = jp.array(self.joint_actuator_idx)
        # get pressure sensor geometries
        # self.pressure_sensor_ids = [
        #     self.model.geom(pressure_sensor_geom).id
        #     for pressure_sensor_geom in PRESSURE_GEOM_NAMES
        # ]
        self.non_robot_geom_ids = []
        for geom in NON_ROBOT_GEOMS:
            try:
                self.non_robot_geom_ids.append(self.model.geom(geom).id)
            except:
                print(f"Failed to find {geom} geom ID.")

    def _randomize_delays(self):
        # delays in actions and observations (10ms to 50ms)
        # round delays to be integer delays in timesteps
        actual_timestep = self.timestep * self.physics_steps_per_control_step
        self.action_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.action_delay = round(self.action_delay / actual_timestep)
        self.joint_observation_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.joint_observation_delay = round(
            self.joint_observation_delay / actual_timestep
        )
        self.local_ang_vel_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.local_ang_vel_delay = round(self.local_ang_vel_delay / actual_timestep)
        self.torso_local_velocity_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.torso_local_velocity_delay = round(
            self.torso_local_velocity_delay / actual_timestep
        )
        self.torso_z_pos_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.torso_z_pos_delay = round(self.torso_z_pos_delay / actual_timestep)
        self.local_gravity_vector_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.local_gravity_vector_delay = round(
            self.local_gravity_vector_delay / actual_timestep
        )
        self.contact_sensor_delay = random.uniform(
            MIN_DELAY * self.randomization_factor, MAX_DELAY * self.randomization_factor
        )
        self.contact_sensor_delay = round(self.contact_sensor_delay / actual_timestep)

        # make buffers for observations and actions
        self.action_buffer = [self.data.ctrl] * (int)(self.action_delay)
        self.joint_angles_buffer = [jp.array([0] * len(JOINT_NAMES))] * (int)(
            self.joint_observation_delay
        )
        self.joint_velocities_buffer = [jp.array([0] * len(JOINT_NAMES))] * (int)(
            self.joint_observation_delay
        )
        self.local_ang_vel_buffer = [jp.array([0] * 3)] * (int)(
            self.local_ang_vel_delay
        )
        self.torso_local_velocity_buffer = [jp.array([0] * 3)] * (int)(
            self.torso_local_velocity_delay
        )
        self.torso_z_buffer = [0] * (int)(self.torso_z_pos_delay)
        self.local_gravity_vector_buffer = [jp.array([0, 0, -1])] * (int)(
            self.local_gravity_vector_delay
        )
        self.contact_sensor_buffer = [(0, 0)] * (int)(self.contact_sensor_delay)

    def _randomize_joint_properties(self):
        # randomize joint properties
        for joint in JOINT_NAMES:
            self.model.joint(joint).armature[0] = self.base_model.joint(joint).armature[
                0
            ] + (
                random.uniform(0, JOINT_ARMATURE_MAX_CHANGE) * self.randomization_factor
            )
            self.model.joint(joint).stiffness[0] = self.base_model.joint(
                joint
            ).stiffness[0] + (
                random.uniform(0, JOINT_STIFFNESS_MAX_CHANGE)
                * self.randomization_factor
            )
            self.model.joint(joint).margin[0] = self.base_model.joint(joint).margin[
                0
            ] + (random.uniform(0, JOINT_MARGIN_MAX_CHANGE) * self.randomization_factor)
            self.model.joint(joint).range[0] = self.base_model.joint(joint).range[0] + (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
            self.model.joint(joint).range[1] = self.base_model.joint(joint).range[1] + (
                random.uniform(-JOINT_RANGE_MAX_CHANGE, JOINT_RANGE_MAX_CHANGE)
                * self.randomization_factor
            )
        for joint in JOINT_NAMES:
            kp = (
                JOINT_P_GAIN
                + random.uniform(-JOINT_P_GAIN_MAX_CHANGE, JOINT_P_GAIN_MAX_CHANGE)
                * self.randomization_factor
            )
            self.model.actuator(joint).gainprm[0] = kp
            self.model.actuator(joint).biasprm[1] = -kp
            self.model.actuator(joint).forcerange[0] = self.base_model.actuator(
                joint
            ).forcerange[0] + (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )
            self.model.actuator(joint).forcerange[1] = self.base_model.actuator(
                joint
            ).forcerange[1] + (
                random.uniform(
                    -JOINT_FORCE_LIMIT_MAX_CHANGE, JOINT_FORCE_LIMIT_MAX_CHANGE
                )
                * self.randomization_factor
            )

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
            for coef in self.base_model.geom("floor").friction
        ]
        # vary the mass of all limbs randomly
        for i in range(self.model.nbody - 1):
            self.model.body(i + 1).mass[0] = max(
                0.00001,
                self.base_model.body(i + 1).mass[0]
                + random.uniform(
                    -MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                    MAX_MASS_CHANGE_PER_LIMB * self.randomization_factor,
                ),
            )
        # attach a random external mass (up to 0.1 kg) to a randomly chosen limb
        self.model.body(random.randint(1, self.model.nbody - 1)).mass[
            0
        ] += random.uniform(0, MAX_EXTERNAL_MASS_ADDED * self.randomization_factor)

    def _randomize_floor_heightmap(self):
        noise = PerlinNoise(octaves=15)
        xpix, ypix = 8, 8
        height_field = np.array(
            [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
        )
        height_field -= np.min(height_field)
        height_field /= np.max(height_field)

        max_floor_height = MIN_FLOOR_BUMP_HEIGHT + self.randomization_factor * (
            MAX_FLOOR_BUMP_HEIGHT - MIN_FLOOR_BUMP_HEIGHT
        )

        self.model.hfield_data = height_field.reshape(-1) * max_floor_height

    def _randomize_joint_positions(self):
        # randomize joint initial states (CPU)
        joint_pos_range = JOINT_INITIAL_OFFSET_MIN + self.randomization_factor * (
            JOINT_INITIAL_OFFSET_MAX - JOINT_INITIAL_OFFSET_MIN
        )
        for i in self.joint_qpos_idx:
            random_val = random.uniform(-joint_pos_range, joint_pos_range)
            self.data.qpos[i] = self.data.qpos[i] + random_val

    def _randomize_starting_position(self):
        # randomize initial position of the robot
        xy_offset = XY_POSITION_INITIAL_OFFSET_MIN + self.randomization_factor * (
            XY_POSITION_INITIAL_OFFSET_MAX - XY_POSITION_INITIAL_OFFSET_MIN
        )
        z_offset = Z_POSITION_INITIAL_OFFSET_MIN + self.randomization_factor * (
            Z_POSITION_INITIAL_OFFSET_MAX - Z_POSITION_INITIAL_OFFSET_MIN
        )
        quat_offset = QUAT_INITIAL_OFFSET_MIN + self.randomization_factor * (
            QUAT_INITIAL_OFFSET_MAX - QUAT_INITIAL_OFFSET_MIN
        )
        if self.IS_STANDUP:
            _Z_POS = Z_INITIAL_POS_STANDUP
            _INITIAL_QUAT = INITIAL_QUAT_STANDUP
        else:
            _Z_POS = Z_INITIAL_POS
            _INITIAL_QUAT = INITIAL_QUAT
        self.data.qpos[self.free_joint_qpos_idx] = X_INITIAL_POS + random.uniform(
            -xy_offset, xy_offset
        )
        self.data.qpos[self.free_joint_qpos_idx + 1] = Y_INITIAL_POS + random.uniform(
            -xy_offset, xy_offset
        )
        self.data.qpos[self.free_joint_qpos_idx + 2] = _Z_POS + random.uniform(
            0, z_offset
        )
        self.data.qpos[self.free_joint_qpos_idx + 3] = _INITIAL_QUAT[
            0
        ] + random.uniform(-quat_offset, quat_offset)
        self.data.qpos[self.free_joint_qpos_idx + 4] = _INITIAL_QUAT[
            1
        ] + random.uniform(-quat_offset, quat_offset)
        self.data.qpos[self.free_joint_qpos_idx + 5] = _INITIAL_QUAT[
            2
        ] + random.uniform(-quat_offset, quat_offset)
        self.data.qpos[self.free_joint_qpos_idx + 6] = _INITIAL_QUAT[
            3
        ] + random.uniform(-quat_offset, quat_offset)

    def _randomize_control_inputs(self):
        # initialize random control inputs
        if USE_CONTROL_INPUTS:
            self.control_input_velocity = jp.array(
                [
                    random.uniform(
                        -1 * CONTROL_INPUT_MAX_VELOCITY,
                        CONTROL_INPUT_MAX_VELOCITY,
                    ),
                    random.uniform(
                        -1 * CONTROL_INPUT_MAX_VELOCITY,
                        CONTROL_INPUT_MAX_VELOCITY,
                    ),
                ]
            )
            self.control_input_yaw = jp.array(
                [
                    random.uniform(
                        -1 * CONTROL_INPUT_MAX_YAW,
                        CONTROL_INPUT_MAX_YAW,
                    )
                ]
            )
            if RANDOMIZATION_FACTOR_AFFECTS_CONTROL_INPUT:
                self.control_inputs_yaw = (
                    self.control_inputs_yaw * self.randomization_factor
                )
                self.control_inputs_velocity = (
                    self.control_inputs_velocity * self.randomization_factor
                )
        else:
            self.control_input_velocity = jp.array([0, 0])
            self.control_input_yaw = jp.array([0])

    def _init_sim_trackers(self):
        self.previous_action = self.data.ctrl
        self.latest_action = self.data.ctrl
        self.previous_reward, _, _ = self._get_reward()
        self.next_force_start_time = 0
        self.next_force_direction = jp.zeros((2))
        self.next_force_magnitude = 0
        self.next_force_duration = 0
        self.next_force_body = 0

    def reset(self, seed=None, options=None):
        try:
            del self.renderer
        except:
            pass

        super().reset(seed=seed)
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
        else:
            self.rng_key = jax.random.PRNGKey(random.randint(0, 100))

        self._randomize_control_inputs()

        # randomize model-dependent properties
        self._randomize_dynamics()
        self._randomize_joint_properties()
        self._randomize_floor_heightmap()

        if self.enable_rendering:
            self.renderer = mujoco.Renderer(self.model, 720, 1080)

        # create data from model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)

        # call self.data-dependent randomizations
        self._randomize_delays()
        self._randomize_joint_positions()
        self._randomize_starting_position()

        # initialize environment trackers
        self._init_sim_trackers()

        # clean up any unreferenced variables
        gc.collect()

        _, _, terminated, truncated, _ = self.step()
        if self._check_collision() or terminated or truncated:
            # print("COLLISION: Re-generating a starting state")
            return self.reset()

        return self._get_obs(), {}

    def _get_torso_velocity(self):  # NOTE: in global reference frame (NWU)
        return self.data.qvel[:3]

    def _get_torso_angular_velocity(self):  # NOTE: in global reference frame (NWU)
        return self.data.qvel[3:6]

    def _get_joint_torques(self):
        return (
            self.data.qfrc_constraint[self.joint_dof_idx]
            + self.data.qfrc_smooth[self.joint_dof_idx]
        )

    def _get_joint_positions(self):
        return self.data.qpos[self.joint_qpos_idx]

    def _get_joint_velocities(self):
        return self.data.qvel[self.joint_dof_idx]

    def _get_torso_quaternion(self):  # in NWU
        return self.data.xquat[self.torso_idx]

    def _get_torso_z_pos(self):
        return self.data.xpos[self.torso_idx, 2]

    def _get_contact_sensor_data(self):
        pressure_values = np.zeros((8))
        for i in range(len(self.pressure_sensor_ids)):
            for ci in range(len(self.data.contact.geom1)):
                if self.data.contact.geom1[ci] == self.pressure_sensor_ids[i]:
                    pressure_values[i] += (
                        abs(self.data.efc_force[self.data.contact.efc_address[ci]])
                        + 0.0001  # add a small amount to the force since sometimes contacts have a force of 0 (for binary contact checking, we want all contacts to have a force)
                    )
                if self.data.contact.geom2[ci] == self.pressure_sensor_ids[i]:
                    pressure_values[i] += (
                        abs(self.data.efc_force[self.data.contact.efc_address[ci]])
                        + 0.0001  # add a small amount to the force since sometimes contacts have a force of 0 (for binary contact checking, we want all contacts to have a force)
                    )
        pressure_values = np.where(pressure_values > MIN_FORCE_FOR_CONTACT, 1.0, 0.0)
        binary_foot_contact_state_left = np.clip(
            np.sum(pressure_values[: len(self.pressure_sensor_ids) // 2]), 0, 1
        )
        binary_foot_contact_state_right = np.clip(
            np.sum(pressure_values[len(self.pressure_sensor_ids) // 2 :]), 0, 1
        )
        return binary_foot_contact_state_left, binary_foot_contact_state_right

    def _get_obs(self):

        # joint positions
        joint_pos_noise = (
            self.randomization_factor
            * (JOINT_ANGLE_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=[len(self.joint_qpos_idx)])
        )
        joint_angles = self._get_joint_positions() + joint_pos_noise

        # joint velocities
        joint_vel_noise = (
            self.randomization_factor
            * (JOINT_VELOCITY_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=[len(self.joint_qpos_idx)])
        )
        joint_velocities = self._get_joint_velocities() + joint_vel_noise

        # local angular velocity
        ang_vel_noise = (
            self.randomization_factor
            * (GYRO_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(3,))
        )

        torso_quat = self._get_torso_quaternion()
        # local_ang_vel = (
        #     inverseRotateVectors(torso_quat, self._get_torso_angular_velocity())
        #     + ang_vel_noise
        # )

        local_ang_vel = self._get_torso_angular_velocity() + ang_vel_noise

        # local velocity
        # local_vel_noise = (
        #     self.randomization_factor
        #     * VELOCIMETER_NOISE_STDDEV
        #     * jax.random.normal(key=self.rng_key, shape=(3,))
        # )
        # torso_local_velocity = (
        #     inverseRotateVectors(torso_quat, self._get_torso_velocity())
        #     + local_vel_noise
        # )
        torso_z = self._get_torso_z_pos()
        torso_z += (
            MIN_Z_POS_SENSOR_NOISE_STDDEV
            + self.randomization_factor
            * (MAX_Z_POS_SENSOR_NOISE_STDDEV - MIN_Z_POS_SENSOR_NOISE_STDDEV)
        ) * jax.random.normal(key=self.rng_key)

        # gravity direction (local)
        quaternion_noise = (
            self.randomization_factor
            * (IMU_NOISE_STDDEV / 180.0 * jp.pi)
            * jax.random.normal(key=self.rng_key, shape=(4,))
        )
        noisy_torso_quat = torso_quat + quaternion_noise
        self.torso_quat = noisy_torso_quat
        local_gravity_vector = inverseRotateVectors(
            noisy_torso_quat, jp.array([0, 0, -1])
        )

        # foot contact states
        # binary_foot_contact_state_left, binary_foot_contact_state_right = (
        #     self._get_contact_sensor_data()
        # )

        # cycle observations through observation buffers
        self.joint_angles_buffer.append(joint_angles)
        self.joint_velocities_buffer.append(joint_velocities)
        self.local_ang_vel_buffer.append(local_ang_vel)
        # self.torso_local_velocity_buffer.append(torso_local_velocity)
        self.torso_z_buffer.append(torso_z)
        self.local_gravity_vector_buffer.append(local_gravity_vector)
        # self.contact_sensor_buffer.append(
        #     (binary_foot_contact_state_left, binary_foot_contact_state_right)
        # )
        # get oldest (delayed) observations
        joint_angles = self.joint_angles_buffer.pop(0)
        joint_velocities = self.joint_velocities_buffer.pop(0)
        local_ang_vel = self.local_ang_vel_buffer.pop(0)
        # torso_local_velocity = self.torso_local_velocity_buffer.pop(0)
        torso_z = self.torso_z_buffer.pop(0)
        local_gravity_vector = self.local_gravity_vector_buffer.pop(0)
        # binary_foot_contact_state_left, binary_foot_contact_state_right = (
        #     self.contact_sensor_buffer.pop(0)
        # )

        # calculate clock phase observations (no delay on these)
        # clock_phase_sin = jp.array([jp.sin(self.data.time)])
        # clock_phase_cos = jp.array([jp.cos(self.data.time)])
        # clock_phase_complex = (
        #     (clock_phase_sin)
        #     / (2 * jp.sqrt((clock_phase_sin * clock_phase_sin) + 0.04))
        # ) + 0.5

        # concatenate all observations into a single array
        delayed_observations = jp.concatenate(
            (
                joint_angles,  # rad
                joint_velocities,  # rad / s
                local_ang_vel,  # rad/s
                # torso_local_velocity,  # m/s
                local_gravity_vector,  # unit vector
                torso_z,  # m
                # np.array([binary_foot_contact_state_left]),
                # np.array([binary_foot_contact_state_right]),
                # self.control_input_velocity,  # as defined in reset
                # self.control_input_yaw,  # as defined in reset
                # clock_phase_sin,  # as defined in paper on potential rewards
                # clock_phase_cos,  # as defined in paper on potential rewards
                # clock_phase_complex,  # as defined in paper on potential rewards
            )
        )

        return np.array(delayed_observations, dtype=np.float64)

    def _check_collision(self):
        return len(self.data.contact.geom1) > 0

    def _check_self_collision(self):
        self_collision = False
        for i in range(len(self.data.contact.geom1)):
            if (
                self.data.contact.geom1[i] not in self.non_robot_geom_ids
                and self.data.contact.geom2[i] not in self.non_robot_geom_ids
            ):
                self_collision = True
        return self_collision

    def _get_reward(self):

        torso_global_velocity = self._get_torso_velocity()
        torso_z_pos = self._get_torso_z_pos()
        torso_quat = self._get_torso_quaternion()
        joint_torques = self._get_joint_torques()
        is_self_colliding = self._check_self_collision()

        reward, isTerminal, truncated = self.reward_fn(
            torso_global_velocity,
            self.control_input_velocity,
            torso_quat,
            self.control_input_yaw,
            torso_z_pos,
            joint_torques,
            self.previous_action / (jp.pi / 2),
            self.latest_action / (jp.pi / 2),
            is_self_colliding,
            self.data.time,
            self.max_simulation_time,
        )

        if self.reward_override is not None:
            return float(self.reward_override), bool(isTerminal), bool(truncated)

        if self.use_potential_rewards:
            _reward = reward - self.previous_reward
            self.previous_reward = reward
            reward = _reward

        return float(reward), bool(isTerminal), bool(truncated)

    def _apply_external_forces(self):
        # apply forces to the robot to destabilise it
        if self.data.time >= self.next_force_start_time + self.next_force_duration:
            self.next_force_start_time = self.data.time + random.uniform(
                MIN_EXTERNAL_FORCE_INTERVAL, MAX_EXTERNAL_FORCE_INTERVAL
            )
            self.next_force_duration = random.uniform(
                MIN_EXTERNAL_FORCE_DURATION, MAX_EXTERNAL_FORCE_DURATION
            )
            self.next_force_magnitude = random.uniform(
                MIN_EXTERNAL_FORCE_MAGNITUDE * self.randomization_factor,
                MAX_EXTERNAL_FORCE_MAGNITUDE * self.randomization_factor,
            )
            self.next_force_direction = np.array(
                [random.uniform(-1, 1), random.uniform(-1, 1)]
            )
            self.data.xfrc_applied[self.next_force_body][0] = 0
            self.data.xfrc_applied[self.next_force_body][1] = 0
            self.next_force_body = random.randint(1, len(self.data.xfrc_applied) - 1)

            while np.linalg.norm(self.next_force_direction) == 0:
                self.next_force_direction = np.array(
                    [random.uniform(-1, 1), random.uniform(-1, 1)]
                )
            self.next_force_direction = self.next_force_direction / np.linalg.norm(
                self.next_force_direction
            )
        if (
            self.data.time > self.next_force_start_time
            and self.data.time < self.next_force_start_time + self.next_force_duration
        ):
            self.data.xfrc_applied[self.next_force_body][0] = (
                self.next_force_direction[0] * self.next_force_magnitude
            )
            self.data.xfrc_applied[self.next_force_body][1] = (
                self.next_force_direction[1] * self.next_force_magnitude
            )

    def _apply_action(self, action):
        # cycle action through action buffer
        if action is None:
            action = self.data.ctrl
        else:
            action_noise = (
                self.randomization_factor
                * (JOINT_ACTION_NOISE_STDDEV / 180.0 * jp.pi)
                * jax.random.normal(key=self.rng_key, shape=action.shape)
            )
            action = action + action_noise
        self.action_buffer.append(action)
        action_to_take = self.action_buffer.pop(0)
        # actions should be inputted to the environment in the -1 to 1 range, and they are mapped here to -pi/2 and pi/2 accordingly
        # action_to_take = jp.clip(jp.array(action_to_take), -1, 1)
        # action_to_take = action_to_take * (jp.pi / 2)
        self.data.ctrl = action_to_take
        self.previous_action = self.latest_action
        self.latest_action = action_to_take

    def step(self, action=None):

        self._apply_action(action)

        self._apply_external_forces()

        # step simulation
        for _ in range(self.physics_steps_per_control_step):
            mujoco.mj_step(self.model, self.data)

        reward, terminated, truncated = self._get_reward()

        if terminated or truncated:
            info = {"is_success": truncated}
        else:
            info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if self.enable_rendering:
            self.renderer.update_scene(
                self.data, camera="track", scene_option=self.scene_option
            )
            frame = self.renderer.render()
            time.sleep(1 / CONTROL_FREQUENCY)
            if mode == "human":
                cv2.imshow("CPU Sim View", frame)
                cv2.waitKey(1)
            else:
                return frame


if __name__ == "__main__":
    sim = CPUEnv(
        xml_path=SIM_XML_PATH,
        reward_fn=SELECTED_REWARD_FUNCTION,
        randomization_factor=0,
        enable_rendering=True,
    )
    obs = sim.reset()

    total_step_time = 0
    total_step_calls = 0

    total_reward = 0
    n_steps = 0

    while True:
        action = np.random.uniform(-1, 1, len(JOINT_NAMES))
        # action = 0 * np.ones(len(JOINT_NAMES))

        start_time = time.time()
        obs, reward, isTerminal, isTruncated, _ = sim.step(action)
        # print(obs)
        end_time = time.time()
        total_step_time += end_time - start_time
        total_step_calls += 1
        # print(f"Step Time: {total_step_time / total_step_calls}")

        total_reward += reward
        n_steps += 1

        # print(reward)
        sim.render("human")

        if isTerminal or isTruncated:
            print(
                "Cumulative Reward: ",
                total_reward,
                "Episode Length: ",
                n_steps,
            )
            total_reward = 0
            n_steps = 0
            sim.reset()
