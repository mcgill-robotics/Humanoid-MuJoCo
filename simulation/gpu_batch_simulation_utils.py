import jax
from jax import numpy as jp
from simulation.simulation_parameters import *
from simulation.reward_functions import *
from jax.scipy.spatial.transform import Rotation
import numpy as np

### ADAPTED FROM MUJOCO SOURCE CODE https://github.com/google-deepmind/mujoco/blob/4ad1dc4b84b408ae585f5553fc4e78da7c95e530/mjx/mujoco/mjx/_src/io.py#L218
### ALLOWS US TO AVOID HAVING TO CALL get_data WHICH IS VERY SLOW
def getFootForces(pressure_sensor_ids, d):
  # below needs m, pressure_sensor_ids arguments
  # ### ALL CODE BELOW WAS TAKEN FROM MUJOCO CODE, USED TO CALCULATE efc_address VALUES
  # ne_connect = (m.eq_type == mujoco.mjtEq.mjEQ_CONNECT).sum()
  # ne_weld = (m.eq_type == mujoco.mjtEq.mjEQ_WELD).sum()
  # ne_joint = (m.eq_type == mujoco.mjtEq.mjEQ_JOINT).sum()
  # ne = ne_connect * 3 + ne_weld * 6 + ne_joint
  # nf = 0
  # nl = int(m.jnt_limited.sum())
  # nc = d.efc_J.shape[-2] - ne - nf - nl
  # efc_type = jp.array([
  #     0,#mujoco.mjtConstraint.mjCNSTR_EQUALITY,
  #     1,#mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
  #     2,#mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,
  #     3,#mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
  # ]).repeat(jp.array([ne, nf, nl, nc]))
  # efc_active = (d.efc_J != 0).any(axis=1)
  # efc_con = efc_type == 3#mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
  # nefc, nc = efc_active.sum(), (efc_active & efc_con).sum()
  # efc_start = nefc - nc
  # ncon = d.contact.dist.shape[0]
  # efc_address = efc_start + jp.arange(0, ncon * 4, 4)
  ##### END OF MUJOCO SOURCE CODE ADAPTATION
      
  ncon = d.contact.dist.shape[0]
  efc_address = jp.arange(0, ncon * 4, 4)
  
  # using efc_address, we can count up the forces on the different pressure sensor geometries
  ordered_efc_forces = jp.abs(d.efc_force[efc_address])
  pressure_values = []
  for ps_id in pressure_sensor_ids:
    forces = ordered_efc_forces * (jp.where(d.contact.geom1 == ps_id, 1, 0) + jp.where(d.contact.geom2 == ps_id, 1, 0))
    pressure_values.append(jp.sum(forces))

  return jp.array(pressure_values)
    

def applyExternalForces(sim_batch):
  # check which simulations need new random force times/durations/magnitudes/directions/target bodies
  should_update_force = sim_batch.data_batch.time > (sim_batch.next_force_start_times + sim_batch.next_force_durations)
  
  xfrc_applied = np.array(sim_batch.data_batch.xfrc_applied)
  
  # for the simulations which need to be updated, randomly generate new values
  if jp.any(should_update_force):
    N = jp.sum(should_update_force)
    updated_next_force_start_times = sim_batch.data_batch.time[should_update_force] + jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_INTERVAL, maxval=MAX_EXTERNAL_FORCE_INTERVAL)
    updated_next_force_durations = jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_DURATION, maxval=MAX_EXTERNAL_FORCE_DURATION)
    updated_next_force_magnitudes = jax.random.uniform(key=sim_batch.rng_key, shape=(N,), minval=MIN_EXTERNAL_FORCE_MAGNITUDE*sim_batch.randomization_factor, maxval=MAX_EXTERNAL_FORCE_MAGNITUDE*sim_batch.randomization_factor)
    updated_next_force_bodies = jax.random.randint(key=sim_batch.rng_key, shape=(N,), minval=1, maxval=sim_batch.data_batch.xfrc_applied.shape[1] - 1)
    updated_next_force_directions = jax.random.ball(key=sim_batch.rng_key, d=2, shape=(N,))
    
    sim_batch.next_force_start_times = sim_batch.next_force_start_times.at[should_update_force].set(updated_next_force_start_times)
    sim_batch.next_force_durations = sim_batch.next_force_durations.at[should_update_force].set(updated_next_force_durations)
    sim_batch.next_force_magnitudes = sim_batch.next_force_magnitudes.at[should_update_force].set(updated_next_force_magnitudes)
    sim_batch.next_force_bodies = sim_batch.next_force_bodies.at[should_update_force].set(updated_next_force_bodies)
    sim_batch.next_force_directions = sim_batch.next_force_directions.at[should_update_force].set(updated_next_force_directions)
    
    xfrc_applied[should_update_force] = np.zeros((N,*sim_batch.data_batch.xfrc_applied.shape[1:]))
    

  # apply force values (times/durations/etc.) to corresponding simulations
  should_apply_force = jp.logical_and((sim_batch.data_batch.time > sim_batch.next_force_start_times), (sim_batch.data_batch.time < (sim_batch.next_force_start_times + sim_batch.next_force_durations)))
  
  if jp.any(should_apply_force):
    applied_forces_x = sim_batch.next_force_directions[should_apply_force][:, 0] * sim_batch.next_force_magnitudes[should_apply_force]
    applied_forces_y = sim_batch.next_force_directions[should_apply_force][:, 1] * sim_batch.next_force_magnitudes[should_apply_force]
    xfrc_applied[np.array(should_apply_force, dtype=bool), np.array(sim_batch.next_force_bodies[np.array(should_apply_force, dtype=bool)], dtype=int), 0:2] = np.array([applied_forces_x, applied_forces_y]).T

  return jp.array(xfrc_applied, dtype=float)

# helper functions
inverseRotateVectors = jax.jit(jax.vmap(lambda q, v : Rotation.from_quat(q).inv().apply(v)))