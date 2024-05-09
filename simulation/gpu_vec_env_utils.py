import jax
from jax import numpy as jp
from .simulation_parameters import *
from simulation.reward_functions import *
from jax.scipy.spatial.transform import Rotation
import numpy as np


### ADAPTED FROM MUJOCO SOURCE CODE https://github.com/google-deepmind/mujoco/blob/4ad1dc4b84b408ae585f5553fc4e78da7c95e530/mjx/mujoco/mjx/_src/io.py#L218
### ALLOWS US TO AVOID HAVING TO CALL get_data WHICH IS VERY SLOW
def getContactSensorData(pressure_sensor_ids, d):
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
        forces = ordered_efc_forces * (
            jp.where(d.contact.geom1 == ps_id, 1, 0)
            + jp.where(d.contact.geom2 == ps_id, 1, 0)
        )
        pressure_values.append(jp.sum(forces))

    pressure_values = jp.array(pressure_values)

    # convert to binary (above some threshold is considered a contact point)
    pressure_values = jp.where(pressure_values > MIN_FORCE_FOR_CONTACT, 1.0, 0.0)
    # convert to left/right contact states
    binary_foot_contact_state_left = jp.clip(jp.sum(pressure_values[:4]), 0, 1)
    binary_foot_contact_state_right = jp.clip(jp.sum(pressure_values[4:]), 0, 1)

    return jp.array([binary_foot_contact_state_left, binary_foot_contact_state_right])


def applyExternalForces(sim_batch):
    # check which simulations need new random force times/durations/magnitudes/directions/target bodies
    should_update_force = sim_batch.data_batch.time > (
        sim_batch.next_force_start_times + sim_batch.next_force_durations
    )

    xfrc_applied = np.array(sim_batch.data_batch.xfrc_applied)

    # for the simulations which need to be updated, randomly generate new values
    if jp.any(should_update_force):
        # get the number of envs which need updated forces
        N = jp.sum(should_update_force)
        # generate N random force start times
        updated_next_force_start_times = sim_batch.data_batch.time[
            should_update_force
        ] + jax.random.uniform(
            key=sim_batch.rng_key,
            shape=(N,),
            minval=MIN_EXTERNAL_FORCE_INTERVAL,
            maxval=MAX_EXTERNAL_FORCE_INTERVAL,
        )
        # generate N random force durations
        updated_next_force_durations = jax.random.uniform(
            key=sim_batch.rng_key,
            shape=(N,),
            minval=MIN_EXTERNAL_FORCE_DURATION,
            maxval=MAX_EXTERNAL_FORCE_DURATION,
        )
        # generate N random force magnitudes
        updated_next_force_magnitudes = jax.random.uniform(
            key=sim_batch.rng_key,
            shape=(N,),
            minval=MIN_EXTERNAL_FORCE_MAGNITUDE * sim_batch.randomization_factor,
            maxval=MAX_EXTERNAL_FORCE_MAGNITUDE * sim_batch.randomization_factor,
        )
        # generate N random force target bodies (body to which the force is applied)
        updated_next_force_bodies = jax.random.randint(
            key=sim_batch.rng_key,
            shape=(N,),
            minval=1,
            maxval=sim_batch.data_batch.xfrc_applied.shape[1] - 1,
        )
        # generate N random force directions
        updated_next_force_directions = jax.random.ball(
            key=sim_batch.rng_key, d=2, shape=(N,)
        )

        # update the simulation with the new force start times at the correct indices
        sim_batch.next_force_start_times = sim_batch.next_force_start_times.at[
            should_update_force
        ].set(updated_next_force_start_times)
        # update the simulation with the new force durations at the correct indices
        sim_batch.next_force_durations = sim_batch.next_force_durations.at[
            should_update_force
        ].set(updated_next_force_durations)
        # update the simulation with the new force magnitudes at the correct indices
        sim_batch.next_force_magnitudes = sim_batch.next_force_magnitudes.at[
            should_update_force
        ].set(updated_next_force_magnitudes)
        # update the simulation with the new force target bodies at the correct indices
        sim_batch.next_force_bodies = sim_batch.next_force_bodies.at[
            should_update_force
        ].set(updated_next_force_bodies)
        # update the simulation with the new force directions at the correct indices
        sim_batch.next_force_directions = sim_batch.next_force_directions.at[
            should_update_force
        ].set(updated_next_force_directions)

        # reset the applied forces to zero for the simulations which have had their forces updated
        xfrc_applied[should_update_force] = np.zeros(
            (N, *sim_batch.data_batch.xfrc_applied.shape[1:])
        )

    # apply force values (times/durations/etc.) to simulations which enter the force application window
    should_apply_force = jp.logical_and(
        (sim_batch.data_batch.time > sim_batch.next_force_start_times),
        (
            sim_batch.data_batch.time
            < (sim_batch.next_force_start_times + sim_batch.next_force_durations)
        ),
    )

    if jp.any(should_apply_force):
        # get the x and y components of the forces to be applied
        applied_forces_x = (
            sim_batch.next_force_directions[should_apply_force][:, 0]
            * sim_batch.next_force_magnitudes[should_apply_force]
        )
        applied_forces_y = (
            sim_batch.next_force_directions[should_apply_force][:, 1]
            * sim_batch.next_force_magnitudes[should_apply_force]
        )
        # update the applied forces for the simulations with the components
        xfrc_applied[
            np.array(should_apply_force, dtype=bool),
            np.array(
                sim_batch.next_force_bodies[np.array(should_apply_force, dtype=bool)],
                dtype=int,
            ),
            0:2,
        ] = np.array([applied_forces_x, applied_forces_y]).T

    # finally, return the updated forces to be applied
    return jp.array(xfrc_applied, dtype=float)


def checkSelfCollision(non_robot_geom_ids, d):
    # check if any of the contact points are between robot geometries
    # to do this, we sum up the number of contacts between each robot geometry and any other geometry along each collision index
    robot_collisions = jp.full(d.contact.geom1.shape, 0.0)
    for id in non_robot_geom_ids:
        robot_collisions += jp.where(d.contact.geom1 == id, 1, 0)
        robot_collisions += jp.where(d.contact.geom2 == id, 1, 0)
    # if the sum of robot contacts is greater or equal to 2 in one index (one collision), then we have a self-collision
    return jp.any(robot_collisions >= 2)


# helper function (to turn global vectors into local vectors given orientation quaternion/vector pairs)
inverseRotateVectors = jax.jit(
    jax.vmap(lambda q, v: Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(v))
)

# def tree_unstack(tree):
#     """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
#     For example, given a tree ((a, b), c), where a, b, and c all have first
#     dimension k, will make k trees
#     [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
#     Useful for turning the output of a vmapped function into normal objects.
#     """
#     leaves, treedef = pytree.flatten(tree)
#     n_trees = leaves[0].shape[0]
#     new_leaves = [[] for _ in range(n_trees)]
#     for leaf in leaves:
#         for i in range(n_trees):
#             new_leaves[i].append(leaf[i])
#     new_trees = [treedef.unflatten(l) for l in new_leaves]
#     return new_trees

# def tree_stack(trees):
#     """Takes a list of trees and stacks every corresponding leaf.
#     For example, given two trees ((a, b), c) and ((a', b'), c'), returns
#     ((stack(a, a'), stack(b, b')), stack(c, c')).
#     Useful for turning a list of objects into something you can feed to a
#     vmapped function.
#     """
#     leaves_list = []
#     treedef_list = []
#     for tree in trees:
#         leaves, treedef = pytree.flatten(tree)
#         leaves_list.append(leaves)
#         treedef_list.append(treedef)

#     grouped_leaves = zip(*leaves_list)
#     result_leaves = [jp.stack(l) for l in grouped_leaves]
#     return treedef_list[0].unflatten(result_leaves)