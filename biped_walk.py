import crocoddyl
import pinocchio
import numpy as np
from crocoddyl.utils.biped import SimpleBipedGaitProblem

# Setting up the robot model
robot = pinocchio.RobotWrapper.BuildFromURDF("id_urdf/humanoid.urdf")
# Create the initial state
q0 = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(12)])
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])
robot.model.referenceConfigurations["half_sitting"] = q0

# Setting up the 3d walking problem
rightFoot = "right_knee_pitch_link"
leftFoot = "left_knee_pitch_link"
gait = SimpleBipedGaitProblem(robot.model, rightFoot, leftFoot)


# Creating the walking problem
stepLength = 0.2  # meters
stepHeight = 0.1  # meters
timeStep = 0.0375  # seconds
stepKnots = 20
supportKnots = 10
problem = gait.createWalkingProblem(
    x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
)

# Solving the 3d walking problem using Feasibility-prone DDP
ddp = crocoddyl.SolverFDDP(problem)

# Using the meshcat displayer, you could enable gepetto viewer for nicer view
display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)
ddp.setCallbacks(
    [
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display),
    ]
)
# Solve the optimal control problem
ddp.th_stop = 1e-9
init_xs = [x0] * (problem.T + 1)
init_us = []
maxiter = 1000
regInit = 0.1
ddp.solve(init_xs, init_us, maxiter, False, regInit)

# Plotting the solution and the DDP convergence
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us)
crocoddyl.plotConvergence(
    log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps
)

# Visualization of the DDP solution in meshcat
display.rate = -1
display.freq = 1
display.displayFromSolver(ddp)