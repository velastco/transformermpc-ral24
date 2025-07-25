import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# %matplotlib widget
import matplotlib

from dynamics.quadrotor import QuadModel, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, check_koz_constraint, obs_positions, obs_radii
from optimization.quad_scenario import n_obs, n_time_rpod
import decision_transformer.manage as DT_manager
from dynamics.QuadEnv import QuadEnv
import time
from decision_transformer.art_closed_loop import AutonomousQuadrotorTransformerMPC, ConvexMPC, MyopicConvexMPC#, TwoPointsBoundaryValueProblemMPC, IterativeSequentialConvexProgrammingMPC
import numpy as np
matplotlib.use('Agg')  # or 'TkAgg' 'Qt5Agg'
import matplotlib.pyplot as plt

model_name_base = 'checkpoint_quad_random_forest_ctgrtg'
n_dagger = 0
mean_costs_art = []
std_costs_art = []
mean_costs_cvx = []
std_costs_cvx = []
# Iterate through all DAgger iterations including iteration 0
for i in range(n_dagger + 1):
    if i == 0:
        model_name = model_name_base
    else:
        model_name = f'{model_name_base}_cl_{i-1}'
    path = f'{root_folder}/optimization/saved_files/closed_loop/dagger_{model_name}.npz'
    try:
        data = np.load(path)
        J_art = data['J_artMPC']
        J_cvx = data['J_cvx']
        # Filter out invalid values (e.g., 0 or negative)
        J_art = J_art[J_art > 0]
        J_cvx = J_cvx[J_cvx > 0]
        mean_costs_art.append(np.mean(J_art))
        std_costs_art.append(np.std(J_art))
        mean_costs_cvx.append(np.mean(J_cvx))
        std_costs_cvx.append(np.std(J_cvx))
    except FileNotFoundError:
        print(f'[warning] Missing file: {path}')
        mean_costs_art.append(np.nan)
        std_costs_art.append(np.nan)
        mean_costs_cvx.append(np.nan)
        std_costs_cvx.append(np.nan)
plt.figure(figsize=(10, 6))
x = np.arange(n_dagger + 1)
plt.errorbar(x, mean_costs_art, yerr=std_costs_art, fmt="-o", capsize=5, label='Transformer MPC (J_artMPC)')
plt.errorbar(x, mean_costs_cvx, yerr=std_costs_cvx, fmt="-s", capsize=5, label='Convex MPC (J_cvx)', alpha=0.7)
plt.xlabel('DAgger Iteration')
plt.ylabel('Mean Control Cost')
plt.title('Control Cost per DAgger Iteration')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Simulation configuration
transformer_model_name = 'checkpoint_quad_random_forest_ctgrtg' #'checkpoint_rtn_v11'/'checkpoint_rtn_v01'
transformer_config = DT_manager.transformer_import_config(transformer_model_name)
mdp_constr = transformer_config['mdp_constr']
timestep_norm = transformer_config['timestep_norm']
dataset_scenario = transformer_config['dataset_scenario']
transformer_ws = 'dyn'
datasets, dataloaders = DT_manager.get_train_val_test_data(mdp_constr=mdp_constr, dataset_scenario=dataset_scenario, timestep_norm=timestep_norm)
train_dataset, val_dataset, test_dataset = datasets
train_loader, eval_loader, test_loader = dataloaders
import torch
state_init = np.array([-1.9, 0.3, 0.26, 0, 0, 0])
state_final = np.array([0.4, 0.01, 0.35, 0, 0, 0])
test_sample = next(iter(test_loader))
data_stats = test_loader.dataset.data_stats
test_sample[0][0,:,:] = (torch.tensor(np.repeat(state_init[None,:], 100, axis=0)) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)#(torch.tensor(xs[:-1,:]) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)#
test_sample[1][0,:,:] = torch.zeros((100,3))#(torch.tensor(us) - data_stats['actions_mean'])/(data_stats['actions_std'] + 1e-6)#
test_sample[2][0,:,0] = torch.zeros((100,))#torch.from_numpy(compute_reward_to_go(test_sample[1][0,:,:]))#
test_sample[3][0,:,0] = torch.zeros((100,))#torch.from_numpy(compute_constraint_to_go(test_sample[0][0,:,:].cpu().numpy(), obs_positions, obs_radii))#
test_sample[4][0,:,:] = (torch.tensor(np.repeat(state_final[None,:], 100, axis=0)) - data_stats['goal_mean'])/(data_stats['goal_std'] + 1e-6)
select_idx = True # set to True to manually select a test trajectory via its index (idx)
idx = 39999 # index of the test trajectory (e.g., dx = 654) 3987 35998
# Sample from test dataset
if select_idx:
    test_sample = test_loader.dataset.getix(idx)
else:
    test_sample = next(iter(test_loader))
if mdp_constr:
    states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
else:
    states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
data_stats = test_loader.dataset.data_stats
state_init = (test_sample[0][0,0,:] * data_stats['states_std'][0]) + (data_stats['states_mean'][0])
state_final = (test_sample[4][0,0,:] * data_stats['goal_std'][0]) + (data_stats['goal_mean'][0])

# QuadModel
qm = QuadModel(verbose=True)
dt = dt.item()
time_sec = np.hstack((time_sec[0,0], time_sec[0,0,-1] + dt))
# Warmstarting and optimization
# Solve Convex Problem
states_ws_cvx, actions_ws_cvx, _, feas_cvx = ocp_no_obstacle_avoidance(qm, state_init, state_final, initial_guess='line')
states_ws_cvx = states_ws_cvx.T
actions_ws_cvx = actions_ws_cvx.T
print('CVX cost:', np.sum(la.norm(actions_ws_cvx, axis=0)**2)/2)
constr_cvx, constr_viol_cvx= check_koz_constraint(states_ws_cvx.T, obs_positions, obs_radii)
# Solve SCP
states_scp_cvx, actions_scp_cvx, J_vect_scp_cvx, feas_scp_cvx, iter_scp_cvx = ocp_obstacle_avoidance(qm, states_ws_cvx.T, actions_ws_cvx.T, state_init, state_final)
states_scp_cvx = states_scp_cvx.T
actions_scp_cvx = actions_scp_cvx.T
print('SCP cost:', np.sum(la.norm(actions_scp_cvx, axis=0)**2)/2)
print('J vect', J_vect_scp_cvx)
constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_scp_cvx.T, obs_positions, obs_radii)

# Import the Transformer
model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
model.eval()
inference_func = getattr(DT_manager, 'torch_model_inference_'+transformer_ws)
print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+inference_func.__name__+'()')
rtg = - np.sum(la.norm(actions_ws_cvx, axis=0)**2)/2 if mdp_constr else None #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DT_trajectory, runtime_DT = inference_func(model, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=rtg)
states_ws_DT = np.append(DT_trajectory['xyz_' + transformer_ws], (DT_trajectory['xyz_' + transformer_ws][:,-1] + qm.f(DT_trajectory['xyz_' + transformer_ws][:, -1], DT_trajectory['dv_' + transformer_ws][:, -1])*dt).reshape((6,1)), 1)# set warm start
actions_ws_DT = DT_trajectory['dv_' + transformer_ws]
print('ART cost:', np.sum(la.norm(actions_ws_DT, axis=0)**2)/2)
print('ART runtime:', runtime_DT)
constr_DT, constr_viol_DT = check_koz_constraint(states_ws_DT.T, obs_positions, obs_radii)

# Solve SCP
states_scp_DT, actions_scp_DT, J_vect_scp_DT, feas_scp_DT, iter_scp_DT = ocp_obstacle_avoidance(qm, states_ws_DT.T, actions_ws_DT.T, state_init, state_final)
states_scp_DT = states_scp_DT.T
actions_scp_DT = actions_scp_DT.T
print('SCP cost:', np.sum(la.norm(actions_scp_DT, axis=0)**2)/2)
print('J vect', J_vect_scp_DT)
constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_scp_DT.T, obs_positions, obs_radii)

# Plotting

# 3D position trajectory
ax = plt.figure(figsize=(12,8)).add_subplot(projection='3d')
p1 = ax.plot3D(states_ws_cvx[0,:], states_ws_cvx[1,:], states_ws_cvx[2,:], 'k', linewidth=1.5, label='warm-start cvx')
p2 = ax.plot3D(states_scp_cvx[0,:], states_scp_cvx[1,:], states_scp_cvx[2,:], 'b', linewidth=1.5, label='scp-cvx')
p3 = ax.plot3D(states_ws_DT[0,:], states_ws_DT[1,:], states_ws_DT[2,:], c=[0.5,0.5,0.5], linewidth=1.5, label='warm-start ART-' + transformer_ws)
p4 = ax.plot3D(states_scp_DT[0,:], states_scp_DT[1,:], states_scp_DT[2,:], 'c', linewidth=1.5, label='scp-ART-' + transformer_ws)

for i in range(n_obs):
    p = obs_positions[i]
    r = obs_radii[i]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = p[0] + r * np.outer(np.cos(u), np.sin(v))
    y = p[1] + r * np.outer(np.sin(u), np.sin(v))
    z = p[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3)
ax.set_xlabel('x [m]', fontsize=10)
ax.set_ylabel('y [m]', fontsize=10)
ax.set_zlabel('z [m]', fontsize=10)
ax.grid(True)
ax.legend(loc='best', fontsize=10)
ax.set_aspect('equal')
## plt.savefig(root_folder + '/optimization/saved_files/plots/pos_3d.png')
plt.savefig(root_folder + '/decision_transformer/trajectory_3d.png', dpi=300)
print("Plot saved as trajectory_3d.png")

# Constraint satisfaction
plt.ion()
plt.figure()
plt.plot(time_sec, constr_cvx.T, 'k', linewidth=1.5, label='warm-start cvx')
plt.plot(time_sec, constr_scp_cvx.T, 'b', linewidth=1.5, label='scp-cvx')
plt.plot(time_sec, constr_DT.T, c=[0.5,0.5,0.5], linewidth=1.5, label='warm-start ART-' + transformer_ws)
plt.plot(time_sec, constr_scp_DT.T, 'c', linewidth=1.5, label='scp-ART-' + transformer_ws)
plt.plot(time_sec, np.zeros(n_time_rpod+1), 'r-', linewidth=1.5, label='koz')
plt.xlabel('time [orbits]', fontsize=10)
plt.ylabel('keep-out-zone constraint [-]', fontsize=10)
plt.grid(True)
plt.legend(loc='best', fontsize=10)
## plt.savefig(root_folder + '/optimization/saved_files/plots/constr.png')

plt.show()
plt.show(block=True)  # Keep the window open until you close it
