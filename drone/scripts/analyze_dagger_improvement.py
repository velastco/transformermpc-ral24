import os
import numpy as np
import matplotlib.pyplot as plt

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_root = os.path.join(root_folder, 'optimization/saved_files/closed_loop')

model_base = 'checkpoint_quad_random_forest_ctgrtg'
n_iters = 7

J_mean = []
J_std = []

for i in range(n_iters):
    filename = os.path.join(model_root, f'dagger_{model_base}_cl_{i}.npz')
    if not os.path.exists(filename):
        print(f"Skipping iteration {i}: file not found.")
        continue
    
    data = np.load(filename)
    J_artMPC = data['J_artMPC']
    i_unfeas = data['i_unfeas_artMPC']

    # Filter out unfeasible entries
    feasible_mask = np.ones(J_artMPC.shape[0], dtype=bool)
    feasible_mask[i_unfeas] = False

    J_mean.append(np.mean(J_artMPC[feasible_mask]))
    J_std.append(np.std(J_artMPC[feasible_mask]))

# Plotting
plt.figure(figsize=(8,5))
plt.errorbar(np.arange(len(J_mean)), J_mean, yerr=J_std, fmt='-o', capsize=4, label='ART+MPC')

# Optionally compare with CVX
baseline_file = os.path.join(model_root, f'dagger_{model_base}.npz')
if os.path.exists(baseline_file):
    baseline_data = np.load(baseline_file)
    J_cvx = baseline_data['J_cvx']
    feasible_mask = np.ones(J_cvx.shape[0], dtype=bool)
    feasible_mask[baseline_data['i_unfeas_cvx']] = False
    J_cvx_mean = np.mean(J_cvx[feasible_mask])
    plt.axhline(y=J_cvx_mean, linestyle='--', label='CVX (expert)', color='gray')

plt.xlabel('DAgger Iteration')
plt.ylabel('Control Cost J')
plt.title('Improvement of Transformer via DAgger')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
