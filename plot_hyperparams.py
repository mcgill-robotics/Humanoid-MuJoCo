import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define the path to the text file
file_path = "tuned_hyperparams.txt"

# Define the hyperparameters you want to plot on x and z axes
x_param = 'log_std_init'
z_param = 'ent_coef'

# Regular expression patterns to extract values
trial_pattern = re.compile(r'\[I \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\] Trial (\d+) finished with value: ([\d.]+) and parameters: ({.*?})')

# Initialize lists to store hyperparameters and scores
x_values = []
z_values = []
y_values = []

# Read the text file and extract relevant information
with open(file_path, 'r') as file:
    for line in file:
        match = trial_pattern.match(line)
        if match:
            trial_num = int(match.group(1))
            score = float(match.group(2))
            params = eval(match.group(3))
            if x_param in params and z_param in params:
                x_values.append(params[x_param])
                z_values.append(params[z_param])
                y_values.append(score)

# Convert lists to numpy arrays
X = np.array(list(zip(x_values, z_values)))
y = np.array(y_values)

# Fit curved surface (polynomial regression)
degree = 3  # Adjust the degree of the polynomial as needed
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# Create a meshgrid for the curved surface using unscaled features
x_range = np.linspace(min(x_values), max(x_values), 10)
z_range = np.linspace(min(z_values), max(z_values), 10)
xx, zz = np.meshgrid(x_range, z_range)
yy = model.predict(np.array(list(zip(xx.ravel(), zz.ravel())))).reshape(xx.shape)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the curved surface with colored surface based on score
surf = ax.plot_surface(xx, zz, yy, cmap='viridis', alpha=0.8)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot the original data points
ax.scatter(x_values, z_values, y, c='r', marker='o', label='Original Data')

# Find the coordinates of the maximum score on the surface
max_score_idx = np.unravel_index(np.argmax(yy, axis=None), yy.shape)
max_x = x_range[max_score_idx[1]]
max_z = z_range[max_score_idx[0]]
max_y = yy[max_score_idx]

# Plot indicator for the maximum score
ax.scatter(max_x, max_z, max_y, c='m', marker='*', s=100, label='Max Score')

# Set labels and title
ax.set_xlabel(x_param)
ax.set_ylabel(z_param)
ax.set_zlabel('Score')
ax.set_title('Hyperparameter Optimization with Fitted Curved Surface')

plt.legend()
plt.show()
