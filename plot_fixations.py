import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the results CSV file
df = pd.read_csv('results.csv')

# Create the output folder if it doesn't exist
output_folder = 'fixations'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to plot the fixation paths
def plot_fixation_paths(df, set_id, output_folder):
    subset = df[df['set_id'] == set_id]
    
    # Determine the color based on touch status
    colors = subset.apply(lambda row: 'red' if row['touched'] == 0 else ('blue' if row['touch_time'] == row['touch_time'] else 'green'), axis=1)
    
    # Plot the points
    plt.figure(figsize=(8, 8))
    plt.scatter(subset['x'], subset['y'], c=colors)
    
    # Plot the paths
    touched_points = subset[subset['touched'] == 1].sort_values('touch_time')
    plt.plot(touched_points['x'], touched_points['y'], 'bo-', alpha=0.5)
    
    # Title and labels
    plt.title(f'Set {set_id} Fixation Path')

    plt.xlim(-0.15, 1.15)
    plt.ylim(-0.15, 1.15)
    plt.grid(False)
    
    # Save the plot
    plt.savefig(os.path.join(output_folder, f'{set_id}_fixation_path.png'))
    plt.close()

# Plot and save fixation paths for each set
for set_id in df['set_id'].unique():
    plot_fixation_paths(df, set_id, output_folder)