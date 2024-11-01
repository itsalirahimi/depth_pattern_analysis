import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = [f'Item {i+1}' for i in range(8)]
set1 = np.random.randint(1, 10, size=8)  # Random integers for the first set
set2 = np.random.randint(1, 10, size=8)  # Random integers for the second set

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, set1, width, label='Set 1', color='red')
bars2 = ax.bar(x + width/2, set2, width, label='Set 2', color='blue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of Two Sets of Values')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Show the plot
plt.show()
