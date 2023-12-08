import matplotlib.pyplot as plt
import numpy as np

# Function to read loss data from a file
def read_loss_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    return data[:, 0], data[:, 1]  # epochs, loss

# Read training and validation loss data from files
epochs_train, loss_train = read_loss_data('train_loss.txt')
epochs_val, loss_val = read_loss_data('val_loss.txt')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, loss_train, label='Training Loss')
plt.plot(epochs_val, loss_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('loss_plot.png')

# If you still want to display the plot in the script, uncomment the next line
# plt.show()
