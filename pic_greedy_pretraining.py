# the used method (autoencoders) for pretraining can also be methods such as e.g. Restricted Boltzmann Machines (RBMs), Denoising Autoencoders (DAE), 
# Contractive Autoencoders (CAE), Variational Autoencoders (VAEs), Contrastive Predictive Coding (CPC) or Self-Supervised Learning (Modern Alternative)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors for layers
input_color = '#A2D2FF'
hidden_color = '#B9FBC0'
encoder_color = '#FDE2E4'
fine_tune_color = '#FFB703'

# Draw input layer
ax.add_patch(mpatches.Rectangle((1, 8), 2, 1, color=input_color, label='Input Data (X)'))
ax.text(2, 8.5, 'Input Data (X)', ha='center', va='center', fontsize=12)

# Draw layer 1 (encoder 1)
ax.add_patch(mpatches.Rectangle((1, 6), 2, 1, color=encoder_color))
ax.text(2, 6.5, 'Train Encoder 1\n(X -> H1)', ha='center', va='center', fontsize=10)

# Draw hidden layer 1
ax.add_patch(mpatches.Rectangle((4, 6), 2, 1, color=hidden_color))
ax.text(5, 6.5, 'Hidden Layer 1\n(H1)', ha='center', va='center', fontsize=10)

# Arrow from input to layer 1
ax.annotate('', xy=(2, 7.95), xytext=(2, 7.05), arrowprops=dict(arrowstyle='->', lw=2))

# Arrow from encoder 1 to hidden layer 1
ax.annotate('', xy=(2, 6.95), xytext=(4, 6.95), arrowprops=dict(arrowstyle='->', lw=2))

# Draw layer 2 (encoder 2)
ax.add_patch(mpatches.Rectangle((1, 4), 2, 1, color=encoder_color))
ax.text(2, 4.5, 'Train Encoder 2\n(H1 -> H2)', ha='center', va='center', fontsize=10)

# Draw hidden layer 2
ax.add_patch(mpatches.Rectangle((4, 4), 2, 1, color=hidden_color))
ax.text(5, 4.5, 'Hidden Layer 2\n(H2)', ha='center', va='center', fontsize=10)

# Arrow from hidden layer 1 to layer 2
ax.annotate('', xy=(5, 5.95), xytext=(2, 5.05), arrowprops=dict(arrowstyle='->', lw=2))

# Arrow from encoder 2 to hidden layer 2
ax.annotate('', xy=(2, 4.95), xytext=(4, 4.95), arrowprops=dict(arrowstyle='->', lw=2))

# Draw layer 3 (encoder 3)
ax.add_patch(mpatches.Rectangle((1, 2), 2, 1, color=encoder_color))
ax.text(2, 2.5, 'Train Encoder 3\n(H2 -> H3)', ha='center', va='center', fontsize=10)

# Draw hidden layer 3
ax.add_patch(mpatches.Rectangle((4, 2), 2, 1, color=hidden_color))
ax.text(5, 2.5, 'Hidden Layer 3\n(H3)', ha='center', va='center', fontsize=10)

# Arrow from hidden layer 2 to layer 3
ax.annotate('', xy=(5, 3.95), xytext=(2, 3.05), arrowprops=dict(arrowstyle='->', lw=2))

# Arrow from encoder 3 to hidden layer 3
ax.annotate('', xy=(2, 2.95), xytext=(4, 2.95), arrowprops=dict(arrowstyle='->', lw=2))

# Draw fine-tuning step
ax.add_patch(mpatches.Rectangle((7, 4), 2, 4, color=fine_tune_color))
ax.text(8, 6, 'Supervised\nFine-Tuning\n(Backprop)', ha='center', va='center', fontsize=12)

# Arrow from hidden layer 1 to fine-tuning
ax.annotate('', xy=(6, 6.95), xytext=(7, 6.95), arrowprops=dict(arrowstyle='->', lw=2))

# Arrow from hidden layer 2 to fine-tuning
ax.annotate('', xy=(6, 4.95), xytext=(7, 4.95), arrowprops=dict(arrowstyle='->', lw=2))

# Arrow from hidden layer 3 to fine-tuning
ax.annotate('', xy=(6, 2.95), xytext=(7, 2.95), arrowprops=dict(arrowstyle='->', lw=2))

# Legend for color meanings
legend_handles = [
    mpatches.Patch(color=input_color, label='Input Data (X)'),
    mpatches.Patch(color=encoder_color, label='Unsupervised Encoder (Autoencoder or RBM)'),
    mpatches.Patch(color=hidden_color, label='Hidden Layer (H1, H2, H3)'),
    mpatches.Patch(color=fine_tune_color, label='Supervised Fine-Tuning')
]
ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

plt.show()