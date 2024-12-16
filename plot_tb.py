import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def plot_tensorboard_logs(log_dirs, output_dir):
    event_accs = [EventAccumulator(log_dir) for log_dir in log_dirs]
    for event_acc in event_accs:
        event_acc.Reload()

    # Extract scalar values
    try:
        train_loss_bagging = event_accs[0].Scalars('Train Epoch Loss')
        val_loss_bagging = event_accs[0].Scalars('Validation Loss')
    except:
        train_loss_bagging = event_accs[0].Scalars("Training Epoch Loss")
        val_loss_bagging = event_accs[0].Scalars("Validation Epoch Loss")
    val_accuracy_bagging = event_accs[0].Scalars('Validation Accuracy')

    try:
        train_loss_nonbagging = event_accs[1].Scalars('Train Epoch Loss')
        val_loss_nonbagging = event_accs[1].Scalars('Validation Loss')
    except:
        train_loss_nonbagging = event_accs[1].Scalars("Training Epoch Loss")
        val_loss_nonbagging = event_accs[1].Scalars("Validation Epoch Loss")
    val_accuracy_nonbagging = event_accs[1].Scalars('Validation Accuracy')

    # Prepare data for plotting
    train_loss_bagging_steps = [x.step for x in train_loss_bagging][:20]
    train_loss_bagging_values = [x.value for x in train_loss_bagging][:20]
    val_loss_bagging_steps = [x.step for x in val_loss_bagging][:20]
    val_loss_bagging_values = [x.value for x in val_loss_bagging][:20]
    val_accuracy_bagging_steps = [x.step for x in val_accuracy_bagging][:20]
    val_accuracy_bagging_values = [x.value for x in val_accuracy_bagging][:20]

    train_loss_nonbagging_steps = [x.step for x in train_loss_nonbagging][:20]
    train_loss_nonbagging_values = [x.value for x in train_loss_nonbagging][:20]
    val_loss_nonbagging_steps = [x.step for x in val_loss_nonbagging][:20]
    val_loss_nonbagging_values = [x.value for x in val_loss_nonbagging][:20]
    val_accuracy_nonbagging_steps = [x.step for x in val_accuracy_nonbagging][:20]
    val_accuracy_nonbagging_values = [x.value for x in val_accuracy_nonbagging][:20]

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_nonbagging_steps, train_loss_nonbagging_values, label='Train Loss (Non-Bagging)', linestyle='-')
    plt.plot(val_loss_nonbagging_steps, val_loss_nonbagging_values, label='Validation Loss (Non-Bagging)', linestyle='-')
    plt.plot(train_loss_bagging_steps, train_loss_bagging_values, label='Train Loss (Bagging)', linestyle='--')
    plt.plot(val_loss_bagging_steps, val_loss_bagging_values, label='Validation Loss (Bagging)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BERT Train and Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/loss_plot.png')
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracy_nonbagging_steps, val_accuracy_nonbagging_values, label='Validation Accuracy (Non-Bagging)', linestyle='-')
    plt.plot(val_accuracy_bagging_steps, val_accuracy_bagging_values, label='Validation Accuracy (Bagging)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("BERT Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy_plot.png')
    plt.close()

if __name__ == "__main__":
    log_dirs = [
        "runs/bert_5bagging_training_20241215024709",  # Update with your bagging log directory
        "runs/bert_training_20241215015154",  # Update with your non-bagging log directory
    ]
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_tensorboard_logs(log_dirs, output_dir)
