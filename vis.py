import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from method import *


def fosc_cal(inputs, pert_inputs, teacher_outputs, targets, epsilon):
    grad = fosc_deps(pert_inputs, teacher_outputs, targets)
    alpha_fosc = AlphaFOSC.cal(epsilon, grad, pert_inputs, inputs)
    return alpha_fosc


def visualize_metric_vs_prob(adv_student_model, basic_teacher_model, testloader, device, metric_fn, epsilon, vis_path):
    """
    Visualize the relation between a metric and probability of the target class for each sample in a test loader.
    
    Args:
        model (nn.Module): PyTorch model.
        testloader (DataLoader): PyTorch DataLoader containing the test data.
        device (str): Device to run the model on.
        metric_fn (function): Metric function that takes in the ground truth labels and predicted labels
                              and returns a 1d tensor.
    
    Returns:
        None
    """
    
    # Create empty lists to store the probabilities and metrics
    probs = []
    metrics = []
    
    # Iterate over the test loader
    for i, (images, labels) in enumerate(tqdm(testloader, ncols=0)):
        # Move the data to the specified device
        images, labels = images.to(device), labels.to(device)
        
        # Compute the probabilities for the current batch
        with torch.enable_grad():
            _, pert_inputs = adv_student_model(images, labels)
            teacher_outputs = basic_teacher_model(pert_inputs)
            probs_batch = torch.softmax(teacher_outputs, dim=1)[torch.arange(images.shape[0]), labels].detach()
        
        # Compute the metric for the current batch
        metric_batch = metric_fn(images, pert_inputs, teacher_outputs, labels, epsilon).detach()
        
        # Append the probabilities and metrics to the lists
        probs.append(probs_batch.cpu().numpy())
        metrics.append(metric_batch.cpu().numpy())

        if i > 10:
            break
    
    # Concatenate the lists along the batch dimension
    probs = np.concatenate(probs)
    metrics = np.concatenate(metrics)
    corr = np.corrcoef(probs.flatten(), metrics.flatten())[0, 1]
    
    # Plot the relation between the metric and probability
    plt.scatter(probs, metrics, s=6)
    plt.xlabel("Probability of target class")
    plt.ylabel("FOSC Metric")
    plt.yscale('log')
    plt.title('Correlation {:.4f}'.format(float(corr)))
    plt.savefig(vis_path, dpi=300)
    print(vis_path)
