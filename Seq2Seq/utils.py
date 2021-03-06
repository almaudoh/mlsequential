import matplotlib.pyplot as plt


def plot_grad_flow(grad_flow_lines, figure=2):
    plt.figure(figure)
    for grad_lines in grad_flow_lines:
        grads = grad_lines['grads']
        layers = grad_lines['layers']
        plt.plot(grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)


def plot_parameters(parameters, figure=3):
    ave_grads = []
    layers = []
    for n in parameters:
        layers.append(n)
        ave_grads.append(parameters[n].mean())
    plt.figure(figure)
    plt.plot(ave_grads, alpha=0.3, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
