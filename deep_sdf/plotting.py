import matplotlib.pyplot as plt


def plot_train_stats(loss_hists: list, psnr_hist=None, step_hist=None, save_path="") -> plt.figure:
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    if not step_hist:
        step_hist = list(range(len(loss_hist)))

    fig.suptitle(f"Training curves {save_path}")
    for loss_hist in loss_hists:
        ax.plot(step_hist, loss_hist, c="orange", label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    if psnr_hist:
        ax2 = ax[0].twinx()
        ax2.plot(step_hist, psnr_hist, c="g", label="PSNR")
        ax2.set_ylabel("PSNR")

    if save_path:
        fig.savefig(f"{save_path}.jpg", dpi=300, bbox_inches='tight')

    return fig
