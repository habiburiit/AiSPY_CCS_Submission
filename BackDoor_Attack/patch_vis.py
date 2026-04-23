import math, torch
import matplotlib.pyplot as plt

def make_denorm(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
    mean = torch.tensor(mean).view(1,3,1,1)
    std  = torch.tensor(std).view(1,3,1,1)
    def _denorm(x):
        return (x * std) + mean
    return _denorm

@torch.no_grad()
def show_batch(dl, n=16, title="Samples", denorm=None, figsize=(8,8), save_path=None):
    """
    dl: DataLoader of (x, y_binary, meta), where y_binary=1 if triggered.
    n:  number of images to show (grid)
    denorm: function that inverts normalization; pass make_denorm(...)
    """
    xb, yb, meta = next(iter(dl))
    xb = xb[:n].clone()
    yb = yb[:n].numpy()
    if denorm is not None:
        xb = denorm(xb)
    xb = xb.clamp(0,1)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=figsize)
    for i in range(n):
        img = xb[i].permute(1,2,0).cpu().numpy()
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(img)
        trig = int(yb[i])
        ax.set_title(f"trigger={trig}", fontsize=9)
        ax.axis("off")
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    # plt.show()
