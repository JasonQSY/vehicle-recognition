import numpy as np
import os
import imageio
from tqdm import tqdm

def ensemble(paths):
    """
    Ensemble the surface normal in several models
    """
    num_models = len(paths)
    for idx in tqdm(range(2000)):
        imgs = []
        for path in paths:
            img = imageio.imread(os.path.join(path, "{}.png".format(idx)))
            imgs.append(img / 255)
        output = sum(imgs) / num_models
        output = output * 255
        imageio.imwrite("./save/{}.png".format(idx), output.astype(np.uint8))

if __name__ == "__main__":
    paths = ['./save1', './save2', './save3']
    ensemble(paths)
