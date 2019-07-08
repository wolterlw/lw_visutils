import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import h5py


class Hand():
    """
    Helper class used to visualize hand joints and generate gaussian heatmaps
    """
    colormap = {
            'wrist': 'w',
            'thumb': '#ffc857',
            'index': '#e9724c',
            'middle': '#c5283d',
            'ring': '#00fddc',
            'pinky': '#255f85',
        }

    def __init__(self, joint_array):
        self.fully_visible = (joint_array > 0).all()
        self.array = np.clip(joint_array[:,::-1],0,255)
        self.joints = {
            'wrist': self.array[:1],
            'thumb': self.array[1:5],
            'index': self.array[5:9],
            'middle': self.array[9:13],
            'ring': self.array[13:17],
            'pinky': self.array[17:21],
        }

    def draw(self, axis):
        for k in self.joints.keys():
            axis.plot(
                self.joints[k][:,0],
                self.joints[k][:,1],
                c=self.colormap[k], linewidth=3)
            axis.plot([self.joints['wrist'][0,0],self.joints[k][-1,0]],
                     [self.joints['wrist'][0,1],self.joints[k][-1,1]],
                     c=self.colormap[k])

def visualize(frame, coords):
    fig,ax = plt.subplots(1,1)
    fig.patch.set_visible(False)
    ax.imshow(cv2.resize(frame, (256,256)))
    hand = Hand(coords)
    hand.draw(ax)
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return data


def write_video(frames, coords, filename):
    coords = (coords*256).astype('uint')
    writer = imageio.get_writer(filename, fps=30)
    for frame,crds in zip(frames, coords):
        res = visualize(frame, crds)
        writer.append_data(res)
    writer.close()