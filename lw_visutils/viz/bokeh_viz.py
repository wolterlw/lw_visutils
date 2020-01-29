from IPython.display import display
from ipywidgets import IntSlider, Layout
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
from cv2 import cvtColor, flip, COLOR_RGB2RGBA
import numpy as np

def _process_frame(frame):
    frame = cvtColor(frame, COLOR_RGB2RGBA)
    return np.ascontiguousarray(frame[::-1])

def sequence_viz(frames, scale=0.5):
    """Make a visualization of seqence of RGB frames"""

    height, width,_= frames[0].shape
    frame = _process_frame(frames[0])
    
    p = figure(x_range=(0,width), y_range=(0,height),
            output_backend="webgl", 
            width=int(width*scale), height=int(height*scale))
    p.axis.visible=False
    
    myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
    output_notebook()
    show(p, notebook_handle=True)
    
    def event(change):
        idx = change['new']
        frame = _process_frame(frames[idx])
        myImage.data_source.data['image']=[frame]
        push_notebook()

    slider = IntSlider(max=len(frames)-1)
    slider.observe(event, names='value')
    return slider

# def cont_viz(frame0, scale=0.5):
#     """Make a visualization of stream of RGB frames"""

#     height, width,_= frame0.shape
#     frame = _process_frame(frame0)
    
#     p = figure(x_range=(0,width), y_range=(0,height),
#             output_backend="webgl", 
#             width=int(width*scale), height=int(height*scale))
#     p.axis.visible=False
    
#     myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
#     output_notebook()
#     show(p, notebook_handle=True)
    
#     def new_frame(frame):
#         img = _process_frame(frame)
#         myImage.data_source.data['image']=[img]
#         push_notebook()
        
#     return new_frame