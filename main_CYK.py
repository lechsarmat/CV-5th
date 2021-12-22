import numpy as np
import imageio as im
import func_CYK as ck

Z = (np.array(im.imread("data/standards/zero.png")) / 255).astype(int)
F = (np.array(im.imread("data/standards/one.png")) / 255).astype(int)
E = (np.array(im.imread("data/examples/example_3.png")) / 255).astype(int)

ck.AnswerFilter( ck.CYK(ck.DataConverter(Z, F, E) ) )