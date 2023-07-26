import numpy as np

def Iij(panel_i,panel_j):
    A = - (panel_i.midx - panel_j.xmin)*np.sin(panel_j.beta) + (panel_i.midy-panel_j.ymid)*np.cos(panel_j.beta)
    B  = (panel_i.midx - panel_j.xmin)**2 + (panel_i.ymid-panel_j.ymin)**2
    C = -1*np.cos(panel_i.beta)*np.sin(panel_j.beta)+np.sin(panel_i.beta)*np.cos(panel_j.beta)
    D = (panel_i.midy - panel_j.ymin)*np.sin(panel_i.beta) + (panel_i.midx - panel_j.xmin)*np.cos(panel_i.beta)
    E = (B-A**2)**0.5
    Sj = panel_j.len

    return C/2*np.log((Sj**2+2*A*Sj+B)/B)+(D-A*C)/E*(np.arctan((Sj+A)/E)-np.arctan(A/E))


def Jij(panel_i, panel_j):
    A = - (panel_i.midx - panel_j.xmin)*np.sin(panel_j.beta) + \
        (panel_i.midy-panel_j.ymid)*np.cos(panel_j.beta)
    B = (panel_i.midx - panel_j.xmin)**2 + (panel_i.ymid-panel_j.ymin)**2
    C = -1*np.cos(panel_i.beta)*np.sin(panel_j.beta) + \
        np.sin(panel_i.beta)*np.cos(panel_j.beta)
    D = (panel_i.midy - panel_j.ymin)*np.sin(panel_i.beta) + \
        (panel_i.midx - panel_j.xmin)*np.cos(panel_i.beta)
    E = (B-A**2)**0.5
    Sj = panel_j.len

    return (D-A*C)/(2*E)*np.log((Sj**2+2*A*Sj+B)/B)-C*(np.arctan((Sj+A)/E)-np.arctan(A/E))
