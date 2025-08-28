from __future__ import annotations 
import numpy as np 


# We do U_shape_curve due to the market will be like U-shape as high trading activity at open and
# Low trading activity midday, High trading again near close the market
# We spilt one day as n_slices, and calculate the weights of trading for each slices
# We define this u_shape_curve as simulate the market which is the u_shape.
def u_shape_curve(n_slices: int) -> np.ndarray: 
    xs = np.linspace(0.0, 1.0, n_slices) 
    w = 0.5 + 2.0 * (xs - 0.5) ** 2 
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


