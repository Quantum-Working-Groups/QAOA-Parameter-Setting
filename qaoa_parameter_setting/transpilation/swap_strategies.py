"""Create swap strategies."""

import numpy as np

from qiskit.transpiler import CouplingMap

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy


def make_2d_grid_swap_strategy(rows, cols):
    """Make a swap strategy for a 2D grid."""

    # Make the S0 and S1 layers on the rows. First define them on a line
    s0 = tuple((idx, idx + 1) for idx in range(0, cols - 1, 2))
    s1 = tuple((idx, idx + 1) for idx in range(1, cols - 1, 2))

    # Next, extend s0 and s1 to be applied to the other rows.
    grid_s0, grid_s1 = [], []
    for row in range(rows):
        if row % 2 == 0:
            grid_s0 += [(swap[0] + row * cols, swap[1] + row * cols) for swap in s0]
            grid_s1 += [(swap[0] + row * cols, swap[1] + row * cols) for swap in s1]
        else:
            grid_s0 += [(swap[0] + row * cols, swap[1] + row * cols) for swap in s1]
            grid_s1 += [(swap[0] + row * cols, swap[1] + row * cols) for swap in s0]
    
    # Now we make the S2 and S3 layers on the columns
    grid_s2, grid_s3 = [], []
    for col in range(cols):
        for row in range(rows-1):
            if row % 2 == 0:
                grid_s2.append((row * cols + col, (row + 1) * cols + col))
            else:
                grid_s3.append((row * cols + col, (row + 1) * cols + col))

    # Finally we build the full swap strategy.
    swap_strat = []
    for ridx in range(int(np.ceil(rows/2))):
        for cidx in range(cols-1):
            if cidx % 2 == 0:
                swap_strat.append(tuple(grid_s0))
            else:
                swap_strat.append(tuple(grid_s1))

        swap_strat.append(grid_s2)
        swap_strat.append(grid_s3)

    cmap = CouplingMap.from_grid(rows, cols)
        
    return SwapStrategy(cmap, swap_strat)
