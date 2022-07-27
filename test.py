# -*- coding: utf-8 -*-
# @Filename: test
# @Date: 2022-07-26 10:19
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import dgl
import dgl.function as fn
import torch as th
g = ... # create a DGLGraph

# case 1
g.ndata['h'] = th.randn((g.num_nodes(), 10))
g.edata['w'] = th.randn((g.num_edges(), 1))
# OK, valid broadcasting between feature shapes (10,) and (1,)
g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
g.ndata['h_new']  # shape: (g.num_nodes(), 10)

# case 2
g.ndata['h'] = th.randn((g.num_nodes(), 5, 10))
g.edata['w'] = th.randn((g.num_edges(), 10))
# OK, valid broadcasting between feature shapes (5, 10) and (10,)
g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
g.ndata['h_new']  # shape: (g.num_nodes(), 5, 10)

# case 3
g.ndata['h'] = th.randn((g.num_nodes(), 5, 10))
g.edata['w'] = th.randn((g.num_edges(), 5))
# NOT OK, invalid broadcasting between feature shapes (5, 10) and (5,)
# shapes are aligned from right
g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))

# case 3
g.ndata['h1'] = th.randn((g.num_nodes(), 1, 10))
g.ndata['h2'] = th.randn((g.num_nodes(), 5, 1))
# OK, valid broadcasting between feature shapes (1, 10) and (5, 1)
g.apply_edges(fn.u_add_v('h1', 'h2', 'x'))  # apply_edges also supports broadcasting
g.edata['x']  # shape: (g.num_edges(), 5, 10)

# case 4
g.ndata['h1'] = th.randn((g.num_nodes(), 1, 10, 128))
g.ndata['h2'] = th.randn((g.num_nodes(), 5, 1, 128))
# OK, u_dot_v supports broadcasting but requires the last dimension to match
g.apply_edges(fn.u_dot_v('h1', 'h2', 'x'))
g.edata['x']  # shape: (g.num_edges(), 5, 10, 1)