import sys
import numpy as np
import matplotlib.pyplot as plt
from HMSM.plots.plots import integrate_2d
from HMSM.tests.force_functions.two_dimensional import neg
import graph_tool.all as gt
import graph_tool.draw as draw

def in_notebook():
    return 'ipykernel' in sys.modules

def int_to_rgb(num, string=False):
    c = np.array([(num%1103)%256, (num%1367)%256, (num%1447)%256, 256])/256
    c = list(c)
    if string:
        c = "#%2x%2x%2x" % c[:-1]*256
    return c

def get_state_pos(tree, state, cg):
    microstates = tree.get_microstates(state)
    centers = cg.get_centers_by_ids(microstates)
    return np.mean(centers, axis=0)

def int_to_rgb(num, string=False):
    c = np.array([(num%1103)%256, (num%1367)%256, (num%1447)%256, 256])/256
    c = list(c)
    if string:
        c = "#%2x%2x%2x" % c[:-1]*256
    return c

def plot_level_graph(tree, level, force, cg, ax=None, loc_2d=True, contour=None):
    #TODO clean up and document this mess of a function
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
    node_ids = tree.get_level(level)
    n = len(node_ids)
    T,_= tree.get_level_T(level, 1)
    st = np.linalg.matrix_power(T, 4096)[0]
    sizes = 10 + st * 50/max(st)
    g = gt.Graph(directed=True)

    g.add_vertex(n+2)
    vprop_pi = g.new_vertex_property("double")
    eprop_tp = g.new_edge_property("double")
    vprop_parent = g.new_vertex_property("vector<double>")
    vprop_color = g.new_vertex_property("vector<double>")
    if loc_2d: 
        vprop_pos = g.new_vertex_property("vector<double>")
        xlim = ylim = 0
    for i in range(n):
        v = g.vertex(i)
        vprop_pi[v] = st[i]#sizes[i]
        if loc_2d:
            pos = get_state_pos(tree, node_ids[i], cg)
            vprop_pos[v] = pos
            xlim = max(xlim, np.abs(pos[0])+3)
            ylim = max(ylim, np.abs(pos[1])+3)
        vprop_parent[v] = int_to_rgb(tree.get_parent(node_ids[i]))
        vprop_color[v] = int_to_rgb(node_ids[i])
        for j in range(n):
            if j!=i and T[i,j]>0:
                e = g.add_edge(v, g.vertex(j))
                eprop_tp[e] = T[i,j]
                
    if loc_2d:
        xlim = (-xlim, xlim)
        ylim = (-ylim, ylim)
        if contour is not None:
            xlim = contour[0]
            ylim = contour[0]
            X, Y, I = contour[1] 
            ax.contour(X,Y,I, alpha=0.4, levels=70, zorder=0, cmap='gray')
            ax.contourf(X,Y,I, alpha=0.4, levels=70, zorder=0, cmap='viridis')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
        v = g.vertex(n)
        vprop_pi[v] = 0
        vprop_pos[v] = (xlim[1]-2, ylim[1]-2)
        v = g.vertex(n+1)
        vprop_pi[v] = 0
        vprop_pos[v] = (xlim[0]+2, ylim[0]+2)
    else:
        xlim = None
    
    vprop_size = draw.prop_to_size(vprop_pi, mi=0.15, ma=0.8, log=False)
    vprop_pen_size = draw.prop_to_size(vprop_size, mi=0.02, ma=0.1)
    eprop_size = draw.prop_to_size(eprop_tp, mi=0.0001, ma=0.08, log=False)
    gt.graph_draw(g, vertex_size=vprop_size, edge_pen_width=eprop_size, pos=vprop_pos, mplfig=ax,
                  vertex_pen_width=vprop_pen_size, vertex_fill_color=vprop_parent, 
                  vertex_color=vprop_color, zorder=1, xlim=xlim)


def plot_states(cg, microstate_ids, ax, label=None, c=None):
    points = []
    for microstate in microstate_ids:
        points.append(cg.sample_from(microstate))
    points = np.array(points)
    ax.scatter(*points.T, label=label, s=215, alpha=0.3, color=c, zorder=0.)

def plot_vertex(cg, vertex, tree, ax):
    points = tree.get_microstates(vertex)
    plot_states(cg, points, ax, c=int_to_rgb(vertex))
    
def plot_level(cg, tree, level, ax, **kwargs):
    count = 0
    for vertex_id, vertex in tree.vertices.items():
        if vertex.height == level:
            count += 1
            plot_vertex(cg, vertex_id, tree, ax)
    ax.set_title(f"Level {level}, number of clusters: {count}")
    ax.axis('square')
    
def plot_tree(cg, tree, force, lim=7):
    #TODO: generalize to when force is not available, or not 2D
    if in_notebook():
        # in ipynb it's tricky to get inline working after switching backend...
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic("%matplotlib inline ")
        plt.plot()
        plt.switch_backend("cairo")
    elif plt.get_backend() != "cairo":
        raise Exception("set pyplot backend to cairo to use plot_tree")

    h = tree.height
    if h==1:
        fig, ax = plt.subplots(1,1, figsize=(10,10*(h)))
        plot_level(cg, tree, 1, ax)
        return
    fig, ax = plt.subplots(h-1,1, figsize=(10,10*(h)))
    xlim = (-lim,lim)
    ylim = (-lim,lim)
    heatmap = integrate_2d(neg(force), xlim, ylim, 0.1)
    contour = (xlim, heatmap)
    for i in range(1,h):
        plot_level(cg, tree, h-i, ax[i-1])
        ax[i-1].use_sticky_edges = False
        ax[i-1].set_xlim(xlim)
        ax[i-1].set_ylim(ylim)
        plot_level_graph(tree, h-i, force, cg, ax=ax[i-1], contour=contour)
