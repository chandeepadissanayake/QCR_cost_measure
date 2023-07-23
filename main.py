import numpy as np
from itertools import chain, combinations
import networkx as nx
import matplotlib.pyplot as plt

alpha = 0.25
beta = 1.5

# G = np.array([
#     [0, 1, 0, 1, 1, 1],
#     [0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0]
# ])

G = np.array([
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])


def V(G):
    return np.arange(stop=G.shape[0], dtype=np.int32)


def E(G):
    rows, cols = np.where(G == 1)
    return [(rows[i], cols[i]) for i in range(len(rows))]


def P_V(G):
    _V = V(G)
    _V = _V.tolist()

    _P_V = chain.from_iterable(combinations(_V, r) for r in range(len(_V) + 1))
    return list(_P_V)


def C(G, Vi, alpha, beta):
    d0 = 0
    for va in range(G.shape[0]):
        for vb in range(G.shape[1]):
            if va not in Vi and vb not in Vi:
                d0 += np.count_nonzero(G[va, vb] == 1)

    d1 = 0
    for vj in Vi:
        for vb in range(G.shape[1]):
            if vb not in Vi:
                d1 += np.count_nonzero(G[vj, vb] == 1) + \
                    np.count_nonzero(G[vb, vj] == 1)

    return beta * len(Vi) + 1 * d0 + alpha * d1


def visualize_graph(G):
    _E = E(G)

    g = nx.Graph()
    g.add_edges_from(_E)

    nx.draw(g, with_labels=True)
    plt.show()


def main():
    visualize_graph(G)

    _P_V = P_V(G)

    _C = []
    for Vi in _P_V:
        _c = C(G, Vi, alpha, beta)
        _C.append(_c)
        print("%s\t\t: %f" % (Vi, _c))

    print()
    _min_c = min(_C)
    min_idx = _C.index(_min_c)
    print("Minimum Vertex Cover: %s\nVertex Covering Number = %d\nCost = %f" %
          (_P_V[min_idx], len(_P_V[min_idx]), _min_c))


if __name__ == '__main__':
    main()
