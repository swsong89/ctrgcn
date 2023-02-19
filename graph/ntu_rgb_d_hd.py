from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
# from graph import tools # s如果需要生成效果图的话需要把这个注释调

num_node = 25

class Graph:
    def __init__(self, CoM=1, labeling_mode='spatial'):
        self.num_node = num_node  # 25
        self.CoM = CoM  # 21
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_hierarchical_graph(num_node, tools.get_edgeset(dataset='NTU', CoM=self.CoM)) # L, 3, 25, 25
        else:
            raise ValueError()
        return A, self.CoM  # (6, 3, 25, 25) 21


if __name__ == '__main__':
    import tools
    com = 1
    g = Graph(CoM=com).A
    print(len(g))  # array A Com 1
    print(g[0][0])
    # import matplotlib.pyplot as plt
    # for i, g_ in enumerate(g[0]):
    #     plt.imshow(np.sum(g_,0), cmap='gray')
    #     cb = plt.colorbar()
    #     plt.savefig('graph/graph_com_{}_{}.png'.format(com, i))
    #     cb.remove()
    #     plt.show()

    # plt.imshow(np.sum(np.sum(g[0],0),0), cmap='gray')
    # cb = plt.colorbar()
    # plt.savefig('graph/graph_com_{}_all.png'.format(com))
    # cb.remove()
    # plt.show()