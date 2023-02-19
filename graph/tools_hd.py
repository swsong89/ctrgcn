import numpy as np

def edge2mat(link, num_node):  # link里面都是每组edge, 25
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):  # edge [(20, 1), (20, 2), (20, 4), (20, 8)]
    Dl = np.sum(A, 0)  # 按列统计 在20的位置是4 dl[20] = 4 A[1,20] A[2,20], A[4,20] A[8,20]
    h, w = A.shape  # 25,25
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)  # Dn[1,20] = 0.25
    AD = np.dot(A, Dn)
    return AD  # 本来标准化是标准化出度，一行应该是出度，但是后面需要T,所以变成一列是出度，

def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A

def get_spatial_graph_original(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def get_graph(num_node, edges):  # V25

    I = edge2mat(edges[0], num_node)  # edges[0]节点本身 本来矩阵A每一行是该节点到其余节点的方向，是出度，因为需要T,所以直接生成的是每一列是出度的方向,即每一行是其余节点对该节点的贡献，每一列是该节点对其余节点的贡献，
    Forward = normalize_digraph(edge2mat(edges[1], num_node))  # 离心 入度方向  节点集是根节点出度方向但是矩阵是入度方向[(20, 1), (20, 2), (20, 4), (20, 8)] Forward[1,20]=0.25 TODO边感觉是从根节点向外，但是Forward计算的好像是入度，即从外向内
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))  # 向心  [(1, 20), (2, 20), (4, 20), (8, 20)]
    A = np.stack((I, Forward, Reverse))  # 本身，出度，入度
    return A # 3, 25, 25 feature * A*T [4, 16, 64, 25] A[25,25] 按行相乘 'n c t u, v u -> n c t v' 每一列是1节点，A本来是每一列是出度的权重，T之后每一行是出度的权重需要标准化,每一列是入度的权重,
        # feature * A 一行一行相乘, A每一行都是其余节点对该节点的贡献,入度,每一列是该节点对其余节点的贡献，出度
def get_hierarchical_graph(num_node, edges):  # # 25,   6[[[],[],[]]] 6个节点集， 每个节点集先是节点本身特征，再是离心，再是向心
    A = []
    for edge in edges:  # 6组
        A.append(get_graph(num_node, edge))
    A = np.stack(A)  # (6, 3, 25, 25) 6组节点集，每组分别是节点本身，入度，
    return A  # A[0,1,1,20] = 0.25 应该是20->1但结果是1->20 那应该行是tartget,列是source [1,20] 是20->1

def get_groups(dataset='NTU', CoM=21):
    groups  =[]
    
    if dataset == 'NTU':
        if CoM == 2:
            groups.append([2])
            groups.append([1, 21])
            groups.append([13, 17, 3, 5, 9])
            groups.append([14, 18, 4, 6, 10])
            groups.append([15, 19, 7, 11])
            groups.append([16, 20, 8, 12])
            groups.append([22, 23, 24, 25])

        ## Center of mass : 21
        elif CoM == 21:  # 节点从根节点开始分了7个等级，7个等级组成6个组
            groups.append([21])
            groups.append([2, 3, 5, 9])
            groups.append([4, 6, 10, 1])
            groups.append([7, 11, 13, 17])
            groups.append([8, 12, 14, 18])
            groups.append([22, 23, 24, 25, 15, 19])
            groups.append([16, 20])

        ## Center of Mass : 1
        elif CoM == 1:
            groups.append([1])
            groups.append([2, 13, 17])
            groups.append([14, 18, 21])
            groups.append([3, 5, 9, 15, 19])
            groups.append([4, 6, 10, 16, 20])
            groups.append([7, 11])
            groups.append([8, 12, 22, 23, 24, 25])

        else:
            raise ValueError()
        
    return groups

def get_edgeset(dataset='NTU', CoM=21):
    groups = get_groups(dataset=dataset, CoM=CoM)  # 7个groups,节点分成7个等级，然后变成6组
    
    for i, group in enumerate(groups):  # 实际作用是将从1到25节点改成0-24
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []
#  0:[20] 1:[1, 2, 4, 8] 2:[3, 5, 9, 0] 3: [6, 10, 12, 16] 4:[7, 11, 13, 17] 5:[21, 22, 23, 24, 14, 18] 6: [15, 19]
    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)  # 根节点和与根节点相连的节点本身  [(20, 20), (1, 1), (2, 2), (4, 4), (8, 8)]
        forward_g = []
        for j in groups[i]:  # 根节点
            for k in groups[i + 1]:  # 与根节点相连的节点
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)  # [(20, 1), (20, 2), (20, 4), (20, 8)] 从根节点向外， 出度
        
        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)  # 从外向根节点，入度[(15, 21), (15, 22), (15, 23), (15, 24), (15, 14), (15, 18), (19, 21), (19, 22), (19, 23), (19, 24), (19, 14), (19, 18)]

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges  # 6[[[],[],[]]] 6个节点集， 每个节点集先是节点本身特征，再是离心，再是向心
    print('edge: ', len(edges))

    ''''
    
    0:
[20]
1:
[1, 2, 4, 8]
2:
[4, 6, 10, 1]
3:
[7, 11, 13, 17]
4:
[8, 12, 14, 18]
5:
[22, 23, 24, 25, 15, 19]
6:
[16, 20]

0:
[20]
1:
[1, 2, 4, 8]
2:
[3, 5, 9, 0]
3:
[7, 11, 13, 17]
4:
[8, 12, 14, 18]
5:
[22, 23, 24, 25, 15, 19]
6:
[16, 20]

0:
[20]
1:
[1, 2, 4, 8]
2:
[3, 5, 9, 0]
3:
[6, 10, 12, 16]
4:
[7, 11, 13, 17]
5:
[21, 22, 23, 24, 14, 18]
6:
[15, 19]
    '''