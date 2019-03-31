
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import randint

G = nx.Graph()
N = 1500
degree = 4
X = 2**degree * 20; Y = 2**degree * 20
pos = {}
CH = nx.Graph()
CHx = X/20; CHy = Y/20
CHn = CHx*CHy
posCH = {}
Clusters = {}

def h1(start, degree):
    if degree == 1:
        CH.add_edge(start,start+1)
        CH.add_edge(start+1,start+CHy+1)
        CH.add_edge(start+CHy+1,start+CHy)
        return start+CHy
    else:
        s1 = h2(start, degree-1)
        CH.add_edge(s1,s1+1)
        s2 = h1(s1+1, degree-1)
        CH.add_edge(s2,s2+CHy)
        s3 = h1(s2+CHy, degree-1)
        CH.add_edge(s3,s3-1)
        return h4(s3-1, degree-1)

def h2(start, degree):
    if degree == 1:
        CH.add_edge(start,start+CHy)
        CH.add_edge(start+CHy,start+CHy+1)
        CH.add_edge(start+CHy+1,start+1)
        return start+1
    else:
        s1 = h1(start, degree -1)
        CH.add_edge(s1,s1+CHy)
        s2 = h2(s1+CHy, degree-1)
        CH.add_edge(s2,s2+1)
        s3 = h2(s2+1, degree-1)
        CH.add_edge(s3,s3-CHy)
        return h3(s3-CHy, degree-1)

def h3(start, degree):
    if degree == 1:
        CH.add_edge(start,start-1)
        CH.add_edge(start-1,start-CHy-1)
        CH.add_edge(start-CHy-1,start-CHy)
        return start-CHy
    else:
        s1 = h4(start, degree-1)
        CH.add_edge(s1,s1-1)
        s2 = h3(s1-1, degree-1)
        CH.add_edge(s2,s2-CHy)
        s3 = h3(s2-CHy, degree-1)
        CH.add_edge(s3,s3+1)
        return h2(s3+1, degree-1)

def h4(start, degree):
    if degree == 1:
        CH.add_edge(start,start-CHy)
        CH.add_edge(start-CHy,start-CHy-1)
        CH.add_edge(start-CHy-1,start-1)
        return start-1
    else:
        s1 = h3(start, degree-1)
        CH.add_edge(s1,s1-CHy)
        s2 = h4(s1-CHy, degree-1)
        CH.add_edge(s2,s2-1)
        s3 = h4(s2-1, degree-1)
        CH.add_edge(s3,s3+CHy)
        return h1(s3+CHy, degree-1)

def create_WSN():
    for i in range(N):
        G.add_node(i)
        pos.update({i:(randint(0,X-1),randint(0,Y-1))})
    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=5)
    plt.show()

def create_CH():
    for i in range(int(CHn)):
        CH.add_node(i+1)
    CHpos = []
    for x in range(10,X+1,20):
        for y in range(10,Y+1,20):
            CHpos.append((x,y))
    posCH.update({i+1:CHpos[i] for i in range(int(CHn))})
    nx.draw_networkx_nodes(G, pos, nodelist= G.nodes(), node_color='green', node_size=5)
    nx.draw_networkx_nodes(CH, posCH, node_color='red', node_size=20)
    plt.show()

def draw_Hilbert():
    h1(1,degree)
    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=5)
    nx.draw_networkx_nodes(CH, posCH, node_color='red', node_size=20)
    nx.draw_networkx_edges(CH, posCH)
    plt.show()

def set_CH():
    for i in range(int(CHn)):
        pCHx = posCH[i + 1][0]
        pCHy = posCH[i + 1][1]
        cluster = []
        for j in range(N):
            pnx = pos[j][0]
            pny = pos[j][1]
            if pnx >= pCHx-10 and pnx < pCHx+10 and pny >= pCHy-10 and pny < pCHy+10:
                cluster.append(j)
        Clusters.update({i+1:cluster})

    for i in range(int(CHn)):
        d_min = float('inf')
        p = None
        node = None
        for j in range(N):
            d = np.sqrt((pos[j][0] - posCH[i+1][0])**2 + (pos[j][1] - posCH[i+1][1])**2)
            if d < d_min:
                p = pos[j]
                d_min = d
                node = j
        posCH[i+1] = p
        G.remove_node(node)

    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=5)
    nx.draw_networkx_nodes(CH, posCH, node_color='red', node_size=20)
    nx.draw_networkx_edges(CH, posCH)
    plt.show()

def final_network():
    for i in range(len(pos)):
        pos['G-'+str(i)] = pos.pop(i)
    for i in range(len(posCH)):
        posCH['CH-'+str(i+1)] = posCH.pop(i+1)

    pos.update(posCH)
    Q = nx.union(G, CH, rename=('G-', 'CH-'))
    for ch in Clusters.keys():
        head = 'CH-'+str(ch)
        for n in Clusters[ch]:
            node = 'G-'+str(n)
            Q.add_edge(head, node)
    N_list = []
    CH_list = []
    for i in pos.keys():
        if 'CH-' in i:
            CH_list.append(i)
        if 'G-' in i:
            N_list.append(i)

    nx.draw_networkx_nodes(Q, pos, nodelist = N_list, node_color = 'green', node_size = 5)
    nx.draw_networkx_nodes(Q, pos, nodelist = CH_list, node_color = 'red', node_size = 20)
    nx.draw_networkx_edges(Q, pos, edge_list = Q.edges())
    plt.show()


create_WSN()
create_CH()
draw_Hilbert()
set_CH()
final_network()