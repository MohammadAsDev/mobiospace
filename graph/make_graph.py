
import numpy as np
import numpy.random as rand

def main():
    nodes = list(range(1 , 10001))
    edges = rand.permutation(nodes).reshape((-1, 2))

    edgelist_content = ""
    edgelist_file= "testing_graph.edgelist"

    for edge in edges:
        edgelist_content += "{} {}\n".format(edge[0] , edge[1])

    handler = open(edgelist_file , "w+")
    handler.writelines(edgelist_content)

if __name__ == "__main__":
    main()





