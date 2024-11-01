
import random 

def main():
    road_file = "road-chesapeake.edgelist"
    road_lines = open(road_file).readlines()
    road_data = [line.strip().split() for line in road_lines]

    weighted_road_data = road_data.copy() 
    _ = [edge.append(str(random.randint(1 , 10))) for edge in weighted_road_data]

    weighted_edgelist_str = "\n".join([" ".join(weighted_edge) for weighted_edge in weighted_road_data])
    
    weighted_road_file = open("weighted-road-chesapeake.edgelist" , "w")
    weighted_road_file.write(weighted_edgelist_str)

if __name__ == "__main__":
    main()
