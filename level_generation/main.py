from wave_function_collapse import *
from conf import TILES_MAP

def main():
    tiles = ['X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']']
    stage = StageWFC(tiles)

    stage_path = 'data/mario-1-1.txt'

    stage.inputStage(stage_path)

    stage.learn_contraints()

    a = stage.generate_level(output_size=(14,100))

    stage.save_level(a, TILES_MAP)
    
if __name__ == "__main__":
    main()