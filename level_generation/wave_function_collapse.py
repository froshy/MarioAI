import numpy as np
import utils


class StageWFC():
    
    def __init__(self, tiles):
        """
        Initializes an empty StageWFC object. Must add stages to learn constraints

        Args:
            tiles (List[str]): a list of strings of possible tiles in the level
            Will cause issues later if level contains a character NOT in tiles
        """
        self.level_paths = None
        self.level_arr = None
        self.tiles = tiles
        self.constraints = self._init_constraints(tiles)


    def _init_constraints(self, tiles):
        return {t:{'n':set(), 'e':set(), 's':set(), 'w':set()} for t in tiles}
    
    def inputStage(self, level):
        self.level_paths = level
        self.level_arr = utils.read_text_to_level(level)


    def learn_contraints(self, window_size=(2.2), stage_id=0):
        
        # look at each element in array stage, 
        # check its neighbors, 
        # then add neighors (in the 4 directions) to 'feasible neighbors'
        if self.level_arr is None: raise Exception("Empty stage. Input a stage using inputStage()")
            
        stage = self.level_arr
        x, y = stage.shape
        for i in range(x):
            for j in range(y):
                
                # north neighbor:
                if i + 1 < x:
                    self.constraints[stage[i,j]]['n'].add(stage[i+1,j])
                # east neighbor:
                if j + 1 < y:
                    self.constraints[stage[i,j]]['e'].add(stage[i,j+1])
                # south neighbor
                if i - 1 > 0:
                    self.constraints[stage[i, j]]['s'].add(stage[i-1, j])
                # west neighbor
                if j - 1 > 0:
                    self.constraints[stage[i,j]]['w'].add(stage[i, j-1])
            