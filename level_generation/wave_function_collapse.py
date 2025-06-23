import numpy as np
import utils
from random import choices
from collections import deque

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3 

class ContradictionError(Exception):
    pass

class StageWFC():
    
    def __init__(self, tiles, start_block='X'):
        """
        Initializes an empty StageWFC object. Must add stages to learn constraints

        Args:
            tiles (List[str]): a list of strings of possible tiles in the level
            Will cause issues later if level contains a character NOT in tiles
        """
        self.level_paths = None
        self.level_arr = None
        self.tiles = self._init_tiles(tiles)
        self.convert_back_tiles = {i:t for t, i in self.tiles.items()}
        self.constraints = self._init_constraints(self.tiles)
        self.pattern_count = self._init_pattern_count(self.tiles)
        self.start_block = start_block

    def _init_tiles(self, tiles):
        return {t:i for i, t in enumerate(tiles)}

    def _init_constraints(self, tiles):
        return {t:{NORTH:set(), EAST:set(), SOUTH:set(), WEST:set()} for t in tiles.values()}
    
    def _init_pattern_count(self, tiles):
        # want to index pattern_count[tile1, dir, tile2] = NUMBER where NUMBER is the 
        # number of times tile2 appears in the direction dir from tile 1 
        return np.zeros((len(self.tiles), 4, len(self.tiles)))
    
    def inputStage(self, level):
        self.level_paths = level
        level_arr = utils.read_text_to_level(level)
        self.level_arr = np.empty(level_arr.shape)
        for x in range(self.level_arr.shape[0]):
            for y in range(self.level_arr.shape[1]):
                self.level_arr[x,y] = self.tiles[level_arr[x][y]]
        self.level_arr = self.level_arr.astype(int)

    def learn_contraints(self, window_size=(2.2)):
        
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
                    self.constraints[stage[i,j]][NORTH].add(stage[i+1,j])
                    self.pattern_count[stage[i,j],NORTH,stage[i+1, j]] += 1
                # east neighbor:
                if j + 1 < y:
                    self.constraints[stage[i,j]][EAST].add(stage[i,j+1])
                    self.pattern_count[stage[i,j],EAST,stage[i, j+1]] += 1
                # south neighbor
                if i - 1 > 0:
                    self.constraints[stage[i, j]][SOUTH].add(stage[i-1, j])
                    self.pattern_count[stage[i,j],SOUTH,stage[i-1, j]] += 1
                # west neighbor
                if j - 1 > 0:
                    self.constraints[stage[i,j]][WEST].add(stage[i, j-1])
                    self.pattern_count[stage[i,j],WEST,stage[i, j-1]] += 1
    
    def generate_level(self, output_size=(14,14)):
        h, w = output_size
        n_tiles = len(self.tiles)
        
        domain = np.ones((h,w, len(self.tiles)), dtype=bool)
        # seed the start block
        si, sj = h-1, 0         # coords of start block
        domain[si,sj,:] = False
        domain[si,sj,self.tiles[self.start_block]] = True
        
        def _entropy(i, j):
            dom = domain[i,j]
            if np.sum(dom) <= 1:
                return np.inf # indicates that this level coord has already been collapsed
            weights = np.zeros(n_tiles)
            for t in np.where(dom)[0]:
                weights[t] = self.pattern_count[t].sum()
            probs = weights[dom] / np.sum(weights[dom])
            return -np.sum(probs * np.log(probs))
        
        
        # def _weighted_sample(i,j):
        #     dom = domain[i,j]
        #     weights = np.zeros(n_tiles)
        #     for t in np.where(dom)[0]:
        #         weights[t] = np.sum(self.pattern_count[t])
        #     mask = dom & (weights > 0)
        #     if not mask.any():
        #         mask = dom
        #         weights = np.ones(n_tiles)
            
        #     candidates = list(np.where(mask)[0])
        #     w = weights[mask]
        #     return choices(candidates, weights=w, k=1)[0]
        
        def _weighted_sample(i,j):
            dom = domain[i,j]
            weights = np.zeros(n_tiles)
            for (di,dj,dir) in zip([1,0,-1,0], [0,1,0,-1], [NORTH, EAST, SOUTH, WEST]):
                ni,nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    nt = np.where(domain[ni,nj])[0][0]
                    opp = (dir+2)%4
                    weights += self.pattern_count[nt, opp, :]
            if not weights[dom].any():
                weights = np.ones(n_tiles)
            candidates = np.where(dom)[0]
            wt = weights[dom]
            return choices(list(candidates), weights=wt, k=1)[0]
                
        def _propagate(i0,j0):
            q = deque([(i0, j0)])
            while q:
                i,j = q.popleft()
                for (di,dj, dir) in zip([1,0,-1,0], [0,1,0,-1], [NORTH, EAST, SOUTH, WEST]):
                    ni,nj = i + di, j+dj
                    
                    if 0<=ni< h and 0<=nj<w:
                        valid = np.zeros(n_tiles, dtype=bool)
                        for t in np.where(domain[i,j])[0]:
                            valid[list(self.constraints[t][dir])] = True
                        new_dom = domain[ni,nj] & valid
                        if not new_dom.any():
                            raise ContradictionError(f'Cell ({ni}, {nj} has no valid tiles)')
                        if np.sum(new_dom) < domain[ni,nj].sum():
                            domain[ni,nj] = new_dom
                            q.append((ni,nj))
                            
        _propagate(si, sj)
        while True:
            candidates = [(i,j) for i in range(h) for j in range(w) if domain[i,j].sum() > 1]
            if not candidates: break
            i_star, j_star = min(candidates, key=lambda ij: _entropy(*ij))
            t_star = _weighted_sample(i_star, j_star)
            domain[i_star, j_star,:] = False
            domain[i_star, j_star, t_star ] = True
            _propagate(i_star,j_star)
            
        level = np.zeros(output_size, dtype=int)
        for i in range(h):
            for j in range(w):
                level[i,j] = np.where(domain[i,j])[0][0]
        level = np.empty(output_size, dtype=str)
        for i in range(h):
            for j in range(w):
                level[i,j] = self.convert_back_tiles[np.where(domain[i,j])[0][0]]
        return level