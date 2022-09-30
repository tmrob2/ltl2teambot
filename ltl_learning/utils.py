import numpy as np

def build_relations(formula, max_len=10):
    ''' Function to construct the skew-symmetric relation matrix. '''

    def length(formula):
        ''' Aux method to recursively determine the length of formula. '''
        if type(formula) == str:
            return 1
        if len(formula) == 2:
            return length(formula[1]) + 1
        if len(formula) == 3:
            return length(formula[1]) + length(formula[2]) + 1

    # define a max_len x max_len matrix with as many 1s on the diagonal as the length of formula
    rel_len = length(formula)   # length of formula
    pad_len = max_len - rel_len # length of padding
    mat = np.diag(np.concatenate([np.ones(rel_len), np.zeros(pad_len)]))

    # define a proper relation vocabulary
    V = {k: v for v, k in enumerate([" ", "s", "p", "l", "r", "p_", "l_", "r_"])}

    def tagger(formula, mat, prev_idx=-1, idx=0, rel=None):
        ''' Aux method to recursively fill the relation matrix based on formula. '''
        def aux(mat, i, j, rel):
            if prev_idx != -1 and rel is not None:
                mat[prev_idx, idx] = V[rel]
                mat[idx, prev_idx] = V[rel + '_']
        if type(formula) == str:
            aux(mat, prev_idx, idx, rel)
            return idx
        if len(formula) == 2:
            aux(mat, prev_idx, idx, rel)
            return tagger(formula[1], mat, idx, idx+1, 'p') # parent relation
        if len(formula) == 3:
            aux(mat, prev_idx, idx, rel)
            offset = tagger(formula[1], mat, idx, idx+1, 'l') # left
            return tagger(formula[2], mat, idx, offset+1, 'r') # right

    tagger(formula, mat)

    # return the properly filled relation matrix of formula
    return mat