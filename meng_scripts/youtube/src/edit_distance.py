# Copyright 2020  Meng Wu

def edit_distance(x1, x2):
    '''
    Input:
        two list, order: ref, hyp
        Note:
            insertion = D(i, j-1) +1
             deletion = D(i-1, j) +1
                match = D(i-1, j-1)
             mismatch = D(i-1, j-1) +1
    Returns:
        edit distance & trace back path
    '''
    M, N = len(x1), len(x2)
    # create empty array with (N+1) * (M+1)
    array = [[0] * (N+1) for i in range(M+1)]
    back = [['hyp'] * (N+1) for i in range(M+1)]
    trans = [['hyp'] * (N+1) for i in range(M+1)]
    opt = [['hyp'] * (N+1) for i in range(M+1)]
    for i in range(M+1):
        for j in range(N+1):
            if i == 0 and j == 0:
                array[i][j] = 0
            elif i == 0 and j != 0:
                array[i][j] = j
                back[i][j] = back[i][j-1] + ' D'
                trans[i][j] = back[i][j-1] + ' **'
                opt[i][j] = opt[i][j-1] + ' ' + x2[j-1] + ':**'
            elif i != 0 and j == 0:
                array[i][j]  = i
                back[i][j] = back[i-1][j] + ' I'
                trans[i][j] = trans[i-1][j] + ' &&'
                opt[i][j] = opt[i-1][j] + ' &&:' + x1[i-1]
            elif x1[i-1] == x2[j-1]:
                c = [array[i-1][j-1], array[i-1][j] + 1, array[i][j-1] + 1]
                array[i][j] = min(c)
                index = c.index(min(c))
                if index == 0:
                    back[i][j] = back[i-1][j-1] + ' C'
                    trans[i][j] = trans[i-1][j-1] + ' ' + x1[i-1]
                    opt[i][j] = opt[i-1][j-1] + ' ' + x1[i-1] + ':' + x1[i-1]
                elif index == 1:
                    back[i][j] = back[i-1][j] + ' I'
                    trans[i][j] = trans[i-1][j-1] + ' &&'
                    opt[i][j] = opt[i-1][j-1] + ' &&:' + x1[i-1]
                else:
                    back[i][j] = back[i][j-1] + ' D'
                    trans[i][j] = trans[i][j-1] + ' **'
                    opt[i][j] = opt[i][j-1] + ' ' + x2[j-1] + ':**'
            else:
                c = [array[i-1][j-1] + 1, array[i-1][j] + 1, array[i][j-1] + 1]
                array[i][j] = min(c)
                index = c.index(min(c))
                if index == 0:
                    back[i][j] = back[i-1][j-1] + ' S'
                    trans[i][j] = trans[i-1][j-1] + ' $$'
                    opt[i][j] = opt[i-1][j-1] + ' ' + x2[j-1] + ':' + x1[i-1]
                elif index == 1:
                    back[i][j] = back[i-1][j] + ' I'
                    trans[i][j] = trans[i-1][j] + ' &&'
                    opt[i][j] = opt[i-1][j] + ' &&:' + x1[i-1]
                else:
                    back[i][j] = back[i][j-1] + ' D'
                    trans[i][j] = trans[i][j-1] + ' **'
                    opt[i][j] = opt[i][j-1] + ' ' + x2[j-1] + ':**'
                    
    return array[M][N], back[M][N], trans[M][N], opt[M][N]