import json
import os
from src.Converter import toStopPoint

def getResults(data,file,dir,Z,X,Y,I,R,Q,FO,GAP):
    routes = [[toStopPoint(v) for v in Z[t]] for t in range(len(Z))]
    periods = []
    for t in range(len(routes)):
        veicles = []
        for v in range(len(routes[t])):
            points = []
            v_qtd_max = 0
            for i in range(len(routes[t][v])):
                products = []
                r_current = 0

                for p in range (len(Q[t][v])): 
                    if(i!=len(routes[t][v])-1): 
                        r_current=R[t][v][p][routes[t][v][i+1]][routes[t][v][i]]

                    products.append({
                        'p':p+1,
                        'qtd':Q[t][v][p][routes[t][v][i]],
                        'r': r_current
                    })
                    v_qtd_max = v_qtd_max+Q[t][v][p][routes[t][v][i]]
                points.append({
                    'point': routes[t][v][i],
                    'x': data['coordXY']['x'][routes[t][v][i]],
                    'y': data['coordXY']['y'][routes[t][v][i]],
                    'products':products
                })
            veicles.append({
                'v':v+1,
                'points':points,
                'v_qtd_max':v_qtd_max
            })

        productions = []
        for p in range (len(X[t])):
            productions.append({
                'p':p+1,
                'qtd': X[t][p],
                'isProduction': int(Y[t][p])
            })

        est = []
        for i in range (len(I[t])):
            e_p_current = []
            e_current = 0
            for p in range(len(I[t][i])):
                if(t==0):
                    e_current = data['I_pi0'][p][i]
                else:
                    e_current = I[t][i][p]
                e_p_current.append({
                    'p':p+1,
                    'qtd':e_current
                })
            est.append({
                'point':i,
                'products': e_p_current
            })

        periods.append({
            't':t+1,
            'veicles':veicles,
            'productions':productions,
            'estq':est
        })

    results = {
        'periods':periods,
        'FO':FO,
        'gap':GAP
    }

    os.makedirs(f'./out/{dir}', exist_ok=True)
    caminho_arquivo = os.path.join(f'./out/{dir}', f"{file}.json")
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(results, arquivo, indent=4, ensure_ascii=False)
    
    return results

