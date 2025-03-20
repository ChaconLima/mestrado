import sys
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.ProcessResults import getResults
from src.GraphDisplay import graphResults

if __name__ == "__main__":

    dir = sys.argv[1]
    file = sys.argv[2]
    isPlot = sys.argv[3]

    data = RD("./data/"+dir+"/"+file+".dat").getDataSet()

    mpprp = MPPRP(data)
    mpprp.solver()
    Z,X,Y,I,R,Q,FO,GAP = mpprp.getResults()

    results = getResults(data,file,dir,Z,X,Y,I,R,Q,FO,GAP)

    if(isPlot=='true'):
        graphResults(results['periods'])