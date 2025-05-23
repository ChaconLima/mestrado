from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.ProcessResults import getResults
from src.GraphDisplay import graphResults

class InstanceProcess:

    def __init__(self,instance,output,isPloat='false',numThreads=None,timeLimit=None):
        self.instance = instance
        self.isPloat = isPloat
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.output=output
        self.isFinished=False

    def isProcessFinished(self):
        return self.isFinished
    
    def process(self):
        data = RD(self.instance).getDataSet()

        mpprp = MPPRP(data,self.output)
        mpprp.solver(timeLimit=self.timeLimit,numThreads=self.numThreads)
        Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT = mpprp.getResults()

        results = getResults(data,self.output,Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT)

        if(self.isPloat=='true'):
            graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},self.output)

        self.isFinished = True
