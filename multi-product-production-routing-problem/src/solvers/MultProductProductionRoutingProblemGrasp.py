#################################################################################################
# Multi Product Prodction Routing Problem Gras
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from src.log.Logger import Logger
from src.solvers.GreedyRandomizedConstructionRoute import GreedyRandomizedConstructionRoute as GR
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as MPPRPG
import numpy as np

class MultProductProductionRoutingProblemGrasp:
    def __init__(self,map,dir,log:Logger,rng:np.random.Generator):
        self.data = map
        self.p=map['num_products']            ##Products  
        self.i=map['num_customers'] + 1       ##Customers
        self.k=map['num_customers'] + 1       ##Customers
        self.t=map['num_periods']             ##Periods
        self.v=map['num_vehicles']            ##Vehicles         
        self.B=map['B']                       ##Production capacity;
        self.b_p=map['b_p']                   ##Time required to produce item 𝑝;
        self.c_p=map['c_p']                   ##Production cost of item 𝑝;
        self.s_p=map['s_p']                   ##Setup cost of item 𝑝;
        self.M=map['M']                       ##Big number 
        self.U_p_i=map['U_pi']                ##Maximum inventory upper bound of item 𝑝 at site i;
        self.I_p_i_0=map['I_pi0']             ##Initial Inventory of item 𝑝 at site 𝑖;
        self.h_p_i=map['h_pi']                ##Inventory cost of item 𝑝 at site 𝑖;
        self.C=map['C']                       ##Vehicle capacity;
        self.f=map['f']                       ##Fixed transportation cost;
        self.a_i_k=map['a_ik']                ##Transportation cost for traveling from node 𝑖 to node k;
        self.d_p_i_t=map['d_pit']             ##Demand of item 𝑝 at customer 𝑖 in period 𝑡.
        self.X_p_t={}                         ##Quantity of item 𝑝 produced in period 𝑡.
        self.Y_p_t={}                         ##1, if item 𝑝 is produced in period 𝑡; or 0, otherwise.
        self.I_p_i_t={}                       ##Inventory of item 𝑝 at site 𝑖 in the end of period 𝑡.
        self.Z_v_i_k_t={}                     ##1, if vehicle v travels along edge (i,k) in period t; or 0, atherwise.
        self.R_p_v_i_k_t={}                   ##Quantity of item 𝑝 transported by vehicle 𝑣 on edge (𝑖, 𝑘) in period 𝑡;
        self.Q_p_v_i_t={}                     ##Quantity of item 𝑝 delivered by vehicle 𝑣 to customer 𝑖 in period 𝑡.
        self.dir = dir
        self.time = 0
        self.solCount = 0
        self.log:Logger = log
        self.max_inter = 100
        self.alfa = 0.2
        self.seed = 123
        self.greedyRoute = GR(log=log)
        self.variables={}
        self.solution={}
        self.mitStart = False
        self.solverGurobi = 0
        self.rng=rng

    def setMitStart(self,mitStart):
        self.mitStart = mitStart

    def setMaxInter(self,max_inter):
        self.max_inter = max_inter
    
    def setAlfa(self,alfa):
        self.alfa = alfa
    
    def setSeed(self,seed):
        self.seed = seed


    def solver(self,numThreads=None,timeLimit=None):

        print("chegou aqui! seed", self.seed)

        for alpha in np.linspace(0, 1, 111):
            inst = MPPRPG(map=self.data,dir=self.dir,log=self.log,rng=self.rng)
            inst.solver()
            fo = inst.getValueObjectiveFunction()

            print(f"alpha: {alpha} solucao: {fo}")
            

        if(self.mitStart==True):
            self.solverGurobi = MPPRP(self.data,self.dir,self.log,{"start":True, "variables":self.variables})
            self.solverGurobi.solver(timeLimit=timeLimit,numThreads=numThreads)


    def convertVariables(self):

        final_solution = self.solution
        Z = np.zeros((self.v,self.i,self.k,self.t), dtype=int)

        '''𝑧𝑣𝑖𝑘𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["route"])):
                for i in range( len(final_solution["routes"][t]["route"][v])):
                    origem = final_solution["routes"][t]["route"][v][i]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino = 0
                    else:
                        destino = final_solution["routes"][t]["route"][v][i+1]
                    Z[v, origem,destino, t] = 1
        R = np.zeros((self.p,self.v,self.i,self.k,self.t), dtype=int)
        Q = np.zeros((self.p,self.v,self.i,self.t), dtype=int)

        '''𝑟𝑝𝑣𝑖𝑘𝑡'''
        '''𝑞𝑝𝑣𝑖𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["demandas"])):
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino_j = 0
                    else:
                        destino_j = final_solution["routes"][t]["demandas"][v]["entregas"][i+1]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        R[p,v,origem_i,destino_j,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["restante_veiculo"]
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        Q[p,v,origem_i,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["qte_entregue"]

        X = np.zeros((self.p,self.t), dtype=int)
        Y = np.zeros((self.p,self.t), dtype=int)
        I = np.zeros((self.p,self.i,self.t), dtype=int)
        '''𝐼𝑝𝑖𝑡'''
        for t in range(len(final_solution["production"])):
            producao = np.zeros((self.p), dtype=int)
            for i in range(len(final_solution["production"][t])):
                for p in range(len(final_solution["production"][t][i])):
                    producao[p]+=final_solution["production"][t][i][p]["producaco"]
                    I[p,i,t] = final_solution["production"][t][i][p]["estoque"] - final_solution["production"][t][i][p]["demanda"]


            for p in range(len(final_solution["production"][t][i])):
                if(producao[p]>0):
                    Y[p,t] = 1
                X[p,t] = producao[p]

        self.variables={"X":X, "Y":Y, "I":I, "Q":Q, "R":R, "Z":Z}

        return Z,X,Y,I,R,Q
    def getResultsSolverHeurisct(self):
        z=self.variables["Z"]
        x=self.variables["X"]
        y=self.variables["Y"]
        ii=self.variables["I"]
        r=self.variables["R"]
        q=self.variables["Q"]
        '''print("*******************************")
        print("============ Z ================")
        print("*******************************")'''
        Z=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            v_list =[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                i_list =[]
                for i in range(self.i):
                    k_list=[]
                    for k in range(self.k):
                        variable = z[v,i,k,t]
                        #print(" origem: ",i," destino: ",k," == ",variable)
                        k_list.append(int(variable))
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        #print("\n\n===============================\n\n")
        '''for t in range(len(Z)):
            print("\n\n============ periodo ",t," ============")
            for v in range(len(Z[t])):
                print("\n============ veiculo ",v," ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string+= str(Z[t][v][i][k]) + "\t"
                    print(string)'''

        '''print("*******************************")
        print("============ Y ================")
        print("*******************************")'''
        Y = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_y=[]
            for p in range(self.p):
                variable = abs(y[p,t])
                p_list_y.append(int(variable))
                #print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        #print("\n\n===============================\n\n")
        '''print("*******************************")
        print("============ X ================")
        print("*******************************")'''
        X = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_x=[]
            for p in range(self.p):
                p_list_x.append(int(x[p,t]))
                #print("produto: ",p," == ",x[p,t])
            X.append(p_list_x)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ I ================")
        print("*******************************")'''
        I=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_i=[]
            for i in range(self.i):
                i_list_i=[]
                #print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    #print("produto: ",p," == ", ii[p,i,t])
                    i_list_i.append(int(ii[p,i,t]))
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ R ================")
        print("*******************************")'''
        R=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    p_list=[]
                    for i in range(self.i):
                        i_list=[]
                        for k in range(self.k):
                            #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente {i} -> cliente {k} -> produto {p} == { r[p,v,i,k,t]}",)
                            #print("\n============ cliente ",i," -> cliente ",k," ============")
                            i_list.append(float(r[p,v,i,k,t]))
                            #print("produto: ",p," == ", r[p,v,i,k,t])
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ Q ================")
        print("*******************************")'''
        Q=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    #print("\n============ cliente ",i," ============")
                    p_list=[]
                    for i in range(self.i):
                        #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente -> {i} -> produto {p} == { q[p,v,i,t]}",)
                        #print("produto: ",p," == ",q[p,v,i,t])
                        p_list.append(int(q[p,v,i,t]))
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        #print("\n\n===============================\n\n")
    
        return Z,X,Y,I,R,Q,0,0,0,0,0,0,0   


    def getResults(self):
        if(self.mitStart==True):
            return self.solverGurobi.getResults()

        return self.getResultsSolverMetaHeurisct()
