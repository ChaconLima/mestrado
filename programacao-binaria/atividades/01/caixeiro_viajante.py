# -*- coding: utf-8 -*-
"""Cópia de atividade_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hsogp1rvHjoMM2i_zcTgs1Bohzun2Lfb
"""

#!pip install ortools

#################################################################################################
# Problema do Caixeiro Viajante
#Copyright 2023 Mateus Chacon, Mario Villalba, Danielle Gomes e Felipe Ribeiro

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from time import gmtime, strftime
from ortools.linear_solver import pywraplp
from gurobipy import *
import tsplib95
import networkx as nx
import numpy as np
import time
#************************************************************************************************
#FUNÇÃO DE LEITURA
#************************************************************************************************
def leia(fileAq):
  problem = tsplib95.load('./instancias/'+fileAq+'.tsp')
  graph = problem.get_graph()
  return nx.to_numpy_array(graph)
#************************************************************************************************
#FUNÇÃO QUE EXECUTA O SOLVER - ORTOOLS
#************************************************************************************************
def modeloOrtools(distances, subTours ,timeExe,useMTZ=True, writeLp=False):
  n = distances.shape[0]
  #criando as variaveis xij
  # solver = pywraplp.Solver.CreateSolver('CP-SAT')
  solver = pywraplp.Solver.CreateSolver('SAT')
  # solver = pywraplp.Solver.CreateSolver('SCIP')
  x = {}
  for i in range(n):
      for j in range(n):
          if i != j:
            vname = f'x_{i}_{j}'
            x[i,j] = solver.BoolVar(vname)

  # criando a função objetiva
  solver.Minimize(solver.Sum(distances[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j))

  # adicionando restrição para todo i
  for i in range(n):
      solver.Add(solver.Sum(x[i, j] for j in range(n) if i != j) == 1, name=f'degree_{i}')

  # adicionando restrição para todo j
  for j in range(n):
      solver.Add(solver.Sum(x[i, j] for i in range(n) if i != j) == 1, name=f'degree_{j}')

  # Add MTZ constraints
  if(useMTZ):
    u = {}
    for i in range(1, n):
        vname = f'u_{i}'
        u[i] = solver.IntVar(1 , n-1 ,vname)

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                solver.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')

  # Add restrição sub rotas
  for m in range(len(subTours)):
        r5 = 0
        for i in range(len(subTours[m])):
            for j in range(len(subTours[m])):
                if i != j:
                    r5 += x[subTours[m][i], subTours[m][j]]
        solver.Add(r5 <= len(subTours[m]) - 1, name=f"sub_{m}_{i}_{j}")

  # gera o lp do modelo
  lp = "model-LP-"+str(strftime("%Y-%m-%dT%H:%M:%S", gmtime()))+".mps"
  if writeLp:
    with open(lp, "w") as out_f:
        mps_text = solver.ExportModelAsLpFormat(False)
        out_f.write(mps_text)

  #configuração da execução
  solver.set_time_limit(timeExe*1000) 

  #executa o solver
  init = time.time()
  status = solver.Solve()
  fim = time.time()
  if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    coords = []
    for i in range(n):
      for j in range(n):
        if (i != j):
          if round(x[j,i].solution_value()) == 1:
            coords.append([i,j])
    print('')
    print('Solution: Objective value =', solver.Objective().Value(),' AND execution in =', round(fim-init), '[s]\n')
    return solver.Objective().Value(), coords,0
  else:
    print("Infeasible")
    return 0, [], 0
#************************************************************************************************
#FUNÇÃO QUE EXECUTA O SOLVER - GUROBI
#************************************************************************************************
def modeloGurobi(distances, subTours ,timeExe,useMTZ=True, writeLp=False):
  n = distances.shape[0]
  # Create a Gurobi model
  model = Model()

  # Create variables
  x = {}
  for i in range(n):
      for j in range(n):
          if i != j:
              x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

  # Set objective
  model.setObjective(quicksum(distances[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

  # Add degree constraints para i
  for i in range(n):
      model.addConstr(quicksum(x[i, j] for j in range(n) if i != j) == 1, name=f'degree_{i}')

  # Add degree constraints para j
  for j in range(n):
      model.addConstr(quicksum(x[i, j] for i in range(n) if i != j) == 1, name=f'degree_{j}')

  # Add MTZ constraints
  if(useMTZ):
    u = {}
    for i in range(1, n):
        u[i] = model.addVar(lb=1, ub=n - 1, vtype=GRB.INTEGER, name=f'u_{i}')

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')

   # Add restrição sub rotas
  for m in range(len(subTours)):
        r5 = 0
        for i in range(len(subTours[m])):
            for j in range(len(subTours[m])):
                if i != j:
                    r5 += x[subTours[m][i], subTours[m][j]]
        model.addConstr(r5 <= len(subTours[m]) - 1, name=f"sub_{m}_{i}_{j}")

  model.Params.TimeLimit = timeExe
  #executa o solver
  init = time.time()
  model.optimize()
  fim = time.time()

  # gera o lp do modelo
  lp = "model-LP-"+str(strftime("%Y-%m-%dT%H:%M:%S", gmtime()))+".mps"
  if writeLp:
    model.write(lp)

  # Print the solution
  if model.status == GRB.OPTIMAL:
    coords = []
    fo= model.ObjVal
    gap = model.MIPGap
    for i in range(n):
      for j in range(n):
        if (i != j):
          if round(x[j,i].x) == 1:
            coords.append([i,j])
    print('')
    print('Solution: Objective value =', fo, 'Gap = ',gap,' AND execution in =', round(fim-init), '[s]\n')
    return fo, coords, gap
  else:
    print("Infeasible")
    return 0, [], 0

#************************************************************************************************
#FUNÇÃO QUE RETORNA OS SUB ROTAS
#************************************************************************************************
def createSubTours(coords):
    n = len(coords)
    cities_left = set(range(n))  # Conjunto de índices das cidades não visitadas
    sub_tours = []
    while cities_left:
        current_city = cities_left.pop()  # Inicia um novo sub-tour
        sub_tour = [current_city]
        while True:
            found_next = False
            for i in range(n):
                if coords[current_city][1] == coords[i][0] and i in cities_left:
                    current_city = i
                    sub_tour.append(current_city)
                    cities_left.remove(current_city)
                    found_next = True
                    break
            if not found_next:
                break
        sub_tours.append(sub_tour)
    return sub_tours
#************************************************************************************************
#FUNÇÃO QUE CRIA AS PERMUNTAÇÕES
#************************************************************************************************
import itertools
import random
def permutaciones_en_matriz(input_list, num_permutations):
    permutations_list = []
    input_length = len(input_list)
    
    # Garantir que o número de permutações não exceda o total possível
    num_permutations = min(num_permutations, factorial(input_length))
    
    for _ in range(num_permutations):
        permutation = list(input_list)  # Faz uma cópia do vetor original
        # Aplica o algoritmo de troca de elementos para gerar a próxima permutação
        for i in range(input_length - 1, 0, -1):
            j = random.randint(0, i)
            permutation[i], permutation[j] = permutation[j], permutation[i]
        permutations_list.append(permutation)
    
    return permutations_list

# Função auxiliar para calcular fatorial
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

#************************************************************************************************
# METODO MTZ
#************************************************************************************************
def mtz(fileArq,timeExe,writeLp,solver='ortools'):
  distances = leia(fileArq)
  subTours=[]
  match solver:
    case 'ortools':
      fo, coords, gap = modeloOrtools(distances,subTours,timeExe,True,writeLp)
      subTours = createSubTours(coords)
      return fo,subTours,gap
    case 'gurobi':
      fo, coords, gap = modeloGurobi(distances,subTours,timeExe,True,writeLp)
      subTours = createSubTours(coords)
      return fo,subTours,gap
#************************************************************************************************
# PATAKI MTZ
#************************************************************************************************
def pataki(fileArq,timeExe,writeLp,solver):
  distances = leia(fileArq)
  subTours = []
  match solver:
    case 'ortools':
      for i in range(100):
        print("Iteração: ",i)
        fo, coords, gap = modeloOrtools(distances,subTours,timeExe,False,writeLp)
        subToursCorrent = createSubTours(coords)
        if(len(subToursCorrent)==1):
          return fo,subToursCorrent,gap
        subTours.extend(subToursCorrent)
    case 'gurobi':
      for i in range(100):
        print("Iteração: ",i)
        fo, coords, gap = modeloGurobi(distances,subTours,timeExe,False,writeLp)
        subToursCorrent = createSubTours(coords)
        if(len(subToursCorrent)==1):
          return fo,subToursCorrent,gap
        subTours.extend(subToursCorrent)

def pataki2(fileArq,timeExe,writeLp,solver):
  distances = leia(fileArq)
  subTours = []
  match solver:
    case 'ortools':
      for i in range(100):
        print("Iteração: ",i)
        fo, coords, gap = modeloOrtools(distances,subTours,timeExe,False,writeLp)
        subToursCorrent = createSubTours(coords)
        if(len(subToursCorrent)==1):
          return fo,subToursCorrent,gap
        subTours.extend(subToursCorrent)
    case 'gurobi':
      for i in range(100):
        print("Iteração: ",i)
        fo, coords, gap = modeloGurobi(distances,subTours,timeExe,False,writeLp)
        subToursCorrent = createSubTours(coords)
        if(len(subToursCorrent)==1):
          return fo,subToursCorrent,gap
        
        for j in range(len(subToursCorrent)): 
          subTours.extend(permutaciones_en_matriz(subToursCorrent[j],len(subToursCorrent[j])))

#************************************************************************************************
# FUNÇÃO QUE CHAMA OS METODOS
#************************************************************************************************
def metodos(meth,fileArq,timeExe,writeLp,solver):
  match meth:
    case 'MTZ':
      print("==============================Executando MTZ - ",fileArq,"==============================")
      init = time.time()
      fo, route, gap = mtz(fileArq,timeExe,writeLp,solver)
      fim = time.time()
      print("MTZ - tempo total",round(fim-init), '[s]')
      return fo, route, round(fim-init), gap
    case 'PATAKI':
      print("==============================Executando PATAKI - ",fileArq,"==============================")
      init = time.time()
      fo, route, gap = pataki(fileArq,timeExe,writeLp,solver)
      fim = time.time()
      print("PATAKI - tempo total",round(fim-init), '[s]')
      return fo, route, round(fim-init), gap
    case 'PATAKI-2':
      print("==============================Executando PATAKI - ",fileArq,"==============================")
      init = time.time()
      fo, route, gap = pataki2(fileArq,timeExe,writeLp,solver)
      fim = time.time()
      print("PATAKI - tempo total",round(fim-init), '[s]')
      return fo, route, round(fim-init), gap
#************************************************************************************************
# FUNÇÃO DE IMPRIME
#************************************************************************************************
def imprime(solver,dataset,metodo,fo,time,gap,route):
  lp = "resultados/"+str(solver)+"-"+str(dataset)+"-"+str(metodo)+"-"+str(strftime("%Y-%m-%dT%H:%M:%S", gmtime()))+".txt"
  with open(lp, 'a') as arquivo:
      message = 'Fo:\t'+str(fo)+'\nTempo:\t'+str(time)+'\nGap:\t'+str(gap)+'\nRoute:\t'+str(route)
      arquivo.write(message) 

#************************************************************************************************
#FUNÇÃO MAIN
#************************************************************************************************
def __main__():
  writeLp = False #ligar ou desligar a criação do arquivo lp
  timeLimitSolver = 3600    #limitante do tempo de execução do solver em segundos 3600 => 60 minutos
  solver = 'gurobi'

  fileArq='bays29'
  fo, route, time, gap = metodos('MTZ',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n\tGap:',gap)
  imprime(solver,fileArq,'MTZ',fo,time,gap,route)
  fo, route, time, gap = metodos('PATAKI',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n\tGap:',gap)
  fo, route, time, gap = metodos('PATAKI-2',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n\tGap:',gap)
  imprime(solver,fileArq,'PATAKI',fo,time,gap,route)

  # solver = 'ortools'
  # fileArq='bays29'
  # fo, route, time, gap = metodos('MTZ',fileArq,timeLimitSolver,writeLp,solver)
  # print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n\tGap:',gap)
  # imprime(solver,fileArq,'MTZ',fo,time,gap)
  # fo, route, time, gap = metodos('PATAKI',fileArq,timeLimitSolver,writeLp,solver)
  # print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n\tGap:',gap)
  # imprime(solver,fileArq,'PATAKI',fo,time,gap)
 
  fileArq='brazil58'
  fo, route, time, gap = metodos('MTZ',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')
  imprime(solver,fileArq,'MTZ',fo,time,gap,route)
  fo, route, time, gap = metodos('PATAKI',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')
  imprime(solver,fileArq,'PATAKI',fo,time,gap,route)

  # solver = 'gurobi'
  # fileArq='brazil58'
  # fo, route, time = metodos('MTZ',fileArq,timeLimitSolver,writeLp,solver)
  # print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')
  # fo, route, time = metodos('PATAKI',fileArq,timeLimitSolver,writeLp,solver)
  # print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')

  fileArq='si535'
  fo, route, time, gap = metodos('MTZ',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')
  imprime(solver,fileArq,'MTZ',fo,time,gap,route)
  fo, route, time, gap = metodos('PATAKI',fileArq,timeLimitSolver,writeLp,solver)
  print('Aquivo: ',fileArq, 'Solução: ', '\n\tFO: ',fo,'\n\tRoute:',route,'\n')
  imprime(solver,fileArq,'PATAKI',fo,time,gap,route)
__main__()


