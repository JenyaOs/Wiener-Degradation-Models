import multiprocessing as mp
from ResearchOfPower.FindPowerCriterion import ResearchFunctionChi2, ResearchFunction, OutPutResultsTests, OutPutFirstLine, GenerationDataOFModel, WienerModelWithPowerTrend, WienerModel
import matplotlib.pyplot as plt
from Module.test import Kolmogorov_Test,CramerVonMisesSmirnov_Test,AndersonDarling_Test,Xi_2,ToCheckAdequacyOfModel, GetF
from ResearchOfEstimate.Estimate import function, output, getAverage
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
#блок для расспараллеливания
def research_block(M, x0, path, mutex):

    A = function(M, x0)

    with mutex:
        output(path+"test_cov_D_3_5.txt", M, A)
    return

def input_Ar(filename):
    res = []

    filename = open(filename,"r")
    line = filename.readline()
    while(line):
        string = line.split(" ")
        print(string)
        Ar = []
        for i in range(len(string)-1):
            Ar.append(float(string[i]))
        line = filename.readline()
        res.append(np.array(Ar).transpose())
    #graph(Ar, label, criterion)
    filename.close()
    return res

#основная программа
if __name__=='__main__':
    def researchers(M, x0, path, mk_mode=True):
     #  OutPutFirstLine(path, H)

        i = 0
        thread_list = []
        # Объявляем блокировщик
        mutex = mp.Lock()

        #with Timer("Time with threads"):
        if mk_mode:
            while i < 16:
                data_global = []
                #for j in range(M):
                #    data = GenerationDataOFModel([1, 1, 1.2], [k/2 for k in range(N)], K, WienerModelWithPowerTrend)
                #    data_global.append(data)
                thread = mp.Process(target=research_block, args=(M, x0, path, mutex))
                i += 2
                print(i)
                thread_list.append(thread)

        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()


    M = 2075

    path = "estimate/"
    researchers(M, [1, 1, 0.2, 0.5], path, True)
    #researchers(M,[0.5, 1, 0.2, 0.5], path, True)

    A = input_Ar("estimate/test_cov_D_3_5.txt")
    getAverage(np.array(A).transpose(), [1, 1, 0.2, 0.5], 16600)
    #
    #researchers(M,K,N,H,path, True)

 #   M = 2075
 #   K = 40
 #   N = 20
    # H = "H0"
 #   path = "40_20/"
    #researchers(M,K,N,H,path, True)

    #M = 2075
 #   K = 20
 #   N = 40
    #H = "H0"
 #   path = "20_40/"

   # M = 2075
   # K = 40
   # N = 40
    #H = "H0"
    #path = "40_40/"
    #researchers(M,K,N,H,path, True)



#Chi2 = input_Ar("10_10/Хи_2-ПирсонаH0.dat")
#ecdf = ECDF(Chi2)
#plt.plot(ecdf.x, ecdf.y)
#plt.plot(ecdf.x, Xi_2(5).get_LimitDistribution(ecdf.x))
#plt.show()
