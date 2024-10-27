#ANFIS model create and evolv
import itertools
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_model(soln, args, n_mf, n_outputs):
        #Unpack
        #print("create model solution: ", soln)
        X = args[0]                 # Input dataset
        Y = args[1]
        n_mf = np.asarray(n_mf)
        n_outputs = n_outputs

        n_inputs = len(n_mf)           # Number of features/inputs
        n_pf = n_mf.sum()             # Number of premise MFs
        n_cf = n_mf.prod()            # Number of consequent MFs

        #Number of variables
        n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_outputs            # Output dataset
        #print(n_inputs, n_mf, n_pf, n_cf, n_var, n_outputs)
        combs = build_combs(n_mf, n_inputs)
        Xe = expand_input_dataset(X, n_pf, n_mf, n_inputs)
        mu, s, c, A = build_param(soln, n_pf, n_var,n_inputs, n_cf, n_outputs)

        #Calculate the output
        f = forward_steps(X, Xe,  mu, s, c, A, combs, n_outputs, n_cf)
        #print("parameters: ", soln)
        print("actual: ", Y)
        f = f.flatten()
        print("Predicted: ", f)
        #print ("y - f: ", Y-f)
        #f = f + np.mean(Y)
        mmre1 = calc_mmre(Y, f)
        print("mmre1:", mmre1)
        pred1 = calc_pred(Y, f)
        mae1 = mean_absolute_error(Y, f)
        r21 = r2_score(Y, f)
        adjr21 = calc_adjR2(Y,f, n_inputs)
        sa1 = calc_sa(Y,f)
        args_msr = (mae1, adjr21)
        ttl_msr = calc_msr (Y, f, X.shape[1])
        return  args_msr, ttl_msr

def eval_data(Xp, optimal_soln, n_mf, n_outputs):
        # Expand the input dataset to match the number of premise MFs.
        n_mf = np.asarray(n_mf)
        n_outputs = n_outputs
        n_inputs = len(n_mf)           # Number of features/inputs
        n_pf = n_mf.sum()             # Number of premise MFs
        n_cf = n_mf.prod()            # Number of consequent MFs
        # Number of variables
        n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_outputs            # Output dataset
        #print("eval data: ", n_inputs, n_mf, n_pf, n_cf, n_var, n_outputs)
        combs = build_combs(n_mf, n_inputs)
        Xpe = expand_input_dataset(Xp, n_pf, n_mf, n_inputs)
        mu, s, c, A = build_param(optimal_soln, n_pf, n_var,n_inputs, n_cf, n_outputs)

        # Calculate the output
        f = forward_steps(Xp, Xpe,  mu, s, c, A, combs, n_outputs, n_cf)
        Yp = f
        return Yp

def build_combs(n_mf, n_inputs):
        idx = np.cumsum(n_mf)
        v = [np.arange(0, idx[0])]

        for i in range(1, n_inputs):
            v.append(np.arange(idx[i-1], idx[i]))

        list_combs = list(itertools.product(*v))
        combs = np.asarray(list_combs).T
        return combs
def expand_input_dataset(X,n_pf,n_mf,n_inputs):
        """
        Expands the input dataset to match the number of premise MFs. Each MF
        will be paired with the correct feature in the dataset.
        """
        n_samples = X.shape[0]
        Xe = np.zeros((n_samples, n_pf))       # Expanded array
        idx = np.cumsum(n_mf)
        i1 = 0

        for i in range(n_inputs):
            i2 = idx[i]
            Xe[:, i1:i2] = X[:, i].reshape(n_samples, 1)
            i1 = idx[i]

        return Xe

def build_param(soln, n_pf, n_var,n_inputs, n_cf, n_outputs):
        i1 = n_pf
        i2 = 2 * i1
        i3 = 3 * i1
        i4 = n_var
        # Premise function parameters (generalized Bell functions)
        mu = soln[0:i1]
        s = soln[i1:i2]
        c = soln[i2:i3]

        # Consequent function parameters (hyperplanes)
        A = soln[i3:i4].reshape(n_inputs + 1, n_cf * n_outputs)
        return mu, s, c, A
def forward_steps(X, Xe, mu, s, c, A, combs, n_outputs, n_cf):
        """
        Calculate the output giving premise/consequent parameters and the
        input dataset.
        """
        n_samples = X.shape[0]
        """print(self.mu)
        print(self.s)
        print(self.c)
        print(self.A)"""

        # Layer 1: premise functions (pf)
        d = (Xe - mu) / s
        pf = 1.0 / (1.0 + (d * d) ** c)
        #pf = np.exp(-((Xe - mu)**2) / (2 * s**2))
        """Xe2 = Xe.flatten()
        pf = np.exp(-((Xe2 - mu) ** 2.) / float(s) ** 2.)
        print("xe shape: ", Xe2.shape)
        print("pf shape: ", pf.shape)
        pf = pf.reshape(n_samples, 2)
        print("pf shape: ", pf.shape) """

        #dd = np.square((Xe - self.mu)/self.s)
        #pf = np.exp(-dd)
        # Layer 2: firing strenght (W)
        W = np.prod(pf[:, combs], axis=1)   #weights multiplication

        # Layer 3: firing strenght ratios (Wr)
        Wr = W / W.sum(axis=1, keepdims=True)
        # Layer 4 and 5: consequent functions (cf) and output (f)
        #print("X: ", X)
        X1 = np.hstack((np.ones((n_samples, 1)), X))
        f = np.zeros((n_samples, n_outputs))
        for i in range(n_outputs):
            i1 = i * n_cf
            i2 = (i + 1) * n_cf
            """print("wr: ", Wr)
            print("X1: ", X1)
            print("A: ", A[:, i1:i2]) """
            cf = Wr * (X1 @ A[:, i1:i2])
            #print("cf: ",cf)
            #print("Wr and A values: ", Wr, A[:, i1:i2])
            #print("len of cf: ", cf.shape, cf)
            f[:, i] = cf.sum(axis=1)
            #print("predicted:f ", f[:, i])
        return f

#CoRSNS optimization for ANFIS parameters
import numpy as np
import random

def WSM_Co(fit1,fit2):
    weights = np.array([0.5,0.5])
    #print(fit1,fit2)
    min_max = np.array([min(fit1[0],fit2[0]), max(fit1[0],fit2[0])])
    fit1_N = np.array([min_max[0]/fit1[0], fit1[1]/min_max[1]])
    fit2_N = np.array([min_max[0]/fit2[0], fit2[1]/min_max[1]])
    fit1_sum = (weights[0]*fit1_N[0])+ (weights[1]*fit1_N[1])
    fit2_sum = (weights[0]*fit2_N[0])+ (weights[1]*fit2_N[1])
    return fit1_sum, fit2_sum
def optimal_WMC_Co(fit_op):
    #print("optimal_WMC:",fit_op)
    weights = np.array([0.5,0.5])
    min_max = np.zeros(2)
    fit_sum = np.zeros(len(fit_op))
    fit_op1 = np.zeros((len(fit_op),2))
    for i in range(2):
        if(i==0):
            min_max[i] = min(fit_op[:,i])
        else:
            min_max[i] = max(fit_op[:,i])
    for i in range(2):
        if(i==0):
            tmp_val = min_max[i]/fit_op[:,i]
            fit_op1[:,i] = tmp_val
        else:
            tmp_val = fit_op[:,i]/min_max[i]
            fit_op1[:,i] = tmp_val
    for i in range(len(fit_op)):
         fit_sum[i] = (weights[0]*fit_op1[i][0])+ (weights[1]*fit_op1[i][1])
    #print("sum:",fit_sum)
    index_op = np.argmax(fit_sum)
    #print(fit[index])
    return index_op

def CoRSNSoptimization(n_var,args, n_mf, n_outputs):
    nPop = 80
    itr = 30
    nVar = n_var
    X = args[0]
    A_LB = 0.1
    A_UB = 1
    C_LB = 0.2
    C_UB = 4
    #Y = args[1]
    #pop = np.zeros([nPop, nVar])
    maxv = np.amax(X, axis=0)
    minv = np.amin(X, axis=0)
    meanv = np.mean(X, axis=0)
    """for i in range(nPop):
        ran = np.random.uniform(0,1,nVar)
        #pop[i] = minv + ran * (maxv - minv)
        pop[i] = ran * meanv  """
    #----------------initial pop -------------------
    #pop = LB + np.random.rand(nPop, nVar) * (UB - LB)
    pop = np.random.rand(nPop, nVar)
    #pop = np.random.uniform(0, 10, size=(nPop, nVar))
    """for i in range(nPop):
        for j in range(nVar):
            tmp = random.uniform(LB[j],UB[j])
            pop[i][j]= tmp if (pop[i][j]<LB[j] or pop[i][j]>UB[j]) else pop[i][j] """
    #pop_mean = np.mean(pop, axis=0)
    # --------------Calculate fitness array------------------
    fit = np.zeros((nPop, 2))
    fitness_all_msr = np.zeros((nPop, 7))
    #print("population:")
    pp_cnt = 3 * X.shape[1]
    for i in range(nVar):
      if (i<pp_cnt):
        alb = A_LB
        aub = A_UB
      else:
        alb = C_LB
        aub = C_UB
      pop[:,i] = np.interp(pop[:,i], (pop[:,i].min(), pop[:,i].max()), (alb, aub))
    for i in range(nPop):
      fit[i], fitness_all_msr[i] = create_model(pop[i], args, n_mf, n_outputs)
    #print("fitness: ", fit)
    #print(fit)
    # ----------------INITIALIZATION Ends------------------------------
    mood = np.zeros(4)
    case = 1
    for loop in range(itr):
        #print("loop:", loop)
        for i in range(nPop):
            #case = random.randint(1, 4)
            if (loop == 0 or loop == 1 or loop ==2 or loop ==3):
                casev = random.randint(1, 4)
            else:
                casev = np.argmax(mood) + 1
            #imitation
            if casev == 1:
                temp = random.randint(0,nPop-1)
                v = pop[temp] - pop[i]
                r = np.random.uniform(0, 1, nVar)
                R = r * v
                r2 = np.random.uniform(-1, 1, nVar)
                new_X = pop[temp] + (r2 * R)
                new_X[pp_cnt:] = np.interp(new_X[pp_cnt:], (new_X[pp_cnt:].min(), new_X[pp_cnt:].max()), (C_LB, C_UB))
                new_X[:pp_cnt] = np.interp(new_X[:pp_cnt], (new_X[:pp_cnt].min(), new_X[:pp_cnt].max()), (A_LB, A_UB))
                """for j in range(nVar):
                    tmp = random.uniform(LB[j],UB[j])
                    new_X[j]= tmp if (new_X[j]<LB[j] or new_X[j]>UB[j]) else new_X[j]"""
                fit_up, all_msr = create_model(new_X, args, n_mf, n_outputs)
                fit1, fit2= WSM_Co(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
                if fit2 >  fit1:
                    #print("update", fit_up)
                    mood[0]= mood[0] + 1
                    pop[i] = new_X
                    fit[i] = fit_up
                    fitness_all_msr[i] = all_msr
                else:
                    mood[0] = mood[0]-1
                #print("case1:",fit[i])

            #Conversation
            if casev == 2:
                ran_jC = random.randint(0, nPop - 1)
                ran_kC = random.randint(0, nPop - 1)
                #print(fit[i][0],fit[ran_jC][0])
                D = (np.sign(fit[i][0] - fit[ran_jC][0])) * (pop[ran_jC] - pop[i])
                #D =  (pop[ran_jC] - pop[i])
                v_C = np.random.uniform(0, 1, nVar)
                R_C = v_C * D
                new_X = pop[ran_kC] + R_C
                new_X[pp_cnt:] = np.interp(new_X[pp_cnt:], (new_X[pp_cnt:].min(), new_X[pp_cnt:].max()), (C_LB, C_UB))
                new_X[:pp_cnt] = np.interp(new_X[:pp_cnt], (new_X[:pp_cnt].min(), new_X[:pp_cnt].max()), (A_LB, A_UB))
                """for j in range(nVar):
                    tmp = random.uniform(LB[j],UB[j])
                    new_X[j]= tmp if (new_X[j]<LB[j] or new_X[j]>UB[j]) else new_X[j]"""
                #print("updated population:2")
                #print(new_X)
                fit_up, all_msr =  create_model(new_X, args, n_mf, n_outputs)
                #print(fit[i],fit_up)
                #print(fit_up)
                fit1, fit2= WSM_Co(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
                if fit2 >  fit1:
                    #print("update", fit_up)
                    mood[1]= mood[1] + 1
                    pop[i] = new_X
                    fit[i] = fit_up
                    fitness_all_msr[i] = all_msr
                else:
                    mood[1] = mood[1]-1
                #print("case2:",fit[i])

            #Disputation
            if casev == 3:
                commentors = random.randint(1, nPop)
                friend = np.random.randint(0, nPop - 1, commentors)
                M = np.sum(pop[friend],axis=0)/commentors
                v_D = np.random.uniform(0, 1, nVar)
                r_D = random.uniform(1, 2)
                AF = 1 + round(r_D)
                AF = AF * pop[i]
                M = M - AF
                M = v_D * M
                new_X = pop[i] + M
                new_X[pp_cnt:] = np.interp(new_X[pp_cnt:], (new_X[pp_cnt:].min(), new_X[pp_cnt:].max()), (C_LB, C_UB))
                new_X[:pp_cnt] = np.interp(new_X[:pp_cnt], (new_X[:pp_cnt].min(), new_X[:pp_cnt].max()), (A_LB, A_UB))
                """for j in range(nVar):
                    tmp = random.uniform(LB[j],UB[j])
                    new_X[j]= tmp if (new_X[j]<LB[j] or new_X[j]>UB[j]) else new_X[j]"""
                #print("updated population:3")
                #print(new_X)
                fit_up, all_msr = create_model(new_X, args, n_mf, n_outputs)
                #print(fit[i],fit_up)
                #print(fit_up)
                fit1, fit2= WSM_Co(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
                if fit2 >  fit1:
                    #print("update", fit_up)
                    mood[2]= mood[2] + 1
                    pop[i] = new_X
                    fit[i] = fit_up
                    fitness_all_msr[i] = all_msr
                else:
                    mood[2] = mood[2]-1
                #print("case3:",fit[i])
            #Innovation
            if casev == 4:
                d = random.randint(0,nVar-1)
                lb = np.amax(pop, axis=0)[d] #min[d]
                ub = np.amax(pop, axis=0)[d] #max[d]
                j_In = random.randint(0,nPop-1)
                r1_In = random.uniform(0,1)
                r2_In = random.uniform(0,1)
                neww = lb + r1_In * (ub-lb)
                new_X = pop[i]
                new_X[d] = (r2_In * new_X[d]) + ((1-r2_In) * neww)
                new_X[pp_cnt:] = np.interp(new_X[pp_cnt:], (new_X[pp_cnt:].min(), new_X[pp_cnt:].max()), (C_LB, C_UB))
                new_X[:pp_cnt] = np.interp(new_X[:pp_cnt], (new_X[:pp_cnt].min(), new_X[:pp_cnt].max()), (A_LB, A_UB))
                """for j in range(nVar):
                    tmp = random.uniform(LB[j], UB[j])
                    new_X[j]= tmp if (new_X[j] < LB[j] or new_X[j] > UB[j]) else new_X[j]"""
                #print("updated population:4")
                #print(new_X)
                fit_up, all_msr = create_model(new_X, args, n_mf, n_outputs)
                #print(fit[i],fit_up)
                #print(fit_up)
                fit1, fit2= WSM_Co(fit[i], np.array(fit_up)) #fit1 for old, fit2 for updated one
                if fit2 >  fit1:
                    #print("update", fit_up)
                    mood[3]= mood[3] + 1
                    pop[i] = new_X
                    fit[i] = fit_up
                    fitness_all_msr[i] = all_msr
                else:
                    mood[3] = mood[3]-1
                #print("case4:",fit[i])
        index = optimal_WMC_Co(fit)
        #print("afeter optimal:",fit)
        #print("optimal values: ", index, fit[index], fitness_all_msr[index],pop[index])
    return pop[index], index, fit[index], fitness_all_msr[index]

#Anfis model for testing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def info_anfis(n_mf, n_outputs):
    """
    Returns number of premise functions <n_pf>, number of consequent functions
    <n_cf>, and number of variables <n_var> for the ANFIS defined by <n_mf>
    and <n_outputs>.
    """
    n_mf = np.asarray(n_mf)

    n_pf = n_mf.sum()
    n_cf = n_mf.prod()
    n_var = 3 * n_pf + (len(n_mf) + 1) * n_cf * n_outputs

    return n_pf, n_cf, n_var

def find_k_nearest_index(X, k):
    n, m = np.array(X).shape
    temp_index = []
    for i in range(m):
        temp_index.append(np.argsort(X[:, i])[:k])
    return np.array(temp_index).transpose()

def ANFIS_model_testing(X_test, X_train, y_train):
    mu_delta = 0.2
    s_par = [0.5, 0.2]
    c_par = [0.3, 0.4]
    A_par = [-10.0, 10.0]
    n_samples, n_cols = X_train.shape
    #print(n_samples, n_cols)
    #n_mf = list(np.ones(n_cols,dtype=int))
    n_mf = list(np.repeat(2, n_cols))
    n_outputs = 1
    test_len = len(X_test)
    train_len = len(X_train)
    #print(train_len, test_len)
    # ANFIS info
    n_pf, n_cf, n_var = info_anfis(n_mf, n_outputs)
    #print(n_pf, n_cf, n_var)
    nn_k = int((50*train_len)/100)
    #print("train length: ", nn_k, train_len)
    X_test = X_test.reshape(1,-1)
    distance = cdist(X_train, X_test, metric='euclidean')
    index = find_k_nearest_index(distance, k = nn_k)
    train_x = X_train[index[:,0]]
    train_y = y_train[index[:,0]]
    #train_x = X_train
    #train_y = y_train
    args = (train_x, train_y)
    optimal_soln, ind_opti, fit_opti, all_msr_vals = CoRSNSoptimization(n_var, args, n_mf, n_outputs)
    print("Solution:")
    print("optimal fitness values = ", fit_opti)
    print("optimal all measures = ", all_msr_vals)
    print("optimal learner = ", optimal_soln)
    Yp_te = eval_data(X_test, optimal_soln, n_mf, n_outputs)
    return Yp_te

#BiRSNS feature selection
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, Ridge, ElasticNet, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

def tanh(num):
    return (np.exp(num)-np.exp(-num))/(np.exp(num)+np.exp(-num))
def sigmoid(num):
    return 1/(1 + np.exp(-num))  # return (2*s)-1
def find_k_nearest_index(X, k):
    n, m = np.array(X).shape
    temp_index = []
    for i in range(m):
        temp_index.append(np.argsort(X[:, i])[:k])
    return np.array(temp_index).transpose()

def calc_mmre(a, b):
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()
    mre = abs(a-b)/a
    mmre = np.sum(mre)/len(a)
    return mmre
def calc_pred(a, b):
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()
    mre = abs(a-b)/a
    D = [1 if i<=0.25 else 0 for i in mre]
    pred = np.sum(D)/len(a)
    return pred

def fitness_func(features, X, Y):
    ind = np.where(features==1)[0]
    X = X[:, ind]
    total_mmre = 0
    total_pred = 0
    total_msr = np.zeros(7)
    itr = 5
    for x in range(itr):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.30)
        l_reg = LinearRegression()
        #l_reg = DecisionTreeRegressor(max_depth=2)
        #l_reg = SVR
        #l_reg = Lasso(alpha=0.1)
        #l_reg = RandomForestRegressor(max_depth=3)
        l_reg.fit(X_train, y_train)  # fitting regression
        y_pred = l_reg.predict(X_test)
        total_msr += calc_msr(y_test, y_pred, len(ind))
        #total_mmre += calc_mmre(y_test, y_pred)
        #total_pred += calc_pred(y_test, y_pred)
        total_mmre += mean_absolute_error(y_test, y_pred)
        #total_pred += r2_score(y_test, y_pred)
        total_pred += calc_adjR2(y_test, y_pred, X_train.shape[1])
        arr = (np.round(total_mmre / itr, 4), np.round(total_pred / itr, 4), len(ind))
    return arr, np.round(total_msr/itr,4)

def WSM(fit1,fit2):
    weights = np.array([0.4,0.4,0.2])
    #print(fit1,fit2)
    min_max = np.array([min(fit1[0],fit2[0]), max(fit1[0],fit2[0]), min(fit1[2],fit2[2])])
    fit1_N = np.array([min_max[0]/fit1[0], fit1[1]/min_max[1], min_max[2]/fit1[2]])
    fit2_N = np.array([min_max[0]/fit2[0], fit2[1]/min_max[1], min_max[2]/fit2[2]])
    fit1_sum = (weights[0]*fit1_N[0])+ (weights[1]*fit1_N[1]) + (weights[2]*fit1_N[2])
    fit2_sum = (weights[0]*fit2_N[0])+ (weights[1]*fit2_N[1]) + (weights[2]*fit2_N[2])
    return fit1_sum, fit2_sum

def optimal_WMC(fit_op):
    #print("optimal_WMC:",fit_op)
    weights = np.array([0.4,0.4,0.2])
    min_max = np.zeros(3)
    fit_sum = np.zeros(len(fit_op))
    fit_op1 = np.zeros((len(fit_op),3))
    for i in range(3):
        if(i==0 or i==2):
            min_max[i] = min(fit_op[:,i])
        else:
            min_max[i] = max(fit_op[:,i])
    for i in range(3):
        if(i==0 or i==2):
            tmp_val = min_max[i]/fit_op[:,i]
            fit_op1[:,i] = tmp_val
        else:
            tmp_val = fit_op[:,i]/min_max[i]
            fit_op1[:,i] = tmp_val
    for i in range(len(fit_op)):
         fit_sum[i] = (weights[0]*fit_op1[i][0])+ (weights[1]*fit_op1[i][1]) + (weights[2]*fit_op1[i][2])
    #print("sum:",fit_sum)
    index_op = np.argmax(fit_sum)
    #print(fit[index])
    return index_op

def BiRSNSoptimization(X_train, Y_train):
  pop_size = 20
  iterations = 5
  pop = np.zeros((pop_size, cols))
  fitness_all_msr = np.zeros((pop_size, 7))
  maxv = np.amax(X_train, axis=0)
  minv = np.amin(X_train, axis=0)
  meanv = np.mean(X_train, axis=0)
  for i in range(pop_size):
      ran = np.random.uniform(0,1,cols)
      rand_a  = minv + ran * (maxv - minv)
      rand_a /= rand_a.sum()
      pop[i] = rand_a
  pop_mean = np.mean(pop, axis=0)
  popB = np.zeros((pop_size,cols))
  for i in range(pop_size):
      for j in range(cols):
          #pop[i][j] = random.uniform(0,0.5) if pop[i][j]<pop_mean[j] else random.uniform(0.5,1)
          #pop[i][j] = sigmoid(pop[i][j])
          rand_in =  random.uniform(0, 1)
          popB[i][j] = 1 if rand_in < pop[i][j] else 0
  fit = np.zeros((pop_size,3))
  for i in range(pop_size):
      if (not popB[i].any()):   #when all zeros in popB
          popB [i] = np.random.randint(2, size=cols)
          #popB[i] = np.random.choice([0, 1], size=cols, p=[1./3, 2./3])
      fit[i], fitness_all_msr[i] = fitness_func(popB[i], X_train, Y_train)
  #print(fit)

  mood = np.zeros(4) #mood ranks, initially zeros
  new_B = np.zeros(cols)
  #casev = 1
  for loop in range(iterations):
      #print("loop:", loop)
      for i in range(pop_size):
          #case = random.randint(1, 4)
          if (loop == 0 or loop == 1 or loop ==2 or loop ==3):
              casev = random.randint(1, 4)
          else:
              casev = np.argmax(mood) + 1
          # imitation
          #print("mood number: ", casev)
          if casev == 1:
              temp = random.randint(0,pop_size-1)
              v = pop[temp] - pop[i]
              r = np.random.uniform(0, 1, cols)
              R = r * v
              r2 = np.random.uniform(-1, 1, cols)
              new_X = pop[temp] + (r2 * R)
              for j in range(cols):
                  prob = (1 + (math.sin(2*3.14*new_X[j])))/2
                  new_B[j] = 1-popB[i,j] if  random.uniform(0,1) <= prob else popB[i,j]
              if (not new_B.any()):
                  new_B = np.random.randint(2, size=cols)
                  #new_B = np.random.choice([0, 1], size=cols, p=[1./3, 2./3])
              fit_up, fitness_all_msr_up = fitness_func(new_B, X_train, Y_train)
              #print(fit_up)
              fit1, fit2= WSM(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
              if fit2 >  fit1:
                  #print("update", fit_up)
                  mood[0]= mood[0] + 1
                  pop[i] = new_X
                  popB[i] = new_B
                  fit[i] = fit_up
                  fitness_all_msr[i] = fitness_all_msr_up
              else:
                  mood[0] = mood[0]-1
              #print("case1:",fit[i])

          #Conversation
          if casev == 2:
              ran_jC = random.randint(0, pop_size - 1)
              ran_kC = random.randint(0, pop_size - 1)
              D = (np.sign(fit[i][0] - fit[ran_jC][0])) * (pop[ran_jC] - pop[i])
              v_C = np.random.uniform(0, 1, cols)
              R_C = v_C * D
              new_X = pop[ran_kC] + R_C
              for j in range(cols):
                  prob = (1 + (math.sin(2*3.14*new_X[j])))/2
                  new_B[j] = 1-popB[i,j] if  random.uniform(0,1) <= prob else popB[i,j]
              if (not new_B.any()):
                  new_B = np.random.randint(2, size=cols)
                  #new_B = np.random.choice([0, 1], size=cols, p=[1./3, 2./3])
              fit_up, fitness_all_msr_up = fitness_func(new_B, X_train, Y_train)
              #print(fit_up)
              fit1, fit2= WSM(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
              if fit2 >  fit1:
                  #print("update", fit_up)
                  mood[1]= mood[1] + 1
                  pop[i] = new_X
                  popB[i] = new_B
                  fit[i] = fit_up
                  fitness_all_msr[i] = fitness_all_msr_up
              else:
                  mood[1] = mood[1]-1
              #print("case2:",fit[i])
          #Disputation
          if casev == 3:
              commentors = random.randint(1, pop_size)
              friend = np.random.randint(0, pop_size - 1, commentors)
              M = np.sum(pop[friend],axis=0)/commentors
              v_D = np.random.uniform(0, 1, cols)
              r_D = random.uniform(1, 2)
              AF = 1 + round(r_D)
              AF = AF * pop[i]
              M = M - AF
              M = v_D * M
              new_X = pop[i] + M
              for j in range(cols):
                  prob = (1 + (math.sin(2*3.14*new_X[j])))/2
                  new_B[j] = 1-popB[i,j] if  random.uniform(0,1) <= prob else popB[i,j]
              if (not new_B.any()):
                  new_B = np.random.randint(2, size=cols)
                  #new_B = np.random.choice([0, 1], size=cols, p=[1./3, 2./3])
              fit_up, fitness_all_msr_up = fitness_func(new_B, X_train, Y_train)
              #print(fit_up)
              fit1, fit2= WSM(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
              if fit2 >  fit1:
                  #print("update", fit_up)
                  mood[2]= mood[2] + 1
                  pop[i] = new_X
                  popB[i] = new_B
                  fit[i] = fit_up
                  fitness_all_msr[i] = fitness_all_msr_up
              else:
                  mood[2] = mood[2]-1
              #print("case3:",fit[i])
          #Innovation
          if casev == 4:
              d = random.randint(0,cols-1)
              lb = np.amax(pop, axis=0)[d] #min[d]
              ub = np.amax(pop, axis=0)[d] #max[d]
              j_In = random.randint(0,pop_size-1)
              r1_In = random.uniform(0,1)
              r2_In = random.uniform(0,1)
              neww = lb + r1_In * (ub-lb)
              new_X = pop[i]
              if (not new_B.any()):
                  new_B = np.random.randint(2, size=cols)
              new_X[d] = (r2_In * new_X[d]) + ((1-r2_In) * neww)
              rand_in = np.random.uniform(0,1,cols)
              for j in range(cols):
                  prob = (1 + (math.sin(2*3.14*new_X[j])))/2
                  new_B[j] = 1-popB[i,j] if  random.uniform(0,1) <= prob else popB[i,j]
              if (not new_B.any()):
                  new_B = np.random.randint(2, size=cols)
                  #new_B = np.random.choice([0, 1], size=cols, p=[1./3, 2./3])
              fit_up, fitness_all_msr_up = fitness_func(new_B, X_train, Y_train)
              #print(fit_up)
              fit1, fit2= WSM(fit[i],np.array(fit_up)) #fit1 for old, fit2 for updated one
              if fit2 >  fit1:
                  #print("update", fit_up)
                  mood[3]= mood[3] + 1
                  pop[i] = new_X
                  popB[i] = new_B
                  fit[i] = fit_up
                  fitness_all_msr[i] = fitness_all_msr_up
              else:
                  mood[3] = mood[3]-1
              #print("case4:",fit[i])
      #print("before optimal:",fit)
      index = optimal_WMC(fit)
  #print("afeter optimal:",fit)
  print("optimal values: ", index, fit[index], fitness_all_msr[index],pop[index], popB[index])
  return popB[index]
"""#SNS-ANFIS  on testing data
ind1 = np.where(ind==1)[0]
print(ind1)
lent = len(ind1)
#lent = col1
X_test = X_test[:, ind1]
X_train = X_train[:, ind1]
test_len = len(X_test)
train_len = len(X_train)
pred = anf_main.Anfis_main(X_train, X_test, Y_train, Y_test)"""

#measures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
def calc_adjR2(Yactual, Yestimated, number_of_features):
  a= Yactual.flatten()
  b = Yestimated.flatten()
  #print("lekns: ",len(a))
  rowa = len(a)
  cola = number_of_features
  r2 = r2_score(a, b)
  tmp = (1-r2)*(rowa-1)
  tmp = tmp/(rowa-cola-1)
  adj_r2 = 1 - tmp
  return adj_r2
def calc_sa(Yactual, Yestimated):
  a= Yactual.flatten()
  b = Yestimated.flatten()
  mae = mean_absolute_error(a,b)
  marp = np.mean(abs(a-b))/len(a)
  sa = 1 - (mae/marp)
  return sa
def calc_msr(Yactual, Yestimated, number_of_features):
  a= Yactual.flatten()
  b = Yestimated.flatten()
  #print("lekns: ",len(a))
  rowa = len(a)
  cola = number_of_features
  #rmse
  rmse = sqrt(mean_squared_error (a, b))
  #mae
  mae = mean_absolute_error(a,b)
  #mape
  mape = np.square(np.mean(abs((a - b) / a)))
  #mmre
  mre = abs(a-b)/a
  #print("mre: ", mre)
  mmre = np.mean(mre)
  #mdmre
  mdmre = np.median(mre)
  #bmmre
  tmp1 = abs(a-b)
  tmp2 = np.minimum(a,b)
  tmp = tmp1/tmp2
  bmmre = np.mean(tmp)
  #pred
  D = [1 if i<=0.25 else 0 for i in mre]
  #print("D: ", D)
  pred = np.sum(D)/len(a)
  #adj-r2
  r2 = r2_score(a, b)
  tmp = (1-r2)*(rowa-1)
  tmp = tmp/(rowa-cola-1)
  adj_r2 = 1 - tmp
  #sa
  marp = np.mean(abs(a-b))/len(a)
  sa = 1 - (mae/marp)
  return rmse, mae, mmre, mdmre, bmmre, pred, adj_r2

import numpy as np
import pandas as pd
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import random

data = pd.read_csv(r'/content/miyazaki.csv')
rows, col = data.shape
print(rows, col)
cols = col -1
data = data.values
X = data[:,:-1]
Y = data[:,-1]
#X= X- np.min(X, axis = 0) / (np.max(X, axis = 0) - np.min(X, axis = 0))
#Normalization
#for column in range(cols):
#    X[:,column] = (X[:,column] - X[:,column].min()) / (X[:,column].max() - X[:,column].min())
#mni_max = np.max(Y)-np.min(Y)
#Y = (Y - np.min(Y) ) / mni_max
#3-cross validation
numbers = list(range(rows))
random.shuffle(numbers)
set_size = len(numbers) // 3
set1 = numbers[:set_size]
X_set1 = X[set1, :]
Y_set1 = Y[set1]
set2 = numbers[set_size:2*set_size]
X_set2 = X[set2, :]
Y_set2 = Y[set2]
set3 = numbers[2*set_size:]
X_set3 = X[set3, :]
Y_set3 = Y[set3]


testing_mesrs = np.zeros(7) #number of measures

"""estimated_effort = np.zeros(len(Y))
for i in range(len(X)):
  X_train = X
  Y_train = Y
  X_test = X[i]
  Y_test = Y[i]
  X_train = np.delete(X_train, i, 0)
  Y_train = np.delete(Y_train, i, 0)
  optimized_features = BiRSNSoptimization(X_train, Y_train)
  indx = np.where(optimized_features == 1)[0]
  print("each test: ", i, indx)
  X_train = X_train[:, indx]
  X_test = X_test[indx]
  estimated_effort[i] = ANFIS_model_testing(X_test, X_train, Y_train)
testing_mesrs = calc_msr(Y, estimated_effort, len(indx))
print(np.round(testing_mesrs,4))

"""
indx = [0]
#fold 1
X_train_data = np.vstack((X_set1, X_set2))
Y_train_data = np.hstack((Y_set1, Y_set2))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold1 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set3[:, indx]
Y_test_data = Y_set3
test_data = np.column_stack((X_test_data, Y_test_data))
estimated_effort = np.zeros(len(Y_test_data))
#estimated_effort_try = np.zeros(len(Y_test_data))
for i in range(len(X_test_data)):
  print("\nfold1 project: ", i)
  estimated_effort[i] = ANFIS_model_testing(X_test_data[i], X_train_data, Y_train_data)
  #estimated_effort_try[i] = train_sum + estimated_effort[i]
fold1_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold1 measures: ", fold1_mesrs)
testing_mesrs += fold1_mesrs

#fold 2
X_train_data = np.vstack((X_set1, X_set3))
Y_train_data = np.hstack((Y_set1, Y_set3))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold2 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set2[:, indx]
Y_test_data = Y_set2
test_data = np.column_stack((X_test_data, Y_test_data))
estimated_effort = np.zeros(len(Y_test_data))
for i in range(len(X_test_data)):
  print("\nfold2 project: ", i)
  estimated_effort[i] = ANFIS_model_testing(X_test_data[i], X_train_data, Y_train_data)
  #estimated_effort_try[i] = train_sum + estimated_effort[i]
fold2_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold2 measures: ", fold2_mesrs)
testing_mesrs += fold2_mesrs

#fold 3
X_train_data = np.vstack((X_set2, X_set3))
Y_train_data = np.hstack((Y_set2, Y_set3))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold3 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set1[:, indx]
Y_test_data = Y_set1
test_data = np.column_stack((X_test_data, Y_test_data))
estimated_effort = np.zeros(len(Y_test_data))
#opti_sl = np.array([0.88244772,1.10359605,0.65555995,9.17350184,23.52336747,17.60813814,24.53550386,21.41144928,8.26589005,3.07555565])
for i in range(len(X_test_data)):
  print("\nfold3 project: ", i)
  estimated_effort[i] = ANFIS_model_testing(X_test_data[i], X_train_data, Y_train_data)
  #estimated_effort_try[i] = train_sum + estimated_effort[i]
fold3_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold3 measures: ", fold3_mesrs)
testing_mesrs += fold3_mesrs

print("final measure values:")
print(np.round(testing_mesrs/3,4))

import numpy as np
import pandas as pd
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import random

data = pd.read_csv(r'/content/kitchenham.csv')
rows, col = data.shape
print(rows, col)
cols = col -1
data = data.values
X = data[:,:-1]
Y = data[:,-1]
#X= X- np.min(X, axis = 0) / (np.max(X, axis = 0) - np.min(X, axis = 0))
#Normalization
#for column in range(cols):
#    X[:,column] = (X[:,column] - X[:,column].min()) / (X[:,column].max() - X[:,column].min())
#mni_max = np.max(Y)-np.min(Y)
#Y = (Y - np.min(Y) ) / mni_max
#3-cross validation
numbers = list(range(rows))
random.shuffle(numbers)
set_size = len(numbers) // 3
set1 = numbers[:set_size]
X_set1 = X[set1, :]
Y_set1 = Y[set1]
set2 = numbers[set_size:2*set_size]
X_set2 = X[set2, :]
Y_set2 = Y[set2]
set3 = numbers[2*set_size:]
X_set3 = X[set3, :]
Y_set3 = Y[set3]

testing_mesrs = np.zeros(7)
indx = [2]
#fold 1
X_train_data = np.vstack((X_set1, X_set2))
Y_train_data = np.hstack((Y_set1, Y_set2))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold1 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set3[:, indx]
Y_test_data = Y_set3
args = (X_train_data , Y_train_data)
n_mf = list(np.repeat(2, X_train_data.shape[1]))
n_outputs = 1
n_pf, n_cf, n_var = info_anfis(n_mf, n_outputs)
opti_sln, ind_opti, fit_opti, all_msr_vals = CoRSNSoptimization(n_var, args, n_mf, n_outputs)
estimated_effort = eval_data(X_test_data,opti_sln, n_mf, n_outputs)
fold1_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold1 measures: ", fold1_mesrs)
testing_mesrs += fold1_mesrs

#fold 2
X_train_data = np.vstack((X_set1, X_set3))
Y_train_data = np.hstack((Y_set1, Y_set3))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold2 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set2[:, indx]
Y_test_data = Y_set2
args = (X_train_data , Y_train_data)
n_mf = list(np.repeat(2, X_train_data.shape[1]))
n_outputs = 1
n_pf, n_cf, n_var = info_anfis(n_mf, n_outputs)
opti_sln, ind_opti, fit_opti, all_msr_vals = CoRSNSoptimization(n_var, args, n_mf, n_outputs)
estimated_effort = eval_data(X_test_data,opti_sln, n_mf, n_outputs)
fold2_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold2 measures: ", fold2_mesrs)
testing_mesrs += fold2_mesrs

#fold 3
X_train_data = np.vstack((X_set2, X_set3))
Y_train_data = np.hstack((Y_set2, Y_set3))
train_data = np.column_stack((X_train_data, Y_train_data))
#optimized_features = BiRSNSoptimization(X_train_data, Y_train_data)
#indx = np.where(optimized_features == 1)[0]
print("\nfold3 returned feature set: ", indx)
X_train_data = X_train_data[:, indx]
Y_train_data = Y_train_data
X_test_data = X_set1[:, indx]
Y_test_data = Y_set1
args = (X_train_data , Y_train_data)
n_mf = list(np.repeat(2, X_train_data.shape[1]))
n_outputs = 1
n_pf, n_cf, n_var = info_anfis(n_mf, n_outputs)
opti_sln, ind_opti, fit_opti, all_msr_vals = CoRSNSoptimization(n_var, args, n_mf, n_outputs)
estimated_effort = eval_data(X_test_data,opti_sln, n_mf, n_outputs)
fold3_mesrs = calc_msr(Y_test_data, estimated_effort, len(indx))
print("fold3 measures: ", fold3_mesrs)
testing_mesrs += fold3_mesrs

print("final measure values:")
print(np.round(testing_mesrs/3,4))