# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:27:12 2020

@author: iver_
"""
import numpy as np
import scipy.io as io

class SimulationModel:
    """
    Template class for an arbitrary model used to simulate the real plant to test controller.
    Should be more sophisticated than the blackboxmodel type.
    """
    
    def __init__(self):
        pass
    
    def model_output(self,u):
        """
        outputs the model given an input, should return the model output. Note that
        the states should be an internal variable.
        :param u: input to the system. Dimension: 1d array of size PNMPCController.n_mv
        :type u: numpy.ndarray
        :returns: one step prediction given u and internal state. Dimension: 1darray of size PNMPCController.n_out
        :rtype: numpy.ndarray
        """
        pass


class BlackBoxModel:
    """
    Template class of the black box model. Interacts with the PNMPC Predictor through association.
    """

    def __init__(self):
        pass

    def get_current_state(self):
        """
        template function that should return the current model state
        """
        pass

    def set_current_state(self, a):
        """
        Template function that should be able to arbitrariely define the state
        :param a: The new values for the state vector.
        :type a: numpy.ndarray
        """
        pass

    def get_n_out(self):
        """
        Template function that should return the number of output variables in the identification model.
        """
        return 1

    def get_n_in(self):
        """
        Template function that should return the number of input variables in the identification model.
        """
        return 1

    def update(self, u):
        """
        Template function of the black-box model one-step update. Necessary for the PNMPC free response computation.
        Function should return output at one time step
        :param u: The control action input to the system. Dimension: (n_mv x 1) or 1d
        :type u: numpy.ndarray
        """
        pass

    def get_dydu(self, u, y):
        """
        Template function that should return the gradient between the output and the control action.
        """
        pass

def sparsity(M, psi):
    """
    Explointing the point to point multiplication operator of python, this function
    merely turns a random number of elements in a matrix into zero.
    the probability of an element being zero is psi, and M is the input matrix.
    """
    N = np.empty_like(M)
    for linha in range(len(N)):
        for coluna in range(len(N[linha])):
            prob = np.random.rand()
            if prob < psi:
                N[linha][coluna] = 0
            else:
                N[linha][coluna] = 1


    return N*M

def RFRAS(min,max,num_steps,minimum_step):
    """
    Actually the APRBS computation.
    :param min: low-end of the values the curve can assume.
    :type min: float
    :param max: high end of the values the curve can assume.
    :type max: float
    :param num_steps: size in time-steps of the signal.
    :type num_steps: float
    :param minimum_step: Minimum number of steps the signal will hode one values.
    :returns: The whole aprbs signal (1d array of size num_steps)
    :rtype: numpy.ndarray
    """
    RFRAS_sig = np.empty(num_steps)
    val = min + (max - min)*np.random.rand()
    for i in range(num_steps):

        if i % minimum_step  == 0:
            val = change_or_not(val,min,max)


        RFRAS_sig[i] = val

    return RFRAS_sig


def change_or_not(x,min_val,max_val):
    y = 0
    p_change = np.random.rand()
    if p_change < 0.5:
        y = x
    else:
        y = min_val + (max_val - min_val)*np.random.rand()
    return y


def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x 	 
    return y
 

def feature_descaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = x*(xmax-xmin) + xmin 
    else:
        y = x
    return y


def grad_tanh(z):
    """
    Gradient of the elementwise atan.
    """
    
    
    return np.diag(1 - np.tanh(z.flatten())**2)


def reg_least_squares(X,Y,reg): 
    """
    Analytical response to the least squares. Returns a matrix of collumn vectors. Internal function to the ESN input.
    :param X: Features/inputs of the model. dimension: (number of training data x number of features)
    :type X: numpy.ndarray
    :param Y: Real outputs of the system to fit the model. Dimension: (number of training data x number of outputs).
    :type Y: numpy.ndarray
    :param reg: Regularization parameter. Scalar.
    :type reg: float.
    :returns: The training weights. Dimension: (number of features x number of outputs)
    :rtype: numpy.ndarray
    """
        
        
    P = np.dot(X.T,X)
    R = np.dot(X.T,Y)
    theta = np.linalg.solve(P+reg*np.eye(P.shape[0],P.shape[1]),R)               
        
        
    return theta



class EchoStateNetwork(BlackBoxModel):
    """
    Echo State Network class, child of the BlackBoxModel class.
    """

    def __init__(self,neu,n_in,n_out,
                 gama=0.5,ro=1,psi=0.5,in_scale=0.1,
                 bias_scale=0.5,alfa=10,forget = 1,
                 initial_filename="initial",
                 load_initial = False,save_initial = False,output_feedback = False, 
                 noise_amplitude = 0, 
                 out_scale = 0,default_warmupdrop=100):
        """
        The class initializes with the following parameters, all are scalars unless stated otherwise:
        :param neu: number of neurons.
        :param ro: spectral radius of the networks.
        :param psi: sparsity factor.
        :param in_scale: relative gan of the input to the reservoir and the reservoir to reservoir weights.
        :param bias_cale: relative gain of the bias.
        :param alfa: Initial Condition of the RLS algorithm.
        :param forget: forgetting factor of the RLS algorithm.
        :param initial_filename: file name where de weights of the network are to be saved.
        :type initial_filename: string
        :param load_initial: Will the weights be loaded from an already saved file?
        :type load_initial: bool
        :param save_initial: Do you wish to save the initial weights obtained randomly?
        :type save_initial: bool
        :output_feedback: will the network have feedback?
        :type output_feedback: bool
        :param noise_amplitude: if you desire the presence of state noise, this is the noise amplitude.
        :param out_scale: Scaling of the output feedback weights, in case output_feedback is enabled.
        :param default_warmupdrop: how much examples are to dropped each time data is acquired from a simulation.
        """
        #All matrixes are initialized under the normal distribution.
        print("initializing reservoir")
        self.neu = neu
        self.n_in = n_in
        self.n_out = n_out
        
        self.psi = psi #the network's sparcity, in 0 to 1 notation
        # Reservoir Weight matrix.
        self.Wrr0 = np.random.normal(0,1,[neu,neu])
        self.Wrr0 = sparsity(self.Wrr0, self.psi)
        # input-reservoir weight matrix
        self.Wir0 = np.random.normal(0,1,[neu,n_in])
        # bias-reservoir weight matrix
        self.Wbr0 = np.random.normal(0,1,[neu,1])

        self.Wor0 = np.random.normal(0,1,[neu,n_out])

        #self.Wbo = np.random.normal(0,1,[n_out,1])
        # reservoir-output weight matrix
        
        self.Wro = np.random.normal(0,1,[n_out,neu+1])
        #self.Wro = np.zeros([n_out,neu])

        self.leakrate = gama #the network's leak rate
        self.ro = ro #the network's desired spectral radius
        self.in_scale = in_scale #the scaling of Wir.
        self.bias_scale = bias_scale #the scaling of Wbr

        # learning rate of the Recursive Least Squares Algorithm
        self.alfa = alfa
        # forget factor of the RLS Algorithm
        self.forget = forget
        self.output_feedback = output_feedback

        #self.a = np.random.normal(0, 1, [neu, 1])
        self.a = np.zeros([neu, 1],dtype=np.float64)
        #save if save is enabled
        if save_initial:
            self.save_initial_fun(initial_filename)

        #load if load is enabled
        if load_initial:
            self.load_initial(initial_filename)

        # the probability of a memeber of the Matrix Wrr being zero is psi.
        self.Wrr = self.Wrr0
        #forcing Wrr to have ro as the maximum eigenvalue
        eigs = np.linalg.eigvals(self.Wrr)
        radius = np.abs(np.max(eigs))
        #normalize matrix
        self.Wrr = self.Wrr/radius
        #set its spectral radius to rho
        self.Wrr *= ro


        #scale tbe matrices
        self.Wbr = bias_scale*self.Wbr0
        self.Wir = in_scale*self.Wir0
        self.Wor = out_scale*self.Wor0
        
        
        #define the noise amplitude.
        self.noise = noise_amplitude


        #covariance matrix
        self.P = np.eye(neu+1)/alfa
        
        
        #variables related to the accumulation of data
        self.cum_data_input = np.array([]) #accumulated input data
        self.cum_data_output = np.array([]) #accumulated output data
        self.number_of_simulations = 0 # number of simulations
        self.simulations_start = [] #list that tracks each simulation start.
        self.default_warmupdrop = default_warmupdrop #default warumupdrop
        
        
    def get_n_out(self):
        """
        gets the number of outputs
        :returns: number of outputs in the network.
        :rtype: int
        """
        return self.n_out
    
    def get_n_in(self):
        """
        gets the number of inputs
        :returns: number of inputs in the network.
        :rtype: int
        """
        return self.n_in
    
    def get_current_state(self):
        """
        returns the current state of the network.
        :returns: network states, dimension (self.neu x 1)
        :rtype: numpy.ndarray
        """
        return self.a
    
    def set_current_state(self,a):
        """
        arbitrarierly defines the current network state.
        :param a: new network state, dimension: (self.neu x 1)
        :type a: numpy.ndarray
        """
        self.a = a

    def get_wro(self,n=0): 
        """
        returns a row of Wro given a certain output.
        :param n: outnput number desired.
        :type n: int
        :returns: Wro,row n.
        :rtype: numpy.ndarray
        """

        return self.Wro[n]

    def training_error(self,ref):
        """
        For RLS. given a desired output ref, computes the training error.
        """

        Ref = np.array(ref,dtype = np.float64)
        if self.n_out > 1:
            Ref = Ref.reshape(len(ref),1)

        e = np.dot(self.Wro,np.vstack((np.atleast_2d(1.0),self.a))) - Ref
        return e

    def train(self,ref):
        """
        RLS, NOT main training function. Given a certain desired output ref (dim self.n_out x 1),
        updates the training weights self.Wro.
        """
        e = self.training_error(ref)
        #the P equation step by step
        a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
        self.P = \
        self.P/self.forget - np.dot(np.dot(np.dot(self.P,a_wbias),a_wbias.T),self.P)/ \
        (self.forget*(self.forget + np.dot(np.dot(a_wbias.T,self.P),a_wbias)))
        #self.sigma_q = (1 - 1 / (self.K_a * self.neu)) * \
        #               self.sigma_q + \
        #               (1 - (1 - 1 / (self.K_a * self.neu))) * \
        #               (np.dot(np.dot(self.a.T, self.P), self.a)) \
        #               ** 2
        for saida in range(self.n_out):

            #Transpose respective output view..
            Theta = self.Wro[saida,:]
            Theta = Theta.reshape([self.neu+1,1])


            #error calculation
            Theta = Theta - e[saida]*np.dot(self.P,a_wbias)
            Theta = Theta.reshape([1,self.neu+1])

            self.Wro[saida,:] = Theta




    def update(self,inp,y_in = np.atleast_2d(0),training = False ):
        """
        State updating function of the network. Ignore y_in if you are not using output_feedback.
        :param inp: input to the network (self.n_in x 1) or 1d array (self.n_in)
        :type inp: numpy.ndarray
        :param y_in: output feedback, dim (self.n_out x 1).
        :type y_in: numpy.ndarray
        :param training: The function would like to know if you are using update to gather data for training (internal functions only).
        :type training: bool
        :returns: network output. 
        """
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
        Y_in = np.array(y_in)
        Y_in = Y_in.reshape(Y_in.size, 1)
        if (y_in == 0).all():
            Y_in = np.zeros([self.n_out,1])
        if Input.size == self.n_in:                
            z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
            if self.output_feedback:
                z += np.dot(self.Wor,Y_in)
            if self.noise > 0 and training:
                z += np.random.normal(0, self.noise, [self.neu, 1])                
            self.a = (1-self.leakrate)*self.a + self.leakrate*np.tanh(z)
            
            a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
            y = np.dot(self.Wro,a_wbias)
            return y
        else:
            raise ValueError("input must have size n_in")

    def copy_weights(self, outer_esn):
        """
        Receives weights from another ESN.
        """
        if self.Wro.shape == outer_esn.Wro.shape:
            self.Wro = np.copy(outer_esn.Wro)
            self.Wrr = np.copy(outer_esn.Wrr)
            self.Wir = np.copy(outer_esn.Wir)
            self.Wbr = np.copy(outer_esn.Wbr)
        else:
            print("shapes of the weights are not equal")

    def save_reservoir(self,fileName):
        """
        Save current reservoir into file named string fileName.
        """
        data = {}
        data['Wrr'] = self.Wrr
        data['Wir'] = self.Wir
        data['Wbr'] = self.Wbr
        data['Wro'] = self.Wro
        data['a0'] = self.a
        io.savemat(fileName,data)

    def load_reservoir(self,fileName):
        """
        Load reservoir from fileName.
        """
        data = {}
        io.loadmat(fileName, data)
        self.load_reservoir_from_array(data)

    def load_reservoir_from_array(self, data):
        """
        Load reservoir given a dictionary with the following keys:
        Wrr for reservoir to reservoir weights.
        Wir for input to reservoir weights.
        Wbr for bias to reservoir weights.
        Wro for reservoir to output weights.
        """
        self.Wrr = data['Wrr']
        self.Wir = data['Wir']
        self.Wbr = data['Wbr']
        self.Wro = data['Wro']
        if 'a0' in data:  # check by Eric
            self.a = data['a0']
            
        # added by Eric - start
        if 'Wro_b' in data:
            self.Wro = np.hstack((data['Wro_b'], self.Wro))

        if 'leak_rate' in data:
            try:
                self.leakrate = data['leak_rate'][0][0]
            except:
                self.leakrate = data['leak_rate']
        self.neu = self.Wrr.shape[0]
        self.n_in = self.Wir.shape[1]
        self.n_out = self.Wro.shape[0]            
        # added by Eric - end
        

    def load_initial(self,filename):
        """
        Load initial conditions from a filename.
        """
        data = {}
        print("loading reservoir")
        io.loadmat(filename, data)
        self.Wrr0 = data['Wrr']
        self.Wir0 = data['Wir']
        self.Wbr0 = data['Wbr']
        self.Wro = data['Wro']
        #self.Wor0 = data['Wor']
        self.a = data['a0']

    def save_initial_fun(self,filename):
        """
        Save initial conditions to a filename.
        """
        data = {}
        data['Wrr'] = self.Wrr0
        data['Wir'] = self.Wir0
        data['Wbr'] = self.Wbr0
        data['Wor'] = self.Wor0
        data['Wro'] = self.Wro
        data['a0'] = self.a
        print( "saving reservoir")
        io.savemat(filename, data)

    
#    def trainLMS(self,ref):
#
#        learningrate = 1
#
#        e = self.trainingError(ref)
#        for saida in range(self.n_out):
#            Theta = self.Wro[saida, :]
#            Theta = Theta.reshape([self.neu, 1])
#            Theta = Theta - learningrate*e*self.a/np.dot(self.a.T,self.a)
#            self.Wro[saida, :] = Theta.T
#
    def offline_training1(self,X,Y,regularization,warmupdrop): #X is a matrix in which X[i,:] is all parameters at time i. Y is a vector of desired outputs.
        A = np.empty([Y.shape[0]-warmupdrop,self.neu])
        for i in range(Y.shape[0]):
            if self.output_feedback:
                if i > 0:
                    self.update(X[i,:],Y[i-1,:])
                else:
                    self.update(X[i,:])
                if i > warmupdrop:
                    A[i-warmupdrop,:] = self.a.T
            else:
                self.update(X[i, :])
                if i > warmupdrop:
                    A[i - warmupdrop, :] = self.a.T
                    
                    
        A_wbias = np.hstack((np.ones([A.shape[0],1]),A))

        P = np.dot(A_wbias.T,A_wbias)
        R = np.dot(A_wbias.T,Y[warmupdrop:])
            #print R,"R"
        Theta = np.linalg.solve(P+regularization*np.eye(self.neu+1,self.neu+1),R)
        self.Wro = Theta.T


    def offline_training1(self,X,Y,regularization,warmupdrop): #X is a matrix in which X[i,:] is all parameters at time i. Y is a vector of desired outputs.
        A = np.empty([Y.shape[0]-warmupdrop,self.neu])
        for i in range(Y.shape[0]):
            if self.output_feedback:
                if i > 0:
                    self.update(X[i,:],Y[i-1,:])
                else:
                    self.update(X[i,:])
                if i > warmupdrop:
                    A[i-warmupdrop,:] = self.a.T
            else:
                self.update(X[i, :])
                if i > warmupdrop:
                    A[i - warmupdrop, :] = self.a.T
                    
                    
        A_wbias = np.hstack((np.ones([A.shape[0],1]),A))

        P = np.dot(A_wbias.T,A_wbias)
        R = np.dot(A_wbias.T,Y[warmupdrop:])
            #print R,"R"
        Theta = np.linalg.solve(P+regularization*np.eye(self.neu+1,self.neu+1),R)
        self.Wro = Theta.T


    def offline_training(self,X,Y,regularization,warmupdrop): #X is a matrix in which X[i,:] is all parameters at time i. Y is a vector of desired outputs.
        """
        Simplest form of training, with an outsidely defined warmupdrop.
        Follows the same logic areg_least_squares. I do not recommend the use of this function. 
        """
        A = np.empty([Y.shape[0]-warmupdrop,self.neu],training = True)
        for i in range(Y.shape[0]):
            if self.output_feedback:
                if i > 0:
                    self.update(X[i,:],Y[i-1,:],training = True)
                else:
                    self.update(X[i,:],training = True)
                if i > warmupdrop:
                    A[i-warmupdrop,:] = self.a.T
            else:
                self.update(X[i, :],training = True)
                if i > warmupdrop:
                    A[i - warmupdrop, :] = self.a.T
                    
                    
        A_wbias = np.hstack((np.ones([A.shape[0],1]),A))

        P = np.dot(A_wbias.T,A_wbias)
        R = np.dot(A_wbias.T,Y[warmupdrop:])
            #print R,"R"
        Theta = np.linalg.solve(P+regularization*np.eye(self.neu+1,self.neu+1),R)
        self.Wro = Theta.T
        
        
    def add_data(self,input_data,output_data,warmupdrop):
            """
            Adds data to the class database. Run this function after a single simulation. The purpose of this function
            is to accumulate data from many different simulations using the same system.
            :param input_data: data with the input, (number of data x self.n_in)
            :type input_data: numpy.ndarray
            :param output_data: data with the desired output (number of data x self.n_out)
            :type output_data: numpy.ndarray
            :param warmupdrup: the number of data that shall be dropped when gathering data
            """
            #warning, always handle both cum_Data variables in pair. THis function adds data for the ESN to train.
            
            
            self.simulations_start.append(self.cum_data_input.shape[0])
            A = np.empty([input_data.shape[0]-warmupdrop,self.neu])
            for i in range(input_data.shape[0]):
                if self.output_feedback:
                    if i > 0:
                        self.update(input_data[i,:],output_data[i-1,:],training = True)
                    else:
                        self.update(input_data[i,:],training = True)
                    if i > warmupdrop:
                        A[i-warmupdrop,:] = self.a.T
                else:
                    self.update(input_data[i, :],training = True)
                    if i > warmupdrop:
                        A[i - warmupdrop, :] = self.a.T
            if self.cum_data_input.size == 0:
                self.cum_data_input = A
                self.cum_data_output = output_data[warmupdrop:]
                
            else:
                self.cum_data_input = np.vstack([self.cum_data_input,A])
                self.cum_data_output = np.vstack([self.cum_data_output,output_data[warmupdrop:]])
                
            self.number_of_simulations += 1
            #self.reset()
            

    
    
    def cum_train_cv(self,min_reg,max_reg,tests=50,folds = 10): #use after using add_data to add data.
        """
        main training function, use after at least one instance of add_data. Comes with built-in cross validation. Float unless
        stated otherwise
        :param min_reg: minimum regularization value.
        :param max_reg: maximum regularization value.
        :param reg_test: number of tests made. Default at 50.
        :type reg_test: int
        :param folds: number of folds in the cross-validation. Defaults at 10.
        :type folds: int
        :returns: the resulting training error of the cv, and the best regularization parameter.
        :rtype: tupĺe
        """
        #crossvalidation startup
        reg_list = np.linspace(min_reg,max_reg,tests) #regularization parameter candidates
        error_list = np.empty(folds) #list of errors.
        val_size = self.cum_data_input.shape[0]//folds #size of each subset.
        
        A_wbias = np.hstack((np.ones([self.cum_data_input.shape[0],1]),self.cum_data_input)) #add bias

        
        best_error = 9999999999999999999.0 ## comparator
        best_reg = 0.0
        
        
        
        for i,regularization in enumerate(reg_list): 
            
            for fold in range(folds):
                #separate training set and validation set
                if fold < folds - 1: #separate folds
                    training_A = np.vstack([A_wbias[:val_size*fold,:],A_wbias[val_size*fold+val_size:,:]])                
                    training_y = np.vstack([self.cum_data_output[:val_size*fold,:],
                                            self.cum_data_output[val_size*fold+val_size:,:]])
                else:
                    training_A = A_wbias[:val_size*fold,:]
                    training_y = self.cum_data_output[:val_size*fold,:]
                
                cv_A = A_wbias[val_size*fold:val_size*fold+val_size,:]
                cv_y = self.cum_data_output[val_size*fold:val_size*fold+val_size,:] 
                
                #run training
                theta = reg_least_squares(training_A,training_y,regularization)
                error_list[fold] =  np.mean((cv_y - np.dot(cv_A,theta))**2)
                
            mean_error = np.mean(error_list)
            if mean_error < best_error:
                best_error = mean_error
                best_reg = reg_list[fold]
            
        
        
        
    
        self.Wro = reg_least_squares(A_wbias,self.cum_data_output,best_reg).T
        
        
        #the functon returns the cv error of the weights used.
        return best_error,best_reg     

    def reset(self):
        """
        Resets the networks bringing the state into 0.
        """

        self.a = np.zeros([self.neu, 1])

    def get_forgetingfactor(self):

        return self.forget
    
    def get_state_from(self,outer_esn):
        """
        Copy state from another network
        """
        
        self.a = outer_esn.a
        
    def get_derivative_df_du(self, inp,y_in = np.atleast_2d(0)):
        """
        Obtains dfdu gradient, for gradient calculation purposes.
        """
        
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
        #Y_in = np.zeros([self.n_out,1])
        z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
        if not (y_in == 0).all():
            Y_in = np.array(y_in)
            Y_in = Y_in.reshape(Y_in.size, 1)
            z += np.dot(self.Wor,Y_in)
                     
        J = grad_tanh(z)
        
        return np.dot(J,self.Wir)
    
    def get_derivative_df_dx(self, inp,y_in = np.atleast_2d(0)):
        """
        obtains dfdx gradient, for gradient calculation purposes.
        """
        
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)

        z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
        if not (y_in == 0).all():
            Y_in = np.array(y_in)
            Y_in = Y_in.reshape(Y_in.size, 1)
            z += np.dot(self.Wor,Y_in)
        
        J = grad_tanh(z)

        z1 = self.Wrr
        if self.output_feedback:
            z1 += np.dot(self.Wor,self.Wro[:,1:])
        z1 = np.dot(J,z1)
        
        return (1-self.leakrate)*np.eye(self.neu) + self.leakrate * z1
        
    def get_current_output(self):
        """
        Returns the output of the network.
        """
        a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
        return np.dot(self.Wro,a_wbias)
    
    def covariance_reset(self,diag_alpha):
        self.P = np.eye(self.neu)/diag_alpha
    

    @staticmethod
    def new_rnn_from_weights(weights):
        esn_4tanks = EchoStateNetwork(
            neu = 400,
            n_in = 2,
            n_out = 2,
            gama = 0.1,
            ro = 0.99,
            psi = 0.0,
            in_scale = 0.1,
            bias_scale = 0.1,
            initial_filename = "4tanks1",
            load_initial = False,
            save_initial = False,
            output_feedback = False)
        esn_4tanks.load_reservoir_from_array(weights)
        esn_4tanks.reset()
        return esn_4tanks
    

