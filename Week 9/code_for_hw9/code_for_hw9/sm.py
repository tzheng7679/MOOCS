from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        ss = [self.start_state]
        outs = []
        for i in range(len(input_seq)):
            trans = self.transition_fn(ss[i], input_seq[i])
            ss.append(trans)
            outs.append(self.output_fn(trans))
        
        return outs


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = [0, 0] # Change

    def transition_fn(self, s, x):
        # Your code here
        carry = (s[0] + x[0] + x[1]) // 2
        out = (s[0] + x[0] + x[1]) % 2

        return [carry, out]

    def output_fn(self, s):
        # Your code here
        return s[1]


class Reverser(SM):
    start_state = [0, []]

    def transition_fn(self, s, x):
        # Your code here
        if x == 'end': s[0] = 1        
        
        elif s[0] == 0: s[1].append(x)
        
        return s

    def output_fn(self, s):
        # Your code here
        if s[0] == 0 or len(s[1]) == 0: return None 
        else: return s[1].pop()


class RNN(SM):
    start_state = None

    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx = Wsx

        self.Wss = Wss
        self.Wss_0 = Wss_0

        self.Wo = Wo
        self.Wo_0 = Wo_0
        
        self.f1 = f1
        self.f2 = f2

        self.start_state = np.zeros(shape = (Wss.shape[1], 1))

    def transition_fn(self, s, i):
        # Your code here
        return self.f1(\
            self.Wsx@i + self.Wss@s + self.Wss_0
        )

    def output_fn(self, s):
        # Your code here
        return self.f2(\
            self.Wo@s + self.Wo_0
        )

# 1.5
# Enter the parameter matrices and vectors for an instance of the RNN class such that the output is 1 if the cumulative sum of the 
# inputs is positive, -1 if the cumulative sum is negative and 0 if otherwise. Make sure that you scale the outputs so that the 
# output activation values are very close to 1, 0 and -1. Note that both the inputs and outputs are 1x1.

# Hint: np.tanh may be useful. Remember to convert your python lists to np.array.
"""
Wsx = np.array([[1]])   # Your code here
Wss = np.array([[1]])   # Your code here
Wo = np.array([[1]])      # Your code here
Wss_0 = np.array([[0]]) # Your code here
Wo_0 = np.array([[0]])  # Your code here
f1 = lambda x : np.sum(x)     # Your code here, e.g. lambda x : x
f2 = lambda x : np.tanh(x * 100)     # Your code here
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)
"""

# 1.6
# Enter the parameter matrices and vectors for an instance of the RNN class such that it implements the following autoregressive model:
# yt=1yt−1−2yt−2+3yt−3yt​=1yt−1​−2yt−2​+3yt−3​
# when xt=yt−1xt​=yt−1​.
# Note: unlike in the lab, as a result of your call to RNN, your initial (start) state vector in the RNN will be initialized to all zeros (i.e., our RNN implementation enforces this requirement). Instead, you should assume that the initial value for xx will be provided , e.g., x1=5x1​=5 in the lab example, and that each successive xt=yt−1xt​=yt−1​ is also provided to the transduce method.
Wsx = np.array([[1, 0, 0]]).T # Your code here
Wss = np.array(
    [[0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]]
)   # Your code here
Wo = np.array([[1,-2,3]])    # Your code here
Wss_0 = np.zeros_like(Wss) # Your code here
Wo_0 = np.array([[0]])  # Your code here
f1 = lambda x : x   # Your code here
f2 = lambda x : np.array([[np.sum(x)]])    # Your code here
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)