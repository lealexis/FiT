
from queue import Queue
from qunetsim.objects import Qubit
from threading import Event, Timer 

class OutSocketFull(Exception):
    pass

class QuPipe(object):
    """Quantum pipeline object simulating the pipeline nature of a channel on 
    which qubits fly from sender to receiver with a predefined *delay* time. The 
    passed delay time must be bigger than the inter-qubit time. 
    """
    def __init__(self, delay=None):#, verbose=False):
        self._delay = delay
        self._queue = Queue()
        self.Qframe_in_transmission = Event()
        self.pipe_load = 0
        self.out_socket = []
        #self.verbose = verbose

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        self._delay = delay

    def put(self, flying_qubit:Qubit):

        if self.pipe_load == 0:
            self.Qframe_in_transmission.set()
        self.pipe_load += 1
        #print(self.pipe_load)
        t_get = Timer(self._delay, self.get)
        self._queue.put(flying_qubit)
        t_get.start()
            
    def get(self):
        if len(self.out_socket) == 0 :
            self.pipe_load -= 1 
            if self.pipe_load == 0:
                self.Qframe_in_transmission.clear()
            self.out_socket.append(self._queue.get())
            #if self.verbose:
            #print(self.pipe_load)
            #print("{} Qubits are on the Channel".format(self.pipe_load)) 
        else:
            raise OutSocketFull("Qubit has not yet been taken from output socket.")


class ClassicPipe(object):
    """ Pipelined non-simultaneous duplex channel for transmission of the 
    classical feedbacks between the comunicating parties. Delay should be 
    bigger than the delay of the Quantum pipelined channel.
    """
    def __init__(self, delay):
        self._delay = delay
        self.fst_feedback_in_trans = Event()
        self.snd_feedback_in_trans = Event()
        self._queue = Queue()
        self.pipe_load = 0
        self.feedback_num = 0
        self.out_socket = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        self._delay = delay
    
    def put(self, cbit):

        if (self.pipe_load == 0) and (self.feedback_num == 0):
            self.fst_feedback_in_trans.set()
            self.feedback_num += 1
        if not self.fst_feedback_in_trans.is_set() and (self.pipe_load == 0)\
                                                 and (self.feedback_num == 1):
            self.snd_feedback_in_trans.set()
            self.feedback_num += 1
        self.pipe_load += 1
        t_get = Timer(self.delay, self.get)
        self._queue.put(cbit)
        t_get.start()


    def get(self):
        if len(self.out_socket) == 0:
            self.out_socket.append(self._queue.get()) 
            self.pipe_load -= 1
            if (self.pipe_load == 0) and (self.feedback_num == 1):
                self.fst_feedback_in_trans.clear()
            if (self.pipe_load == 0) and (self.feedback_num == 2):
                self.snd_feedback_in_trans.clear()
                self.feedback_num = 0
        else:
            raise OutSocketFull("C-bit has not yet been taked from output socket.")