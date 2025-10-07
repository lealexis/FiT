from qunetsim.components import Host
from qunetsim.objects import Qubit
import pandas as pd
import numpy as np 
import time

class EPRgenBadParameterException(Exception):
    pass

class HostNotStartedException(Exception):
    pass

class EPRgenBadTypException(Exception):
    pass

class EPR_generator(object):
    def __init__(self, host:Host, max_fid=0.99, min_fid=0.5, max_dev=0.1, 
                 min_dev=0.01, f_mu=None, f_sig=None, mu_phase=None, 
                 sig_phase=None, spread=None, typ="sinusoidal"):

        if host._queue_processor_thread is None:
            raise HostNotStartedException("host must be started before"
                                          " initializing its EPR generator.")
        self.host = host
        if max_fid < min_fid:
            raise EPRgenBadParameterException("max_fid must be bigger than"
                                              " min_fid.")
        if (max_fid > 1) or (min_fid < 0.25):
            raise EPRgenBadParameterException("max_fid must be less than or "
                                              "equal to 1.\nmin_fid must be "
                                              "greather than or equal to 0.25")
        self.max_mu = max_fid
        self.min_mu = min_fid
        
        if max_dev < min_dev:
            raise EPRgenBadParameterException("max_dev must be bigger than"
                                              " min_dev.")
        if (max_dev > 1) or (min_dev < 0):
            raise EPRgenBadParameterException("max_dev must be less than or "
                                              "equal to 1.\nmin_dev must be "
                                              "greather than or equal to 0")
        self.max_sig = max_dev
        self.min_sig = min_dev
        if spread is None:
            self.spread = 0.2
        if (typ == "sinusoidal") or (typ == "gaussian"):
            self.typ = typ
            bias_mu_sin = (max_fid + min_fid) /2
            bias_sig_sin = (max_dev + min_dev)/2
            if self.typ == "sinusoidal":     
                amp_mu_sin = (max_fid - min_fid)/2
                amp_sig_sin = (max_dev - min_dev)/2
                self.amp_mu_sin = amp_mu_sin
                self.bias_mu_sin = bias_mu_sin
                self.amp_sig_sin = amp_sig_sin
                self.bias_sig_sin = bias_sig_sin
                self.mu = None
                self.sig = None
                if f_mu is None:
                    # EPR frame of 43 qubits take 5.2 seconds
                    # we assume a default T = 20.8 s ~ f = 0.05 Hz for mean value
                    self.wf_mu = np.pi / 10
                else:
                    self.wf_mu = 2*np.pi*f_mu

                if f_sig is None:
                    # we assume a default Tau = 2T ~ f = 0.1 Hz for standard deviation
                    self.wf_sig = np.pi / 5
                else:
                    self.wf_sig = 2*np.pi*f_sig

                if sig_phase is None:
                    #we assume a default phase of pi for the sigma sinusoidal
                    self.phi_sig = np.pi 
                else:
                    self.phi_sig = sig_phase

                if mu_phase is None:
                    self.phi_mu = np.pi
                else:
                    self.phi_mu = mu_phase

            else:
                self.mu = bias_mu_sin
                self.sig = bias_sig_sin
                self.amp_mu_sin = None
                self.bias_mu_sin = None
                self.amp_sig_sin = None
                self.bias_sig_sin = None
                self.wf_mu = None
                self.wf_sig = None
                self.phi_sig = None
                self.phi_mu = None
        else:
            raise EPRgenBadTypException("typ must be sinusoidal or gaussian, "
                                        "{ntyp} is not implemented".format(ntyp=typ))
        self.started = False
        self._start_time = None
        
    def _start(self):
        if self.typ == "sinusoidal":
            self.history = pd.DataFrame({"time_gauss": pd.Series(dtype="float"),
                                         "mu": pd.Series(dtype="float"),
                                         "sig": pd.Series(dtype="float"),
                                         "time_fid": pd.Series(dtype="float"),
                                         "fid": pd.Series(dtype="float"),
                                         "time_gen": pd.Series(dtype="float"),
                                         "fid_gen": pd.Series(dtype="float")})
        else:
            self.history = pd.DataFrame({"time_fid": pd.Series(dtype="float"),
                                         "fid": pd.Series(dtype="float"),
                                         "time_gen": pd.Series(dtype="float"),
                                         "fid_gen": pd.Series(dtype="float")})
        self._start_time =  time.perf_counter()
        self.started = True

    @property
    def start_time(self):
        return self._start_time

    def start(self):
        self._start()

    def _actualize_history(self, df_to_add):
        if self.history.size == 0:
            self.history = pd.concat([self.history, df_to_add])
        else:
            if self.history.iloc[-1].isnull().values.any():
                boolcols = self.history.columns.isin(df_to_add.columns.values)
                vals = df_to_add.values
                vals = vals.reshape((vals.shape[1],))
                self.history.iloc[-1, boolcols] = vals
            else:
                self.history = pd.concat([self.history, df_to_add])

    def _mu_sig_sin(self):
        t = time.perf_counter() - self.start_time
        mu = self.amp_mu_sin * np.sin(self.wf_mu*t + self.phi_mu) + self.bias_mu_sin
        sig = self.amp_sig_sin * np.sin(self.wf_sig*t + self.phi_sig) + self.bias_sig_sin
        
        df_gauss = pd.DataFrame([[t, mu, sig]], columns=["time_gauss", "mu", "sig"])
        self._actualize_history(df_to_add=df_gauss)
        return mu, sig

    def _get_fidelity_4_gen(self):
        if not self.started:
            self._start() 
        if self.typ == "sinusoidal":
            mu, sig = self._mu_sig_sin()
        else:
            mu = self.mu
            sig = self.sig
        f = np.random.normal(loc=mu, scale=sig)
        t = time.perf_counter() - self.start_time
        df_fid = pd.DataFrame([[t, f]], columns=["time_fid", "fid"])
        self._actualize_history(df_to_add=df_fid)
        if f > 1:
            f = 0.99
        if f < 0.25:
            f = 0.25
        return f

    def _fidelity(self, epr_half: Qubit):
        """Fidelity between an imperfect EPR Pair and the bell state 
        \ket{phi^{+}}, for an imperfect EPR Pair with state vector as follows
        B1 = a_{1} + ib_{1}
        B2 = a_{2} + ib_{2} 
        B3 = a_{3} + ib_{3}
        B4 = a_{4} + ib_{4}

        The Fidelity is defined as 
        F =  0.5((a_{1} + a_{4})^2 + (b_{1} + b_{4})^2)
        """
        epr_ket = epr_half.statevector()[1]

        if np.shape(epr_ket)[0] == 4: # a valid epr_ket
            b1 = epr_ket[0]
            b4 = epr_ket[3]
            fid = 0.5 * (((b1.real + b4.real)**2) + ((b1.imag + b4.imag)**2))
            return fid
        else: # invalid epr_ket
            raise ValueError("Argument passed to epr_halve is not a two dimensional"
                             " qubit system!")
    
    def get_fidelity(self, half: Qubit):
        f = self._fidelity(epr_half=half)
        return f

    def _gen_epr(self, rx1=0, ry1=0, rx2=0, ry2=0):
        """Generate an EPR-Pair by applying x and y rotational gates with angles 
        rx1, ry1, rx2, and ry2 to the first and second qubit before passing them 
        trough the entanglement circuit.
        If default rotation angles are used a perfect EPR-Pair is generated."""

        q1 = Qubit(self.host)
        id1 = q1.id + "-A"
        id2 = q1.id + "-B"
        q1.id = id1
        q2 = Qubit(self.host, q_id=id2)

        # apply rotational gates 
        q1.rx(phi=rx1)
        q1.ry(phi=ry1)
        q2.rx(phi=rx2)
        q2.ry(phi=ry2)

        # entanglement circuit
        q1.H()
        q1.cnot(q2)
        gen_time = time.perf_counter() - self.start_time
        gen_f = self._fidelity(epr_half=q1)
        df_gen = pd.DataFrame([[gen_time, gen_f]], columns=["time_gen", "fid_gen"])
        self._actualize_history(df_to_add=df_gen)

        return q1, q2

    def get_EPR_pair(self):
        """ EPR-Pairs are generated by applying X and Y rotation gates to the zero 
        ket qubits before entering the entanglement circuit. The rotation angles 
        have the same value alpha which generate a EPR-Pair with fidelity F, which 
        could drawn from sinusoidal mean and standard deviation value for the 
        gaussian distribution(check _get_fidelity_4_gen) or from a simple gaussian 
        distribution.
        """
        F = self._get_fidelity_4_gen()
        alpha = 0.5*np.arccos(4*np.sqrt(F) - 3)
        q1, q2 = self._gen_epr(rx1=alpha, ry1=alpha, rx2=alpha, ry2=alpha)
        
        return q1, q2
        
    def get_EPR_pair_gauss_angles(self):
        F = self._get_fidelity_4_gen()
        alpha = 0.5*np.arccos(4*np.sqrt(F) - 3)
        sigma = alpha*self.spread
        rxa, rya, rxb, ryb = np.random.normal(loc=alpha, scale=sigma, size=4)
        q1, q2 = self._gen_epr(rx1=rxa, ry1=rya, rx2=rxb, ry2=ryb)

        return q1, q2

def EPR_Pair_fidelity(epr_halve: Qubit):
    """Fidelity between an imperfect EPR Pair and the bell state 
    \ket{phi^{+}}. For imperfect EPR Pair with state vector as follows
    B1 = a_{1} + ib_{1}
    B2 = a_{2} + ib_{2} 
    B3 = a_{3} + ib_{3}
    B4 = a_{4} + ib_{4}

    The Fidelity is defined as 
    F =  0.5((a_{1} + a_{4})^2 + (b_{1} + b_{4})^2)
    """
    epr_ket = epr_halve.statevector()[1]

    if np.shape(epr_ket)[0] == 4: # a valid epr_ket
        b1 = epr_ket[0]
        b4 = epr_ket[3]
        fid = 0.5 * (((b1.real + b4.real)**2) + ((b1.imag + b4.imag)**2))
        return fid
    else: # invalid epr_ket
        raise ValueError("Argument passed to epr_halve is not a two dimensional"
                         " qubit system!")
