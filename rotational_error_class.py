
from qunetsim.objects import Qubit
import pandas as pd
import numpy as np 
import time

class RotErrorBadParameterException(Exception):
    pass

class RotErrorBadTypException(Exception):
    pass

class Rot_Error(object):
    def __init__(self, max_rot=0.4, min_rot=0.05, max_dev=0.1, 
                 min_dev=0.01, f_mu=None, f_sig=None, mu_phase=None, 
                 sig_phase=None, spread=None, typ="sinusoidal", error_typ="xy"):

        if max_rot < min_rot:
            raise RotErrorBadParameterException("max_rot must be bigger than"
                                              " min_rot.")
        self.max_mu = max_rot
        self.min_mu = min_rot
        
        if max_dev < min_dev:
            raise RotErrorBadParameterException("max_dev must be bigger than"
                                              " min_dev.")
        self.max_sig = max_dev
        self.min_sig = min_dev
        if spread is None:
            self.spread = 0.2
        if (typ == "sinusoidal") or (typ == "gaussian"):
            self.typ = typ
            bias_mu_sin = (max_rot + min_rot) /2
            bias_sig_sin = (max_dev + min_dev)/2
            if self.typ == "sinusoidal":     
                amp_mu_sin = (max_rot - min_rot)/2
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
                    #we assume a default phase of pi for the mu sinusoidal
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
            raise RotErrorBadTypException("typ must be sinusoidal or gaussian, "
                                        "{ntyp} is not implemented".format(ntyp=typ))
        self.started = False
        self._start_time = None
        self.error_typ = error_typ

    def _start(self):
        if self.typ == "sinusoidal":
            self.history = pd.DataFrame({"time_gauss": pd.Series(dtype="float"),
                                         "mu": pd.Series(dtype="float"),
                                         "sig": pd.Series(dtype="float"),
                                         "time_gamm": pd.Series(dtype="float"),
                                         "gamm": pd.Series(dtype="float")})
        else:
            self.history = pd.DataFrame({"time_gamm": pd.Series(dtype="float"),
                                         "gamm": pd.Series(dtype="float")})
        self._start_time =  time.perf_counter()
        self.started = True
        

    def start(self):
        self._start()

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, ti):
        self._start_time = ti

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
        mu = self.amp_mu_sin * np.sin(self.wf_mu*t+ self.phi_mu) + self.bias_mu_sin
        sig = self.amp_sig_sin * np.sin(self.wf_sig*t + self.phi_sig) + self.bias_sig_sin
        
        df_gauss = pd.DataFrame([[t, mu, sig]], columns=["time_gauss", "mu", "sig"])
        self._actualize_history(df_to_add=df_gauss)
        return mu, sig

    def _get_gamma(self):
        if not self.started:
            self._start() 
        if self.typ == "sinusoidal":
            mu, sig = self._mu_sig_sin()
        else:
            mu = self.mu
            sig = self.sig
        gamm = np.random.normal(loc=mu, scale=sig)
        t = time.perf_counter() - self.start_time
        if gamm < 0:
            gamm = 0
        df_gamm = pd.DataFrame([[t, gamm]], columns=["time_gamm", "gamm"])
        self._actualize_history(df_to_add=df_gamm)
        
        return gamm

    def apply_error(self, flying_qubit:Qubit):
        gamma = self._get_gamma()

        if self.error_typ == "xy":
            flying_qubit.rx(phi=gamma)
            flying_qubit.ry(phi=gamma)

        elif self.error_typ == "xz":
            flying_qubit.rx(phi=gamma)
            flying_qubit.rz(phi=gamma)

        elif self.error_typ == "yz":
            flying_qubit.ry(phi=gamma)
            flying_qubit.rz(phi=gamma)

        elif self.error_typ == "xyz":
            flying_qubit.rx(phi=gamma)
            flying_qubit.ry(phi=gamma)
            flying_qubit.rz(phi=gamma)
        else:
            raise RotErrorBadTypException("Pass a valid type of error.")