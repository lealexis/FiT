import pandas as pd
import numpy as np
from math import trunc 
import time

class frame_length_estimator(object):
    def __init__(self, max_q=0.8, min_q=0.2, eff_load=None, freq_q=None, 
                 phase=None):

        self.max_q = max_q
        self.min_q = min_q
        self._amp_q_sin = (max_q - min_q) / 2
        self._bias_q_sin = (max_q + min_q) / 2

        if freq_q is None:
            self._wf_q = np.pi / 5
        else:
            self._wf_q = 2*np.pi*freq_q
        
        if eff_load is None:
            self._eff_load = 35
        else:
            self._eff_load = eff_load

        if phase is None:
            #we assume a default phase of pi/2 for the sigma sinusoidal
            self._phi = np.pi / 2
        else:
            self._phi = phase

        self._started = False
        self._start_time = None
        
    def _start(self):
        self.history = pd.DataFrame({"time": pd.Series(dtype="float"),
                                     "q": pd.Series(dtype="float"),
                                     "q_pract": pd.Series(dtype="float"),
                                     "frame_len": pd.Series(dtype="int")})
        self._start_time =  time.perf_counter()
        self._started = True

    @property
    def start_time(self):
        return self._start_time
    
    @start_time.setter
    def start_time(self, start_time):
        if self._started:
            self._start_time = start_time
        else:
            self._start()
            self._start_time = start_time

    @property
    def eff_load(self):
        return self._eff_load

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

    def get_frame_len(self):
        t = time.perf_counter() - self._start_time
        q = self._amp_q_sin * np.sin(self._wf_q * t + self._phi) + self._bias_q_sin
        frm_len = self._eff_load / (1 - q)
        frm_len = trunc(frm_len)
        while (frm_len - self._eff_load) % 3 != 0:
            frm_len += 1
        q_pract = 1 - (self._eff_load / frm_len)
        df_frm_len = pd.DataFrame([[t, q, q_pract, frm_len]], 
                            columns=["time", "q", "q_pract", "frame_len"])
        self._actualize_history(df_to_add=df_frm_len)
        return frm_len