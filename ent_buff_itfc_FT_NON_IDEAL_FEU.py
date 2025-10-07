from qunetsim.objects import Qubit
from qunetsim.components import Host
from QuTils import EPR_Pair_fidelity
import numpy as np
import pandas as pd 
import time

class HostNotStartedException(Exception):
    pass

class HostsNotConnectedException(Exception):
    pass

class BadInputException(Exception):
    pass

class FidelitySettingException(Exception):
    pass

class InterfaceIsNotInProcessException(Exception):
    pass

class InterfaceIsInProcessException(Exception):
    pass

class BadPhaseCallException(Exception):
    pass

class IsNotSendersITFCException(Exception):
    pass

class IsNotReceiversITFCException(Exception):
    pass

class EPR_buff_itfc(object):
    
    def __init__(self, host: Host, partner_host_id=None, n_exp=None, 
                 eff_load=None, is_receiver=None, use_max_Fest=None):

        if host._queue_processor_thread is None:
            raise HostNotStartedException("host must be started before"
                                          " initializing its EPR buffer "
                                          "interface.")
        if (partner_host_id is None) or (type(partner_host_id) is not str):
            raise ValueError("partner_host_id must be a string.")
        
        if not (partner_host_id in host.quantum_connections):
            raise HostsNotConnectedException("host and partner_host must be"
                                             " connected.")

        self.host = host
        self.partner_host_id = partner_host_id
        
        if n_exp is None:
            self.n = 2**6
            self.n_exp = 6
        else:
            # TODO: check validity of argument frame_capacity 
            self.n = 2**n_exp
            self.n_exp = n_exp

        if eff_load is None:
            self.eff_load = 40
        else:
            self.eff_load = eff_load 
        
        self.buffer_info = {key:({"f_est":[]},{"qubit_ids":[]}) for key in  np.linspace(0, (self.n - 1), self.n, dtype=int)}
        self.is_empty = True
        self.is_full = False
        self.f_IDs = set(np.linspace(0, (self.n-1), self.n, dtype=int))
        self.u_IDs = []
        self.ID_in_process = ([],[])
        self.in_process = False
        self._start_time = None
        self.history = None
        if is_receiver is None:
            self.is_receiver=False
        else:
            self.is_receiver= is_receiver
        self.started = False
        if use_max_Fest is None:
            # use the uID with highest fidelity first
            self.use_max_Fest =  True
        else:
            # use the oldest uID first
            self.use_max_Fest = use_max_Fest

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        if self.history is None:
            self._start_history()
        if not self.started:
            self.started = True
        self._start_time = start_time

    def _start_history(self):
        if self.is_receiver:
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "F_est_ideal":pd.Series(dtype="float"),
                                                   "ti_meas": pd.Series(dtype="float"),
                                                   "ti_F_val": pd.Series(dtype="float"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "ACK": pd.Series(dtype="int"),
                                                   "t_finish": pd.Series(dtype="float"),
                                                   "uID": pd.Series(dtype="int"),
                                                   "Fid": pd.Series(dtype="float"),
                                                   "t_init_sdc": pd.Series(dtype="float"),
                                                   "t_end_sdc": pd.Series(dtype="float")})

            self.SDC_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "t_end_rtrv": pd.Series(dtype="float"),
                                                   "t_finish": pd.Series(dtype="float"),
                                                   "valid_C_info": pd.Series(dtype="int")})

        else: # Sender
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "F_est_b_xy": pd.Series(dtype="float"),
                                                   "ti_meas": pd.Series(dtype="float"),
                                                   "ti_F_val": pd.Series(dtype="float"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "ACK": pd.Series(dtype="int"),
                                                   "t_finish": pd.Series(dtype="float"),
                                                   "ipID_drop": pd.Series(dtype="int"),
                                                   "nuID_drop":pd.Series(dtype="int"),
                                                   "Fid": pd.Series(dtype="float")})

            self.SDC_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "t_end_rtrv": pd.Series(dtype="float"),
                                                   "t_finish": pd.Series(dtype="float")})

        self.In_halves_history = pd.DataFrame({"t_in": pd.Series(dtype="float"),
                                               "Fid_in": pd.Series(dtype="float")})
    

    def _actualize_histories(self, df_to_add, kind:str):
        if kind == "epr": 
            if self.is_receiver: # valid for receiver
                if self.EPR_frame_history.size == 0:
                    self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
                else:
                    if np.count_nonzero(self.EPR_frame_history.iloc[-1].isnull().values) == 4:
                        self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
                    else:
                        boolcols = self.EPR_frame_history.columns.isin(df_to_add.columns.values)
                        vals = df_to_add.values
                        vals = vals.reshape((vals.shape[1],))
                        self.EPR_frame_history.iloc[-1, boolcols] = vals

            else:                # valid for sender
                if self.EPR_frame_history.size == 0:
                    self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
                else:
                    if np.count_nonzero(self.EPR_frame_history.iloc[-1].isnull().values) == 3:
                        self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
                    else:
                        boolcols = self.EPR_frame_history.columns.isin(df_to_add.columns.values)
                        vals = df_to_add.values
                        vals = vals.reshape((vals.shape[1],))
                        self.EPR_frame_history.iloc[-1, boolcols] = vals

        elif kind == "sdc":
            if self.SDC_frame_history.size == 0:
                self.SDC_frame_history = pd.concat([self.SDC_frame_history, df_to_add])
            else:
                if self.SDC_frame_history.iloc[-1].isnull().values.any():
                    boolcols = self.SDC_frame_history.columns.isin(df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.SDC_frame_history.iloc[-1, boolcols] = vals
                else:
                    self.SDC_frame_history = pd.concat([self.SDC_frame_history, df_to_add])
        else:
            if self.In_halves_history.size == 0:
                self.In_halves_history = pd.concat([self.In_halves_history, df_to_add])
            else:
                if self.In_halves_history.iloc[-1].isnull().values.any():
                    boolcols = self.In_halves_history.columns.isin(df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.In_halves_history.iloc[-1, boolcols] = vals
                else:
                    self.In_halves_history = pd.concat([self.In_halves_history, df_to_add])

    def _start(self):
        self._start_history()
        self.started = True
        self._start_time = time.time()

    def start(self):
        self._start()

    def _get_tuple_from_fr_ID(self, id_f=None):
        if id_f is None:
            raise BadInputException("A valid id for the frame must be passed.")
        try:
            frame_info = self.buffer_info[id_f]
        except KeyError:
            print("The id {k} is not a valid frame id.".format(k=id_f))
        else:
            return frame_info
    
    def _get_F_est_from_fr_ID(self, id_f=None):
        fr_info = self._get_tuple_from_fr_ID(id_f=id_f)
        return fr_info[0]["f_est"]

    def _get_qubit_ids_from_fr_ID(self, id_f=None):
        fr_info =  self._get_tuple_from_fr_ID(id_f=id_f)
        return fr_info[1]["qubit_ids"]

    def _clear_frame_info(self, frame_id=None):
        frame_info = self._get_tuple_from_fr_ID(id_f=frame_id)
        frame_info[0]["f_est"].clear()
        frame_info[1]["qubit_ids"].clear()

    def _get_oldest_uID(self):
        return self.u_IDs.pop(0)
    
    def _get_uID_with_max_Fest(self):
        F_mx_id = None
        f_max = 0 
        for fr_id in self.u_IDs:
            f_est = self._get_F_est_from_fr_ID(id_f=fr_id)
            if f_est[0] > f_max:
                f_max = f_est[0]
                F_mx_id = fr_id
        self.u_IDs.remove(F_mx_id)
        return F_mx_id

    def _drop_stored_epr_frame_ID(self, id_f=None):
        qids = self._get_qubit_ids_from_fr_ID(id_f=id_f)
        if self.in_process and (id_f in self.ID_in_process[0]):
            self._reset_is_IN_PROCESS()
        if id_f in self.u_IDs:
            self.u_IDs.remove(id_f)
        for qid in qids:
            self.host.drop_epr(self.partner_host_id, qid)
        self._actualize_ITFC_after_popdrop_ID(id_f=id_f)

    def _append_id_to_ID_ip(self, id_f=None):
        if id_f is None:
            raise ValueError("pass a valid id_f")
        else:
            self.ID_in_process[0].append(id_f)

    def _append_mssg_to_ID_ip(self, mssg:str):
        if type(mssg) is not str:
            raise ValueError("pass a valid string mssg")
        else:
            if self.in_process and (self._get_MSSG_in_process() == "EPR:storing"):
                self.ID_in_process[1].append(mssg)
            else:
                raise InterfaceIsNotInProcessException("Interface must be "
                                                       "processing at *EPR:storing*.")

    def _set_mssg_to_ID_ip(self, mssg=None):
        if self.in_process:
            if len(self.ID_in_process[1]) == 1:
                self.ID_in_process[1].clear()
                self.ID_in_process[1].append(mssg)
            else:
                self.ID_in_process[1].clear()
                for m in mssg:
                    self.ID_in_process[1].append(m)
        else:   
            self.ID_in_process[1].append(mssg)

    def _get_ID_in_process(self):
        ID_ip = None
        if len(self.ID_in_process[0]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.ID_in_process[0]) == 1:
            ID_ip = self.ID_in_process[0][0]
        else:
            ID_ip = self.ID_in_process[0].copy()
        return ID_ip

    def _get_MSSG_in_process(self):
        MSSG_ip = None
        if len(self.ID_in_process[1]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.ID_in_process[1]) == 1:
            MSSG_ip = self.ID_in_process[1][0]
        else:
            MSSG_ip = self.ID_in_process[1].copy()
        return MSSG_ip

    def _reset_is_IN_PROCESS(self):
        if self.in_process:
            self.in_process = False
            self.ID_in_process[0].clear()
            self.ID_in_process[1].clear()
        else:
            raise ValueError("No Frame is being processed. Imposible to reset"
                             " in_process.")

    def _get_nxt_uID(self):
        nxt_ID = None
        if self.use_max_Fest:
            nxt_ID = self._get_uID_with_max_Fest()
        else:
            nxt_ID = self._get_oldest_uID()
        return nxt_ID

    def _actualize_ITFC_after_popdrop_ID(self, id_f):
        self._clear_frame_info(frame_id=id_f)
        if len(self.f_IDs)==0:
            self.is_full = False
        self.f_IDs.add(id_f)
        if len(self.f_IDs)==self.n:
            self.is_empty = True

    # ***************************************************
    # ** Methods for Phases of EPR-Frame Communication **
    # ***************************************************
    def nfID_EPR_START(self):
        if not self.in_process:
            free_list = self.f_IDs.copy()
            free_list = list(free_list)
            nfID = free_list[0]
            t_init = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[nfID, t_init]], columns=["ID", "t_init"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
            self.f_IDs.remove(nfID)
            self._append_id_to_ID_ip(id_f=nfID)
            self._set_mssg_to_ID_ip(mssg="EPR:started")
            self.in_process = True
            return nfID
        else:
            raise InterfaceIsInProcessException("EPR_START cannot be called. "
                                                "Interface is already processing"
                                                " a frame.")

    def store_EPR_PHASE_1(self, epr_half: Qubit):  
        if type(epr_half) is not Qubit:
            raise BadInputException("A valid epr_half must be passed.")
        mssg = self._get_MSSG_in_process()
        valid_mssg = ((mssg == "EPR:started") or (mssg == "EPR:storing")) 
        if self.in_process and valid_mssg: 
            qids = self._get_qubit_ids_from_fr_ID(id_f=self._get_ID_in_process())
            ti = time.perf_counter() - self.start_time 
            fest =  EPR_Pair_fidelity(epr_half)
            actualize_df = pd.DataFrame([[ti, fest]], columns=["t_in", "Fid_in"])
            self._actualize_histories(df_to_add=actualize_df, kind="in")
            q_id = self.host.add_epr(host_id=self.partner_host_id, qubit=epr_half)        
            if len(qids)==0:
                self._set_mssg_to_ID_ip("EPR:storing")
            qids.append(q_id)    
        else:
            raise BadPhaseCallException("EPR_PHASE_1 can repeatedly be called "
                                        "only after EPR_START until the frame "
                                        "was completely received.")
        
    def get_qids_EPR_PHASE_2(self):
        if self.in_process and self._get_MSSG_in_process() == "EPR:storing":
            qids = self._get_qubit_ids_from_fr_ID(id_f=self._get_ID_in_process())
            self._set_mssg_to_ID_ip("EPR:FEU-measuring")
            qubit_ids = qids.copy()
            ti_meas = time.perf_counter() - self.start_time 
            actualize_df = pd.DataFrame([[ti_meas]], columns=["ti_meas"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
            return qubit_ids
        else:
            raise BadPhaseCallException("EPR_PHASE_2 can be called only after" 
                                        "EPR_PHASE_1")

    def get_epr_EPR_PHASE_3(self, qubit_id: str):
        """get the EPR half *qubit_id*. This method is solely used for getting
        the EPR halves to be measured in the FEU(fidelity estimation unit). """

        if type(qubit_id) is not str:
            raise BadInputException("A valid qubit_id must be passed.")
        if self.in_process and self._get_MSSG_in_process() == "EPR:FEU-measuring":
            qids = self._get_qubit_ids_from_fr_ID(id_f=self._get_ID_in_process())
            try:
                idx = qids.index(qubit_id)
            except ValueError:
                print("the qubit id: {id} is corrupted or is not in frame {FrID}".format(id=qubit_id, FrID=frame_id))
            else:
                epr_half = self.host.get_epr(host_id=self.partner_host_id, q_id=qubit_id)
                del(qids[idx])
                if len(qids)==self.eff_load:
                    self._set_mssg_to_ID_ip("EPR:FEU-validating")
                    ti_F_val = time.perf_counter() - self.start_time 
                    actualize_df = pd.DataFrame([[ti_F_val]], columns=["ti_F_val"])
                    self._actualize_histories(df_to_add=actualize_df, kind="epr")
                return epr_half
        else:
            raise BadPhaseCallException("EPR_PHASE_3 can repeatedly be called "
                                        " after EPR_PHASE_2, *UNTIL* all of the"
                                        " qubit to me measured were retrieved.")

    # reaction to ACK and to F_est > F_thres
    def set_F_EPR_END_PHASE(self, F_est=None, to_history=False):
        if to_history:
            if self.is_receiver:
                actualize_df = pd.DataFrame([[F_est]], columns=["F_est_ideal"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
            else:
                actualize_df = pd.DataFrame([[F_est]], columns=["F_est_b_xy"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
        else:
            if self.in_process and self._get_MSSG_in_process() == "EPR:FEU-validating":
                ip_ID = self._get_ID_in_process()
                f = self._get_F_est_from_fr_ID(id_f=ip_ID)
                if len(f)==0:
                    f.append(F_est)
                    t_fin = time.perf_counter() - self.start_time
                    actualize_df = pd.DataFrame([[F_est, int(1),t_fin]],
                                                columns=["F_est", "ACK" ,"t_finish"])
                    self._actualize_histories(df_to_add=actualize_df, kind="epr")
                    if len(self.u_IDs)==0:
                        self.is_empty = False
                    self.u_IDs.append(ip_ID)
                    if len(self.u_IDs)==self.n:
                        self.is_full = True
                    self._reset_is_IN_PROCESS()
                else:
                    raise FidelitySettingException("Frame {iD} has already a f_est.".format(iD=ip_ID))
            else:
                raise BadPhaseCallException("set_F_est_EPR_END_PHASE can be called "
                                            "only after EPR_PHASE_3.")

    # reaction to NACK and to F_est < F_thres
    def drop_ip_frame_EPR_END_PHASE(self, fest=None):
        if self.in_process and self._get_MSSG_in_process()=="EPR:FEU-validating":
            self._drop_stored_epr_frame_ID(id_f=self._get_ID_in_process())
            if fest == None:
                fest = 0
            t_fin = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[fest, int(0), t_fin]],
                                            columns=["F_est", "ACK", "t_finish"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
        else:
            raise BadPhaseCallException("drop_ip_frame_EPR_END_PHASE can be "
                                        "called only after EPR_PHASE_3.")

    # reaction to D-NACK 
    def drop_DNACK_EPR_END_PHASE(self):
        if not self.is_receiver:
            if self.in_process and self._get_MSSG_in_process()=="EPR:storing":
                ipID = self._get_ID_in_process()
                nuID = self._get_nxt_uID()
                fid = self._get_F_est_from_fr_ID(id_f=nuID)
                fid = fid[0]           
                self._drop_stored_epr_frame_ID(id_f=ipID)
                self._drop_stored_epr_frame_ID(id_f=nuID)
                t_fin = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[int(0), t_fin, ipID, nuID, fid]],
                                columns=["ACK", "t_finish", "ipID_drop", "nuID_drop", "Fid"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")

                return ipID, nuID
            else:
                raise BadPhaseCallException("drop_DNACK_EPR_END_PHASE can be called"
                                            " only after EPR_PHASE_ONE.")
        else:
            raise IsNotSendersITFCException("Actual method can be called only "
                                            " if the interface owner is sender.")

    # correct EPR-Frame in storing process as SDC-Frame and decode c-info.
    def nuID_CORRECT_epr_as_sdc_EPR_PHASE_2(self):
        if self.is_receiver:
            if self.in_process and (self._get_MSSG_in_process() == "EPR:storing"):
                nuID = self._get_nxt_uID()
                self._append_id_to_ID_ip(nuID)
                self._append_mssg_to_ID_ip("SDC:correct_EPR")
                fid = self._get_F_est_from_fr_ID(id_f=nuID)
                t_init = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[0, 0, int(nuID), fid[0], t_init]], 
                                columns=["ti_meas", "ti_F_val","uID", "Fid", "t_init_sdc"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
                return nuID
            else:
                raise BadPhaseCallException("nxt_uID_CORRECT_epr_as_sdc_EPR_PHASE_2"
                                            " can be called only after EPR_PHASE_1.")
        else:
            raise IsNotReceiversITFCException("Actual method can be called only"
                                              " if the interface owner is "
                                              "receiver.")

    def pop_sync_SDC_END_PHASE(self):
        """Used after using *store_epr_fID_EPR_PHASE_1* and *nxt_uID_SDC_START*
        in order to correct a received SDC-Frame misinterpreted as EPR-Frame. 
        The received SDC-Frame was stored as EPR-Frame. The received and stored
        SDC-Frame *rcv_ID* is poped alongside its corresponding stored EPR-Frame 
        *sto_ID* in order to decode the SDC-encoded classical information."""
        
        if self.is_receiver:
            ip_MSSGs = self._get_MSSG_in_process() 
            mssg_is_valid = ((ip_MSSGs[1]=="SDC:correct_EPR") or (
                                                ip_MSSGs[1]=="SDC:sync_rtrv_sto"))
            if self.in_process and mssg_is_valid:
                ip_IDs = self._get_ID_in_process()

                m1 = (ip_MSSGs[0] == "EPR:storing")
                m2 = (ip_MSSGs[1] == "SDC:correct_EPR")

                if (len(ip_IDs) == len(ip_MSSGs) == 2):
                    rcv_qids = self._get_qubit_ids_from_fr_ID(id_f=ip_IDs[0])
                    sto_qids = self._get_qubit_ids_from_fr_ID(id_f=ip_IDs[1])
                    if m1 and m2: 
                        if len(rcv_qids) == len(sto_qids) == self.eff_load:
                            self._set_mssg_to_ID_ip(["SDC:sync_rtrv_rec", "SDC:sync_rtrv_sto"])
                        else:
                            raise ValueError("Sync retrieving of pairs: frames has not"
                                             " the same amount of pairs.")
                    rcv_qid = rcv_qids.pop(0)
                    sto_qid = sto_qids.pop(0)
                    rcv_half = self.host.get_epr(host_id=self.partner_host_id, q_id=rcv_qid)
                    sto_half = self.host.get_epr(host_id=self.partner_host_id, q_id=sto_qid)
                    if len(rcv_qids) == len(sto_qids) == 0:
                        t_end = time.perf_counter() - self.start_time
                        actualize_df = pd.DataFrame([[t_end]], columns=["t_end_sdc"])
                        self._actualize_histories(df_to_add=actualize_df, kind="epr")
                        self._actualize_ITFC_after_popdrop_ID(id_f=ip_IDs[0])
                        self._actualize_ITFC_after_popdrop_ID(id_f=ip_IDs[1])
                        self._reset_is_IN_PROCESS()
                    return rcv_half, sto_half
            else:
                raise BadPhaseCallException("pop_sync_SDC_END_PHASE can repeatedly "
                                            "be called only after nxt_uID_CORRECT_"
                                            "epr_as_sdc_EPR_PHASE_2 until all stored"
                                            " pairs were retrieved.")
        else: 
            raise IsNotReceiversITFCException("Actual method can be called only"
                                              " if the interface owner is "
                                              "receiver.") 

    # ***************************************************
    # ** Methods for Phases of SDC-Frame Communication **
    # ***************************************************
    
    def nuID_SDC_START(self):
        if not self.in_process:
            nuID = self._get_nxt_uID()
            fest = self._get_F_est_from_fr_ID(id_f=nuID)
            t_init = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[nuID, fest[0], t_init]], 
                                            columns=["ID", "F_est", "t_init"])
            self._actualize_histories(df_to_add=actualize_df, kind="sdc")
            self._append_id_to_ID_ip(id_f=nuID)
            self._set_mssg_to_ID_ip(mssg="SDC:started")
            self.in_process = True
            return nuID
        else:
            raise InterfaceIsInProcessException("EPR_START cannot be called. "
                                                "Interface is already processing"
                                                " a frame.")

    def pop_SDC_END_PHASE(self):
        if (self.in_process and ((self._get_MSSG_in_process()=="SDC:started")
                                    or (self._get_MSSG_in_process()=="SDC:epr-retrieving"))):
            ip_ID = self._get_ID_in_process()
            qids = self._get_qubit_ids_from_fr_ID(id_f=ip_ID)
            if len(qids)==self.eff_load:
                self._set_mssg_to_ID_ip(mssg="SDC:epr-retrieving")
            qid = qids.pop(0)
            epr_half = self.host.get_epr(host_id=self.partner_host_id, q_id=qid)
            if len(qids) == 0:
                t_end = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[t_end]], columns=["t_end_rtrv"])
                self._actualize_histories(df_to_add=actualize_df, kind="sdc")
                self._actualize_ITFC_after_popdrop_ID(id_f=ip_ID)
                self._reset_is_IN_PROCESS()
            return epr_half
        else:
            raise BadPhaseCallException("SDC_END_PHASE can be repeatedly called"
                                        " only after SDC_START until stored "
                                        "frame was consumed.")

    def finish_SDC(self, val_C_info=None):
        if self.is_receiver:
            if val_C_info is None:
                raise ValueError("integer 0(false) or 1(true) for val_C_info validating the"
                                 " sdc decoded C-Info must be passed.")
            else:
                t_fin = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[t_fin, val_C_info]], columns=["t_finish", "valid_C_info"])
                self._actualize_histories(df_to_add=actualize_df, kind="sdc")
        else:
            t_fin = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[t_fin]], columns=["t_finish"])
            self._actualize_histories(df_to_add=actualize_df, kind="sdc")
    
    # The method below is not used!
    def apply_new_Fthres_to_itfc(self, f_thres=None):
        if (f_thres is None) or (f_thres < 0.25) or (f_thres > 1):
            raise ValueError("pass a valid fidelity threshold.")
        
        if not self.in_process:
            for fr_id in self.u_IDs:
                f_est = self._get_F_est_from_fr_ID(id_f=fr_id)
                if f_est[0] < f_thres:
                    self._drop_stored_epr_frame_ID(fr_id)
        else:
            raise InterfaceIsInProcessException("New threshold fidelity can be "
                                                "applied only if interface is "
                                                "not in process.") 