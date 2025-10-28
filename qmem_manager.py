import time

import numpy as np
import pandas as pd
from quTils import get_epr_fidelity
from qunetsim.objects import Qubit
from qunetsim.components import Host


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


class EntanglementManager(object):
    """A class that manages the quantum entanglement stored in a quantum
    memory accessed through host.
    The payload of a quantum frame is stored under a frame identifier
    and each qubit of the payload has a unique qubit id. This
    information is used to retrieve and store qubits from the quantum
    memory; thus, enabling the quantum communication between Alice and
    Bob, each with its own entanglement manager."""
    def __init__(self, host: Host, partner_host_id=None, n_exp=None,
                 eff_load=None, is_receiver=None, use_max_fest=None):

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
        
        if n_exp is None:  # qmem capacity
            self.n = 2**6
            self.n_exp = 6
        else:
            self.n = 2**n_exp
            self.n_exp = n_exp

        if eff_load is None:  # effective payload of a quantum frame
            self.eff_load = 40
        else:
            self.eff_load = eff_load
        
        self.buffer_info = {key:({"f_est":[]}, {"qubit_ids":[]}) for key in
                            np.linspace(0, (self.n - 1), self.n, dtype=int)}
        self.is_empty = True 
        self.is_full = False
        # frame(f) identifiers(ids)
        self.f_ids = set(np.linspace(0, (self.n - 1), self.n, dtype=int))
        # used(u) frame identifiers(ids)
        self.u_ids = []
        # frame identifier(id) in process(either being stored or retrieved)
        self.id_in_process = ([], [])
        self.in_process = False
        self._start_time = None
        self.history = None
        if is_receiver is None:
            self.is_receiver=False
        else:
            self.is_receiver= is_receiver
        self.started = False
        if use_max_fest is None:
            # use the uID with highest fidelity first
            self.use_max_fest =  True
        else:
            self.use_max_fest = use_max_fest


    @property
    def start_time(self):
        """get starting time"""
        return self._start_time


    @start_time.setter
    def start_time(self, start_time):
        """set an arbitrary starting time"""
        if self.history is None:
            self._start_history()
        if not self.started:
            self.started = True
        self._start_time = start_time


    def _start_history(self):
        """set up the corresponding dataframe where the history(all
        relevant data) of the entanglement manager will be stored.
        Two histories are crucial and are stored in separate dataframes,
        EPR and SDC frame history."""
        if self.is_receiver:
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(
                dtype="int"), "t_init": pd.Series(dtype="float"),
                "F_est_ideal":pd.Series(dtype="float"),
                # time(ti) measurement(meas)
                "ti_meas": pd.Series(dtype="float"),
                # time fidelity(F) value(val)
                "ti_F_val": pd.Series(dtype="float"),
                # fidelity estimated(est)
                "F_est": pd.Series(dtype="float"),
                # acknowledgement(ACK)
                "ACK": pd.Series(dtype="int"),
                "t_finish": pd.Series(dtype="float"),
                # used(u) identifier(ID)
                "uID": pd.Series(dtype="int"),
                # Fidelity(Fid)
                "Fid": pd.Series(dtype="float"),
                # used in correction phases, when sdc frame stored as epr frame
                # time(t) initiation(init) and end sdc, used 
                "t_init_sdc": pd.Series(dtype="float"),
                "t_end_sdc": pd.Series(dtype="float")})

            self.SDC_frame_history = pd.DataFrame({"ID": pd.Series(
                dtype="int"), "F_est": pd.Series(dtype="float"),
                "t_init": pd.Series(dtype="float"),
                # time(t) end retrieving(rtrv)
                "t_end_rtrv": pd.Series(dtype="float"),
                "t_finish": pd.Series(dtype="float"),
                # wether sdc decoded info is valid
                "valid_C_info": pd.Series(dtype="int")})

        else: # Sender
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(
                dtype="int"), "t_init": pd.Series(dtype="float"),
                # Fidelity(F) estimated before(b) xy error
                "F_est_b_xy": pd.Series(dtype="float"),
                "ti_meas": pd.Series(dtype="float"),
                "ti_F_val": pd.Series(dtype="float"),
                "F_est": pd.Series(dtype="float"),
                "ACK": pd.Series(dtype="int"),
                "t_finish": pd.Series(dtype="float"),
                # used in correction phases, when sdc frame stored as epr frame
                # in(i) process(p) identifier(ID)
                "ipID_drop": pd.Series(dtype="int"),
                # next(n) used(u) identifier(ID)
                "nuID_drop":pd.Series(dtype="int"),
                "Fid": pd.Series(dtype="float")})

            self.SDC_frame_history = pd.DataFrame({"ID": pd.Series(
                dtype="int"), "F_est": pd.Series(dtype="float"),
                "t_init": pd.Series(dtype="float"),
                "t_end_rtrv": pd.Series(dtype="float"),
                "t_finish": pd.Series(dtype="float")})
        # a history of all epr-pairs halves entering the qmem
        self.In_halves_history = pd.DataFrame({"t_in": pd.Series(
            dtype="float"), "Fid_in": pd.Series(dtype="float")})


    def _actualize_histories(self, df_to_add, kind:str):
        """adds relevant information to history.
        a dataframe to be added is passed in as well as the kind of
        history. This might be either epr, sdc or input of epr pairs."""
        if kind == "epr":
            if self.is_receiver:  # valid for receiver
                if self.EPR_frame_history.size == 0:
                    self.EPR_frame_history = pd.concat([self.EPR_frame_history,
                                                        df_to_add])
                else:
                    if np.count_nonzero(
                        self.EPR_frame_history.iloc[-1].isnull().values) == 4:
                        self.EPR_frame_history = pd.concat([
                            self.EPR_frame_history, df_to_add])
                    else:
                        boolcols = self.EPR_frame_history.columns.isin(
                            df_to_add.columns.values)
                        vals = df_to_add.values
                        vals = vals.reshape((vals.shape[1],))
                        self.EPR_frame_history.iloc[-1, boolcols] = vals

            else:  # valid for sender
                if self.EPR_frame_history.size == 0:
                    self.EPR_frame_history = pd.concat([self.EPR_frame_history,
                                                        df_to_add])
                else:
                    if np.count_nonzero(
                        self.EPR_frame_history.iloc[-1].isnull().values) == 3:
                        self.EPR_frame_history = pd.concat([
                            self.EPR_frame_history, df_to_add])
                    else:
                        boolcols = self.EPR_frame_history.columns.isin(
                            df_to_add.columns.values)
                        vals = df_to_add.values
                        vals = vals.reshape((vals.shape[1],))
                        self.EPR_frame_history.iloc[-1, boolcols] = vals

        elif kind == "sdc":
            if self.SDC_frame_history.size == 0:
                self.SDC_frame_history = pd.concat([self.SDC_frame_history,
                                                    df_to_add])
            else:
                if self.SDC_frame_history.iloc[-1].isnull().values.any():
                    boolcols = self.SDC_frame_history.columns.isin(
                        df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.SDC_frame_history.iloc[-1, boolcols] = vals
                else:
                    self.SDC_frame_history = pd.concat([self.SDC_frame_history,
                                                        df_to_add])
        else:  # epr pair halves history
            if self.In_halves_history.size == 0:
                self.In_halves_history = pd.concat([self.In_halves_history,
                                                    df_to_add])
            else:
                if self.In_halves_history.iloc[-1].isnull().values.any():
                    boolcols = self.In_halves_history.columns.isin(
                        df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.In_halves_history.iloc[-1, boolcols] = vals
                else:
                    self.In_halves_history = pd.concat([self.In_halves_history,
                                                        df_to_add])


    def _start(self):
        self._start_history()
        self.started = True
        self._start_time = time.time()


    def start(self):
        self._start()


    def _get_tuple_from_fr_id(self, id_f=None):
        if id_f is None:
            raise BadInputException("A valid id for the frame must be passed.")
        try:
            frame_info = self.buffer_info[id_f]
        except KeyError:
            print("The id {k} is not a valid frame id.".format(k=id_f))
        else:
            return frame_info
    

    def _get_f_from_fr_id(self, id_f=None):
        """gives the stored estimated fidelity (f) for a frame id(id_f)"""
        fr_info = self._get_tuple_from_fr_id(id_f=id_f)
        return fr_info[0]["f_est"]


    def _get_qids_from_fr_id(self, id_f=None):
        """get qubit identifiers stored under a frame identifier"""
        fr_info =  self._get_tuple_from_fr_id(id_f=id_f)
        return fr_info[1]["qubit_ids"]


    def _clear_frame_info(self, frame_id=None):
        frame_info = self._get_tuple_from_fr_id(id_f=frame_id)
        frame_info[0]["f_est"].clear()
        frame_info[1]["qubit_ids"].clear()


    def _get_oldest_uid(self):
        """get oldest identifier of stored epr frame"""
        return self.u_ids.pop(0)


    def _get_uid_max_fest(self):
        """get id of stored epr frame with maximal fidelity"""
        f_mx_id = None
        f_max = 0
        for fr_id in self.u_ids:
            f_est = self._get_f_from_fr_id(id_f=fr_id)
            if f_est[0] > f_max:
                f_max = f_est[0]
                f_mx_id = fr_id
        self.u_ids.remove(f_mx_id)
        return f_mx_id


    def _drop_stored_epr_frame_id(self, id_f=None):
        qids = self._get_qids_from_fr_id(id_f=id_f)
        if self.in_process and (id_f in self.id_in_process[0]):
            self._reset_ip()
        if id_f in self.u_ids:
            self.u_ids.remove(id_f)
        for qid in qids:
            self.host.drop_epr(self.partner_host_id, qid)
        self._actualize_after_popdrop_id(id_f=id_f)


    def _append_id_to_ip_id(self, id_f=None):
        """adds a frame identifier to the in process identifiers"""
        if id_f is None:
            raise ValueError("pass a valid id_f")
        else:
            self.id_in_process[0].append(id_f)


    def _append_mssg_to_ip_id(self, mssg:str):
        """append a message to ip id. Appended messages are used to
        check validity on the communication phases transitions."""
        if type(mssg) is not str:
            raise ValueError("pass a valid string mssg")
        else:
            if self.in_process and (self._get_mssg_ip() == 
                                    "EPR:storing"):
                self.id_in_process[1].append(mssg)
            else:
                raise InterfaceIsNotInProcessException("Interface must be "
                                                       "processing at "
                                                       "*EPR:storing*.")


    def _set_mssg_to_ip_id(self, mssg=None):
        """set a new message"""
        if self.in_process:
            if len(self.id_in_process[1]) == 1:
                self.id_in_process[1].clear()
                self.id_in_process[1].append(mssg)
            else:
                self.id_in_process[1].clear()
                for m in mssg:
                    self.id_in_process[1].append(m)
        else:
            self.id_in_process[1].append(mssg)


    def _get_id_ip(self):
        """returns the identifier in process"""
        ID_ip = None
        if len(self.id_in_process[0]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.id_in_process[0]) == 1:
            ID_ip = self.id_in_process[0][0]
        else:
            ID_ip = self.id_in_process[0].copy()
        return ID_ip


    def _get_mssg_ip(self):
        """returns the message stored in process"""
        MSSG_ip = None
        if len(self.id_in_process[1]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.id_in_process[1]) == 1:
            MSSG_ip = self.id_in_process[1][0]
        else:
            MSSG_ip = self.id_in_process[1].copy()
        return MSSG_ip


    def _reset_ip(self):
        """reset stored info in in_process(ip)"""
        if self.in_process:
            self.in_process = False
            self.id_in_process[0].clear()
            self.id_in_process[1].clear()
        else:
            raise ValueError("No Frame is being processed. Imposible to reset"
                             " in_process.")


    def _get_nxt_uid(self):
        """returns the next(nxt) frame identifier(id) that is to be used(u)"""
        nxt_ID = None
        if self.use_max_fest:
            nxt_ID = self._get_uid_max_fest()
        else:
            nxt_ID = self._get_oldest_uid()
        return nxt_ID


    def _actualize_after_popdrop_id(self, id_f):
        self._clear_frame_info(frame_id=id_f)
        if len(self.f_ids) == 0:
            self.is_full = False
        self.f_ids.add(id_f)
        if len(self.f_ids) == self.n:
            self.is_empty = True

    # ***************************************************
    # ** Methods for Phases of EPR-Frame Communication **
    # ***************************************************
    def nfid_epr_start(self):
        """starts the epr frame communication. Returns the next(n) 
        free(f) frame (f) identifier(id) to begin the communication
        phases."""
        if not self.in_process:
            free_list = self.f_ids.copy()
            free_list = list(free_list)
            nfID = free_list[0]
            t_init = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[nfID, t_init]], columns=["ID",
                                                                   "t_init"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
            self.f_ids.remove(nfID)
            self._append_id_to_ip_id(id_f=nfID)
            self._set_mssg_to_ip_id(mssg="EPR:started")
            self.in_process = True
            return nfID
        else:
            raise InterfaceIsInProcessException("EPR_START cannot be called. "
                                                "Interface is already "
                                                "processing a frame.")


    def store_epr_phase_1(self, epr_half: Qubit):
        """stores a qubit corresponding to an EPR-pair."""
        if type(epr_half) is not Qubit:
            raise BadInputException("A valid epr_half must be passed.")
        mssg = self._get_mssg_ip()
        valid_mssg = ((mssg == "EPR:started") or (mssg == "EPR:storing"))
        if self.in_process and valid_mssg:
            qids = self._get_qids_from_fr_id(id_f=
                                                  self._get_id_ip())
            ti = time.perf_counter() - self.start_time
            fest =  get_epr_fidelity(epr_half)
            actualize_df = pd.DataFrame([[ti, fest]], columns=["t_in",
                                                               "Fid_in"])
            self._actualize_histories(df_to_add=actualize_df, kind="in")
            q_id = self.host.add_epr(host_id=self.partner_host_id,
                                     qubit=epr_half)
            if len(qids) == 0:
                self._set_mssg_to_ip_id("EPR:storing")
            qids.append(q_id)
        else:
            raise BadPhaseCallException("EPR_PHASE_1 can repeatedly be called "
                                        "only after EPR_START until the frame "
                                        "was completely received.")


    def get_qids_epr_phase_2(self):
        """returns the qubit identidiers stored under the in process
        frame id."""
        if self.in_process and self._get_mssg_ip() == "EPR:storing":
            qids = self._get_qids_from_fr_id(id_f=
                                                  self._get_id_ip())
            self._set_mssg_to_ip_id("EPR:FEU-measuring")
            qubit_ids = qids.copy()
            ti_meas = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[ti_meas]], columns=["ti_meas"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
            return qubit_ids
        else:
            raise BadPhaseCallException("EPR_PHASE_2 can be called only after"
                                        "EPR_PHASE_1")


    def get_epr_phase_3(self, qubit_id: str):
        """get the EPR half *qubit_id*. This method is solely used for
        getting the EPR halves to be measured in the FEU(fidelity
        estimation unit). """

        if type(qubit_id) is not str:
            raise BadInputException("A valid qubit_id must be passed.")
        if (self.in_process and self._get_mssg_ip() == "EPR:FEU-measuring"):
            qids = self._get_qids_from_fr_id(id_f=self._get_id_ip())
            try:
                idx = qids.index(qubit_id)
            except ValueError:
                print("the qubit id: {id} is corrupted or is not in frame "
                      "{FrID}".format(id=qubit_id, FrID=self._get_id_ip()))
            else:
                epr_half = self.host.get_epr(host_id=self.partner_host_id,
                                             q_id=qubit_id)
                del(qids[idx])
                if len(qids) == self.eff_load:
                    self._set_mssg_to_ip_id("EPR:FEU-validating")
                    ti_F_val = time.perf_counter() - self.start_time
                    actualize_df = pd.DataFrame([[ti_F_val]],
                                                columns=["ti_F_val"])
                    self._actualize_histories(df_to_add=actualize_df,
                                              kind="epr")
                return epr_half
        else:
            raise BadPhaseCallException("EPR_PHASE_3 can repeatedly be called "
                                        " after EPR_PHASE_2, *UNTIL* all of "
                                        "the qubit to me measured were "
                                        "retrieved.")


    # reaction to ACK and to F_est > F_thres
    def set_f_epr_end_phase(self, F_est=None, to_history=False):
        """set the estimated fidelity to the frame id in process.
        If to_history, the fidelity is being stored inbetween of the
        epr-frame communication process and is not the end phase of the
        protocol."""
        if to_history:
            if self.is_receiver:
                actualize_df = pd.DataFrame([[F_est]], columns=["F_est_ideal"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
            else:
                actualize_df = pd.DataFrame([[F_est]], columns=["F_est_b_xy"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
        else:
            if (self.in_process and self._get_mssg_ip() == 
                "EPR:FEU-validating"):
                ip_id = self._get_id_ip()
                f = self._get_f_from_fr_id(id_f=ip_id)
                if len(f) == 0:
                    f.append(F_est)
                    t_fin = time.perf_counter() - self.start_time
                    actualize_df = pd.DataFrame([[F_est, int(1),t_fin]],
                                                columns=["F_est", "ACK",
                                                         "t_finish"])
                    self._actualize_histories(df_to_add=actualize_df,
                                              kind="epr")
                    if len(self.u_ids) == 0:
                        self.is_empty = False
                    self.u_ids.append(ip_id)
                    if len(self.u_ids) == self.n:
                        self.is_full = True
                    self._reset_ip()
                else:
                    raise FidelitySettingException("Frame {iD} has already a" 
                                                   " f_est.".format(iD=ip_id))
            else:
                raise BadPhaseCallException("set_f_epr_end_phase can be called"
                                            " only after EPR_PHASE_3.")


    # reaction to NACK and to F_est < F_thres
    def drop_ip_id_epr_end_phase(self, fest=None):
        """drops the frame in process."""
        if self.in_process and self._get_mssg_ip() == "EPR:FEU-validating":
            self._drop_stored_epr_frame_id(id_f=self._get_id_ip())
            if fest is None:
                fest = 0
            t_fin = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[fest, int(0), t_fin]],
                                            columns=["F_est", "ACK",
                                                     "t_finish"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
        else:
            raise BadPhaseCallException("drop_ip_id_epr_end_phase can be "
                                        "called only after epr_phase_3.")


    # reaction to D-NACK
    def drop_dnack_epr_end_phase(self):
        if not self.is_receiver:
            if self.in_process and self._get_mssg_ip() == "EPR:storing":
                ipid = self._get_id_ip()
                nuid = self._get_nxt_uid()
                fid = self._get_f_from_fr_id(id_f=nuid)
                fid = fid[0]
                self._drop_stored_epr_frame_id(id_f=ipid)
                self._drop_stored_epr_frame_id(id_f=nuid)
                t_fin = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[int(0), t_fin, ipid, nuid, fid]],
                                columns=["ACK", "t_finish", "ipID_drop",
                                         "nuID_drop", "Fid"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")

                return ipid, nuid
            else:
                raise BadPhaseCallException("drop_dnack_epr_end_phase can be "
                                            "called only after epr_phase_one.")
        else:
            raise IsNotSendersITFCException("Actual method can be called only "
                                            "if the interface owner is sender")


    # correct EPR-Frame in storing process as SDC-Frame.
    def correct_epr_as_sdc_epr_phase_2(self):
        """correct an sdc frame that was stored at a receivers
        entanglement manager as an epr frame. It also returns the the
        next used frame identifier (nuid) containing the corresponding
        epr-pair halves."""
        if self.is_receiver:
            if self.in_process and (self._get_mssg_ip() == "EPR:storing"):
                nuid = self._get_nxt_uid()
                self._append_id_to_ip_id(nuid)
                self._append_mssg_to_ip_id("SDC:correct_EPR")
                fid = self._get_f_from_fr_id(id_f=nuid)
                t_init = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[0, 0, int(nuid), fid[0],
                                              t_init]], columns=["ti_meas",
                                                                 "ti_F_val",
                                                                 "uID", "Fid",
                                                                 "t_init_sdc"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
                return nuid
            else:
                raise BadPhaseCallException("correct_epr_as_sdc_epr_phase_2"
                                            " can be called only after"
                                            " epr_phase_1.")
        else:
            raise IsNotReceiversITFCException("Actual method can be called"
                                              " only if the interface owner is"
                                              " receiver.")


    def pop_sync_sdc_end_phase(self):
        """Used after using *store_epr_phase_1* and *nuid_sdc_start* in
        order to correct a received SDC-Frame misinterpreted as
        EPR-Frame. The received SDC-Frame was stored as EPR-Frame. The
        stored SDC-Frame is poped qubit by qubit alongside its
        corresponding stored EPR-Frame in order to decode the
        SDC-encoded classical information. Returns an epr-pair encoded 
        with information and stored at receiver."""
        
        if self.is_receiver:
            ip_mssgs = self._get_mssg_ip()
            mssg_is_valid = ((ip_mssgs[1] == "SDC:correct_EPR") 
                             or (ip_mssgs[1] == "SDC:sync_rtrv_sto"))
            if self.in_process and mssg_is_valid:
                ip_ids = self._get_id_ip()

                m1 = (ip_mssgs[0] == "EPR:storing")
                m2 = (ip_mssgs[1] == "SDC:correct_EPR")

                if (len(ip_ids) == len(ip_mssgs) == 2):
                    rcv_qids = self._get_qids_from_fr_id(id_f=ip_ids[0])
                    sto_qids = self._get_qids_from_fr_id(id_f=ip_ids[1])
                    if m1 and m2:
                        if len(rcv_qids) == len(sto_qids) == self.eff_load:
                            self._set_mssg_to_ip_id(["SDC:sync_rtrv_rec",
                                                     "SDC:sync_rtrv_sto"])
                        else:
                            raise ValueError("Sync retrieving of pairs: frames"
                                             " has not the same amount of "
                                             "pairs.")
                    rcv_qid = rcv_qids.pop(0)
                    sto_qid = sto_qids.pop(0)
                    rcv_half = self.host.get_epr(host_id=self.partner_host_id,
                                                 q_id=rcv_qid)
                    sto_half = self.host.get_epr(host_id=self.partner_host_id,
                                                 q_id=sto_qid)
                    if len(rcv_qids) == len(sto_qids) == 0:
                        t_end = time.perf_counter() - self.start_time
                        actualize_df = pd.DataFrame([[t_end]],
                                                    columns=["t_end_sdc"])
                        self._actualize_histories(df_to_add=actualize_df,
                                                  kind="epr")
                        self._actualize_after_popdrop_id(id_f=ip_ids[0])
                        self._actualize_after_popdrop_id(id_f=ip_ids[1])
                        self._reset_ip()
                    return rcv_half, sto_half
            else:
                raise BadPhaseCallException("pop_sync_sdc_end_phase can "
                                            "repeatedly be called only after "
                                            "correct_epr_as_sdc_epr_phase_2 "
                                            "until all stored pairs were "
                                            "retrieved.")
        else:
            raise IsNotReceiversITFCException("Actual method can be called "
                                              "only if the interface owner is "
                                              "receiver.")

    # ***************************************************
    # ** Methods for Phases of SDC-Frame Communication **
    # ***************************************************
    
    def nuid_sdc_start(self):
        """starts the sdc-frame communnication process.
        It returns the frame identifier (id) of the next(n) to be 
        used(u) stored frame."""
        if not self.in_process:
            nuid = self._get_nxt_uid()
            fest = self._get_f_from_fr_id(id_f=nuid)
            t_init = time.perf_counter() - self.start_time
            actualize_df = pd.DataFrame([[nuid, fest[0], t_init]],
                                            columns=["ID", "F_est", "t_init"])
            self._actualize_histories(df_to_add=actualize_df, kind="sdc")
            self._append_id_to_ip_id(id_f=nuid)
            self._set_mssg_to_ip_id(mssg="SDC:started")
            self.in_process = True
            return nuid
        else:
            raise InterfaceIsInProcessException("sdc_start cannot be called. "
                                                "Interface is already "
                                                "processing a frame.")


    def pop_sdc_end_phase(self):
        """pop the corresponding qubit from the entanglement manager to
        be used in the sdc-decoding process."""
        if (self.in_process and ((self._get_mssg_ip() == "SDC:started") 
                                 or (self._get_mssg_ip() == 
                                     "SDC:epr-retrieving"))):
            ip_id = self._get_id_ip()
            qids = self._get_qids_from_fr_id(id_f=ip_id)
            if len(qids) == self.eff_load:
                self._set_mssg_to_ip_id(mssg="SDC:epr-retrieving")
            qid = qids.pop(0)
            epr_half = self.host.get_epr(host_id=self.partner_host_id, 
                                         q_id=qid)
            if len(qids) == 0:
                t_end = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[t_end]], columns=["t_end_rtrv"])
                self._actualize_histories(df_to_add=actualize_df, kind="sdc")
                self._actualize_after_popdrop_id(id_f=ip_id)
                self._reset_ip()
            return epr_half
        else:
            raise BadPhaseCallException("sdc_end_phase can be repeatedly "
                                        "called only after sdc_start until "
                                        "stored frame was consumed.")


    def finish_sdc(self, val_c_info=None):
        if self.is_receiver:
            if val_c_info is None:
                raise ValueError("integer 0(false) or 1(true) for val_c_info "
                                 "validating the sdc decoded C-Info must be "
                                 "passed.")
            else:
                t_fin = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[t_fin, val_c_info]],
                                            columns=["t_finish",
                                                     "valid_C_info"])
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
            for fr_id in self.u_ids:
                f_est = self._get_f_from_fr_id(id_f=fr_id)
                if f_est[0] < f_thres:
                    self._drop_stored_epr_frame_id(fr_id)
        else:
            raise InterfaceIsInProcessException("New threshold fidelity can be"
                                                " applied only if interface is"
                                                " not in process.")