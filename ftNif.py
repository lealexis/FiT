
from qunetsim.components import Network, Host
from qunetsim.objects import Qubit, DaemonThread
import numpy as np 
import random
from PIL import Image
import time

import QuTils
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_itfc_FT_NON_IDEAL_FEU import EPR_buff_itfc
from rotational_error_class import Rot_Error
from pipelined_CQ_channel import QuPipe, ClassicPipe

from threading import Event 
from matplotlib import pyplot as plt
import math

"""An image of 560 bits is sent from Alice to Bob over a quantum channel using superdense coding after distributing 
 EPR-pairs between Alice and Bob. For this purpose a protocol is implemented where EPR-frames and SDC-frames are allowed
 between Alice and Bob; in the case of EPR-frame transmission, also the fidelity of the EPR-frame is estimated and 
 depending on a fidelity threshold the distributed EPR-pairs are stored or dropped. The stored EPR-pairs halves are then 
 retrieved from Alice's quantum memory using the entanglement manager. Chunks of the image as bits are encoded in the 
 EPR-pairs halves at Alice and set over to Bob in a SDC-frame. When Bob receives an SDC-frame he retrieves also the 
 corresponding EPR-pairs halves from his quantum memory using his entanglement manager and decodes the bits from it. 
 When the whole image is transmitted from Alice to Bob the simulation is finalized. The communication is half-duplex 
 from Alice to Bob and the protocol also covers for errors in the header of the quantum-frames, which differentiates 
 them between EPR-frame and SDC-frame. Classical feedback channel is ideal and no decoherence nor photon absorption is 
 considered in this model"""

PRED_LEN = True # whether to use predictive Q-frame length estimator specific for the channel quality of this simulation

if PRED_LEN:
    from pred_Q import pred_frame_len as frm_len_EST # measuring portion Q for the oscillating channel's quality
else:
    from meas_portion import frame_length_estimator as frm_len_EST # measuring portion without adaption to channel's
    # quality

INTER_CBIT_TIME = 0.03 # time between the bits of the feedback signals. 30 ms
INTER_QBIT_TIME = 0.018 # time between the qubits of the quantum frames. 18ms
EFF_LOAD = 40 # amount of EPR-pairs distributed in an EPR-frame. Use a length compatible with 560 bits, each EPR-pair
# transmit 2 bits. Ex:(7*40)*2 = 560
BASE_FIDELITY = 0.5 # Common knowledge about fidelity for Alice & Bob used to reduce the amount of bits in the protocol.
SIG = 0.15 # standard deviation for rotational angle of header qubit; if 0 implies a perfect header.
# thres for mario: 0.6 0.75 0.9
F_THRES = 0.75 # EPR-frames with a fidelity below this threshold will be dropped and above it will be stored.

global sent_mssgs # keeps track of the data sent from alice to bob
sent_mssgs = ""
global dcdd_mssgs # keeps track of decoded data at bob
dcdd_mssgs = ""
global finish_time # simulation timespan in seconds
finish_time = None
global PROCESS # list to store "EPR" or "SDC" used to determine which kind of quantum frame will be sent next
PROCESS = []
global SAVE_MARIO # to save final received image set to true
SAVE_MARIO = False

"""Definition of classical data to be sent from Alice to Bob. An image 
is used to be sent with entangled-assisted communication."""

im = Image.open("../../../Data_results/mario_sprite.bmp")
pixels = np.array(im)
im.close()
coloumns, rows, colors = pixels.shape
dtype = pixels.dtype

hashes = [] # used for image's binary preparation
hashes_ = []

palette = {} # used at receiving mario image
indices = {} # used to prepare image's binary
for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        hashes.append(hashed)
        hashes_ = hashes.copy()
        palette[hashed] = color

hashes = list(set(hashes))

for i, hashed in enumerate(hashes): # preparing for binary extraction
    indices[hashed] = i

def binaryTupleFromInteger(i):
    """Transforms integer into binary. Used to represent the image as binary."""
    return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])

def integerFromBinaryTuple(a, b):
    """Transforms binary into integer. Used to recreate received image from binary to int."""
    return a * 2 ** 1 + b * 2 ** 0

global received_mario # to store the received image
received_mario = np.zeros((coloumns, rows, colors), dtype=dtype) # empty array to store received data.

global im_2_send # holds image to be sent from Alice to Bob as binary
im_2_send = ""

for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        index = indices[hashed]
        b1, b2 = binaryTupleFromInteger(index) # transform to binary
        im_2_send += str(b1) + str(b2) # store as binary string

def int2bin(i,length):
    """integer into binary. Used to transmit fidelity as binary in classical feedback."""
    return [int(j) for j in list(bin(i)[2:].zfill(length))]

def bin2int(bin_str):
    """binary into integer. Used to interpret classical feedback containing fidelity. """
    return int(bin_str,2)

def sEPR_proto(host: Host, receiver_id , epr_gen: EPR_generator, 
                   qmem_itfc: EPR_buff_itfc, quPipeChann:QuPipe, 
                   clsicPipeChann: ClassicPipe, error_gen: Rot_Error, 
                   proto_finished:Event, on_demand:Event, epr_demand_end:Event,
                   len_est:frm_len_EST, sRand, F_thres=0.5, verbose_level=0):
    """Distributes an EPR-frame from Alice to Bob. Each time this method is used, an EPR-frame is
    generated at Alice and distributed to Bob and a fidelity estimation process takes place after this distribution"""

    F_ideal_BF_chann = 0  # tracks the fidelity of each generated EPR-pair before their distribution over noisy quantum
    # channel, ideal because the fidelity of each EPR pair is accessed without destroying them

    nfID = qmem_itfc.nfID_EPR_START()  # get ID of EPR-frame to be generated and distributed
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/EPR - send EPR-Frame nfID:{id_epr}".format(id_epr=nfID))
    
    head_qbit = QuTils.superposed_qubit(host, sigma=SIG)  # signals the quantum frame as EPR-frame
    head_qbit.send_to(receiver_id=receiver_id)  # add receiver to qubit
    error_gen.apply_error(flying_qubit=head_qbit)  # simulating noisy channel error
    quPipeChann.put(head_qbit)  # passing header qubit to channel
    frame_len = len_est.get_frame_len()  # length of EPR-Frame

    for i in range(frame_len):
        q1, q2 = epr_gen.get_EPR_pair()  # EPR-pair is generated in loop
        F_ideal_BF_chann += epr_gen.get_fidelity(half=q1)  # tracks fidelity
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)  # local EPR-pair half
        if verbose_level == 1:
            print("ALICE/EPR - send EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)  # add receiver to EPR-pair half
        error_gen.apply_error(flying_qubit=q2)  # simulating noisy channel error
        quPipeChann.put(q2)  # passing EPR-pair half to channel
        time.sleep(0.002)  # wait for receiver to be ready
    F_ideal_BF_chann = (F_ideal_BF_chann / frame_len)  # Average fidelity of EPR-frame before noisy channel

    qmem_itfc.set_F_EPR_END_PHASE(F_est=F_ideal_BF_chann, to_history=True)  # keeps tracks of fidelity before channel

    # Alice waits for feedback from Bob
    cbit_counter = 0
    clsicPipeChann.fst_feedback_in_trans.wait()  # Hear classic channel for feedback
    ip_qids = None  # for qubit ids of local EPR-pairs halves
    recv_meas = []  # to store received Bob's measurement outputs
    d_nack = False
    f_thres_recv = ""  # to store binary fidelity threshold from Bob
    THRES = None  # to store decided Threshold
    while True:
        try:
            bit = clsicPipeChann.out_socket.pop() # persistent hearing of classical feedback channel
        except IndexError:
            continue
        else:  # a bit was received from the classical feedback channel
            cbit_counter += 1
            if cbit_counter == 1:  # interpret feedback header
                if bit == 0:  # D-NACK
                    ipID, nuID = qmem_itfc.drop_DNACK_EPR_END_PHASE()  # deleting corresponding local EPR-pairs halves
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv D-NACK: dropping (nfID,nuID)=({ip},{u})".format(ip=ipID, u=nuID))

                    clsicPipeChann.feedback_num = 0
                    d_nack = True
                    break

                else:  # EPR-FEEDBACK
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv EPR-Feedback")
                    ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()  # qubit ids of local EPR-pairs halves
                    continue
            else:
                if cbit_counter < 8:  # keep receiving bits of Bob's fidelity threshold
                    f_thres_recv += str(bit)
                else:
                    recv_meas.append(bit)  # receive Bob's measurement output

                if clsicPipeChann.fst_feedback_in_trans.is_set():  # there is still incoming feedback bits
                    continue
                else:  # first feedback was completely received
                    break
    if d_nack:  # wait accordingly before finalizing sEPR_proto()
        if on_demand.is_set():
            epr_demand_end.wait()
        else:
            proto_finished.wait()
    else:
        f_thres_recv = (bin2int(f_thres_recv) + 50) / 100  # Bob's received fidelity threshold. 0,5 is basis threshold.
        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - random measurements")
        meas_amount = len(ip_qids) - qmem_itfc.eff_load  # raw load - eff load = amount of measurements
        meas_qids = sRand.sample(ip_qids, int(meas_amount))  # random qubits for fidelity estimation: X-,Y-,& Z-Meas.
        # EPR-FEEDBACK measurement order [mx , my, mz]
        local_meas = []

        # random X-Measurements
        x_qids = sRand.sample(meas_qids, int(meas_amount/3))  # 1/3 of meas_amount to be X-Measured
        for idq in x_qids:  # after loop meas_qids contains 1/3 less qubits
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            mq.H()  # Apply hadamard to implement X-Meas
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output
            meas_qids.remove(idq)  # remove idq from meas_qids

        # random Y-Measurements
        s_dagger_mtrx = np.array([[1,0],[0, -1j]])  # needed for Y-Meas
        y_qids = sRand.sample(meas_qids, int(len(meas_qids)/2))  # take half of meas_qids, this is 1/3 of the original
        for idq in y_qids:  # after loop meas_qids contains tha last 1/3 of qubits to be measured
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            mq.custom_gate(s_dagger_mtrx)  # Apply s_dagger and hadamard to implement Y-Meas
            mq.H()
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output
            meas_qids.remove(idq)  # remove idq from measuring list

        # random Z-Measurements
        for idq in meas_qids:  # use the rest qids for Z-Meas, this is 1/3 of the original
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output

        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - Fidelity estimation")

        # Fidelity estimation after measurements by qubit error rate for X-, Y-, and Z-measurement
        meas_count = 0 
        mxyz_len = len(meas_qids)  # amount of X, Y, and Z measurement outputs
        qberx = 0  # initial values of 0 for qber
        qbery = 0 
        qberz = 0

        for idx in range(len(recv_meas)): # calculate different QBER looping over measurement outputs
            meas_count += 1
            if meas_count < (mxyz_len + 1):  # first 1/3 of measurements are X-Measurements outputs
                if recv_meas[idx] == local_meas[idx]:  # both equal --> no error
                    continue
                else:  # different --> error
                    qberx += 1
            elif mxyz_len < meas_count < (2*mxyz_len + 1):  # 2nd 1/3 of measurements are Y-Measurements outputs
                if recv_meas[idx] == local_meas[idx]:  # both equal --> error
                    qbery += 1
                else:  # different --> no error
                    continue
            else: # last third are Z-Measurements outputs
                if recv_meas[idx] == local_meas[idx]:  # both equal --> no error
                    continue
                else:  # different --> error
                    qberz += 1
        # divide by amount of measurements and QBER is calculated for X, Y and Z measurements
        qberx = qberx / mxyz_len
        qbery = qbery / mxyz_len
        qberz = qberz / mxyz_len

        # as in paper of dahlberg
        F_est_dahl = 1 - (qberx + qbery + qberz) / 2
        print("ALICE/EPR - Dahlberg estimated Fidelity: {}".format(F_est_dahl))
        
        F_send = math.trunc(F_est_dahl * 1000)  # take first three decimals
        F_EST = F_send / 1000  # to be set on local entanglement manager
        F_send = F_send - 500  # to be set to BOB

        if F_thres > f_thres_recv: # use Alice's or Bob's Fidelity threshold
            THRES =  F_thres
        else:
            THRES = f_thres_recv # Bob's threshold

        if THRES <= F_EST:  # ACK
            qmem_itfc.set_F_EPR_END_PHASE(F_est=F_EST)  # set estimated fidelity to remaining EPR-pairs
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - uID:{id_u} - Fid:{f}".format(id_u=nfID, f=F_EST))
                print("ALICE/EPR - send EPR-ACK")
            # The epr-ack is composed as:  [1, F_est]
            epr_ack = [1]
            epr_ack.extend(int2bin(F_send, 9))
            for bit in epr_ack: # send epr-ack over channel bit by bit
                clsicPipeChann.put(bit)
                time.sleep(INTER_CBIT_TIME)

        else:  # NACK
            qmem_itfc.drop_ip_frame_EPR_END_PHASE(fest=F_EST)  # drop remaining EPR-pairs because of low fidelity
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - F_est < F_thres:{fest} < {fth}".format(fest=F_EST, fth=THRES))
                print("ALICE/EPR - send NACK")
            clsicPipeChann.put(0)  # send NACK to Bob

        if on_demand.is_set():  # wait accordingly before finalizing sEPR_proto()
            epr_demand_end.wait()
        else:
            proto_finished.wait()

# im_2_send binary string to be sent
# SDC-encoded C-Info will always arrive and be decoded at BOB
def sSDC_proto(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, 
                      quPipeChann:QuPipe, clsicPipeChann: ClassicPipe, 
                      error_gen: Rot_Error, proto_finished:Event, 
                      verbose_level=0):
    """Send an SDC-frame from Alice to Bob. The binary information to be encoded into Alice's EPR-pairs halves is
    selected from im_2_send 2 bits at a time and are encoded in a single EPR-pair half."""
    global im_2_send
    global sent_mssgs
    sent_mssg = ""
    nuID = qmem_itfc.nuID_SDC_START()  # select EPR-frame with highest fidelity to SDC-encode
    if (verbose_level==0) or (verbose_level==1):
        print("ALICE/SDC - send SDC-Frame nuID:{id_u}".format(id_u=nuID))
    
    head_qbit = QuTils.superposed_qubit(host, sigma=SIG)  # noisy header qubit in |0>
    head_qbit.X()  # transform header qubit to state |1>
    error_gen.apply_error(flying_qubit=head_qbit)  # channel noise
    head_qbit.send_to(receiver_id=receiver_id)  # change owner of qubit, not actually sending
    quPipeChann.put(head_qbit)  # sending qubit to Bob
    time.sleep(INTER_QBIT_TIME)  # prevent overloading channel, due to receiver processing time
    for mi in range(qmem_itfc.eff_load):
        # taking the first two bits & deleting them from im_2_send, they are added to sent mssg
        # and further encoded into the EPR-pair half before sending them over the channel
        msg = im_2_send[0:2]
        im_2_send = im_2_send[2:]
        sent_mssg += msg
        q_sdc = qmem_itfc.pop_SDC_END_PHASE()
        QuTils.dens_encode(q_sdc, msg)
        if verbose_level==1:
            print("ALICE/SDC - send SDC-encoded epr half Nr:{q_i}".format(q_i=mi))
        q_sdc.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q_sdc)  # channel noise
        quPipeChann.put(q_sdc)  # sending qubit through channel
        time.sleep(INTER_QBIT_TIME)  # prevent overloading of channel
    sent_mssgs += sent_mssg  # actualize sent messages
    if (verbose_level==0) or (verbose_level==1):
        print("ALICE/SDC - send C-Info: {cmsg}".format(cmsg=sent_mssg))
    proto_finished.wait()  # wait for receiver to finish
    qmem_itfc.finish_SDC()  # finish process in entanglement manager

def sender_protocol(host: Host, receiver_id, epr_gen: EPR_generator,
                    qmem_itfc: EPR_buff_itfc, QpipeChann: QuPipe,
                    CpipeChann: ClassicPipe, error_gen: Rot_Error, 
                    proto_finished: Event, on_demand: Event, epr_demand_end: Event,
                    finish_simu: Event, len_est: frm_len_EST, sRandom, F_thres=0.5,
                    verbose_level=0):
    """Alice protocol to transmit EPR- and SDC-frames depending on which kind of Job is in PROCESS"""

    if (F_thres < 0.5) or (F_thres > 1):
        raise ValueError("F_thres must live in region 0.5 <= F_thres < 1")
    
    global PROCESS
    sRandom.seed(1100)  # same seed used in Alice & Bob for random processes in fidelity estimation
    while True:
        try:
            process_type = PROCESS.pop()
        except IndexError:
            continue
        else:
            if process_type == "EPR":
                if qmem_itfc.is_full: # go back to get next element from PROCESS
                    print("ALICE/EPR - Memory interface is full.\n")
                    continue
                else:
                    proto_finished.clear()  # event is set back to 0
                    print("\nALICE/EPR - Starting EPR-Frame distribution.")
                    sEPR_proto(host, receiver_id , epr_gen, qmem_itfc, 
                               QpipeChann, CpipeChann, error_gen, 
                               proto_finished, on_demand, epr_demand_end,
                               len_est, sRandom, F_thres, verbose_level)
                    continue  # go back to get next element from PROCESS
            elif process_type == "SDC":
                if qmem_itfc.is_empty:  # ON DEMAND EPR THEN SDC
                    proto_finished.clear()  # event is set back to 0

                    on_demand.set()  # event set to 1
                    att_nr = 0  # count attempts to distribute an EPR-frame

                    while True:  # Distribute one EPR-frame
                        att_nr += 1 
                        print("\nALICE/SDC - ON DEMAND EPR distribution attempt {}\n".format(att_nr))
                        sEPR_proto(host, receiver_id , epr_gen, qmem_itfc, 
                                   QpipeChann, CpipeChann, error_gen, 
                                   proto_finished, on_demand, epr_demand_end,
                                   len_est, sRandom, F_thres, verbose_level)
                        epr_demand_end.clear()  # set to 0
                        if not qmem_itfc.is_empty:  # EPR-frame was distributed
                            break
                        else:  # keep trying to distribute an EPR-frame
                            continue

                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, verbose_level)  # Send SDC-frame
                    on_demand.clear()
                    continue  # go back to get next element from PROCESS
                else:  # send SDC-Frame
                    proto_finished.clear()
                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, verbose_level)
                    continue  # go back to get next element from PROCESS
            else:  # Image was completely sent from Alice to Bob
                finish_simu.set()
                break
                

def put_next_process(proto_finished: Event, job_prob=0.5):
    """Whenever the protocol finishes the ditribution of an EPR- or SDC-frame, signaled by the
    proto_finished event, it will be appended to PROCESS either SDC or EPR. SDC is chosen
    accordingly to the job arrival probability job_prob.
     """
    global PROCESS
    while True:
        proto_finished.wait()
        if len(PROCESS) == 0:
            job_prob_var = random.random()
            if job_prob_var > (1 - job_prob):
                PROCESS.append("SDC")
            else:
                PROCESS.append("EPR")
        continue

def receiver_protocol(host:Host, qmem_itfc: EPR_buff_itfc, quPipeChann: QuPipe,
                      clsicPipeChann: ClassicPipe, proto_finished: Event,
                      on_demand: Event, epr_demand_end: Event, rRandom, F_thres=0.5,
                      verbose_level=0):
    """A quantum frame is received from the quPipeChann and the header is measured to decide if quantum frame is an
    EPR-frame or SDC-frame. Then the protocol runs correspondingly interpreting the quantum load as EPR or SDC and
    sending classical feedback's between Alice and Bob until the protocol is finalized and the complete quantum load is
    validated."""

    if (F_thres < 0.5) or (F_thres > 1):
        raise ValueError("F_thres must live in region 0.5 <= F_thres =< 1")

    # for epr-feedback
    F_thres = math.trunc(F_thres * 100) - 50  # basis fidelity of 0.5 is assumed
    F_thres = int2bin(F_thres, 6) 

    global im_2_send
    global dcdd_mssgs  # all the SDC-decoded (received) messages at Bob, works as Bob's classical memory
    global finish_time  # to track duration of the simulation
    global SAVE_MARIO  # bool, store or not store received image
    global received_mario
    global PROCESS  # used to enforce finalization of the simulation

    rRandom.seed(1100)  # same seed used in Alice & Bob for random processes in fidelity estimation
    count = 0  # store amount of received qubits in a quantum frame
    Type = 0  # 0 means epr frame, 1 means data frame
    dcdd_mssg = None
    f_est_ideal = 0
    frame_id = None
    qbit = None
    while True:
        try:  # continuous hearing of quantum channel
            qbit = quPipeChann.out_socket.pop()
        except IndexError:
            continue
        else:  # qubit was received, proceed to interpret it
            count += 1
            if count == 1:
                head = qbit.measure()
                if head == 1:  # SDC-Frame
                    if qmem_itfc.is_empty:  # Header was corrupted, it is an EPR-Frame
                        Type = 0
                        # TODO: change name from qmem_itfc to entanglement manager
                        frame_id = qmem_itfc.nfID_EPR_START()  # id of EPR-frame being received
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is stored EPR-Frames, interpret payload as SDC-Frame
                    else:
                        Type=1
                        dcdd_mssg = ""  # To store SDC-decoded message
                        frame_id = qmem_itfc.nuID_SDC_START()  # id of stored EPR-frame used for SDC-decoding
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                # EPR-Frame
                else:
                    # Header was corrupted, it must have been SDC-Frame
                    if qmem_itfc.is_full:
                        Type=1
                        dcdd_mssg = ""  # To store SDC-decoded message
                        frame_id = qmem_itfc.nuID_SDC_START()  # id of stored EPR-frame used for SDC-decoding
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is space in entanglement manager, interpret payload as EPR-Frame
                    else:
                        Type = 0                        
                        frame_id = qmem_itfc.nfID_EPR_START()  # id of EPR-frame being received
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
            else:
                if Type == 0:  # receive EPR-frame
                    if verbose_level == 1: 
                        print("BOB  /EPR - recv EPR halve Nr: {hnum}".format(hnum=(count - 1)))
                    f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)  # Fidelity of EACH distributed EPR-pair --> ideal
                    qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)
                else:  # receive SDC-frame
                    if verbose_level==1:
                        print("BOB  /SDC - recv SDC-encoded epr half Nr:{q_i}".format(q_i=(count - 1)))
                    if qmem_itfc.in_process:  # in_process = there is still EPR-halves to decode messages
                        retrieved_epr_half = qmem_itfc.pop_SDC_END_PHASE()
                        decoded_string = QuTils.dense_decode(retrieved_epr_half, qbit)
                        dcdd_mssg += decoded_string
                    else:  # all EPR-halves were consumed, destroy qubits by measuring
                        qbit.measure()

                # Quantum frame is still in transmission
                if quPipeChann.Qframe_in_transmission.is_set():
                    continue
                # Quantum frame was  completely received
                else:
                    if Type == 1:
                        if count == (qmem_itfc.eff_load + 1):  # SDC-Frame
                            if (verbose_level == 0) or (verbose_level == 1):   
                                    print("BOB  /SDC - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                            qmem_itfc.finish_SDC(val_C_info=int(1))  # decoded information is valid
                            dcdd_mssgs += dcdd_mssg
                            # IMAGE was completely sent ---> Process data & STOP SIMULATION
                            if len(im_2_send) == 0:
                                finish_time = time.time()
                                print("SIMULATION END: Image was completely sent!")
                                received_indexes = []
                                c1 = 0 
                                c2 = 0
                                for i in range(int(len(dcdd_mssgs)/2)):
                                    c1, c2 = tuple(map(int, dcdd_mssgs[i*2:i*2 +2]))
                                    rec_index = integerFromBinaryTuple(c1, c2)
                                    received_indexes.append(rec_index)

                                for row in range(rows):
                                    for column in range(coloumns):
                                        received_hash = hashes[received_indexes[0]]
                                        received_color = palette[received_hash]
                                        received_mario[column, row, :] = received_color
                                        received_indexes.pop(0)

                                received_im = Image.fromarray(received_mario)

                                if SAVE_MARIO:
                                    received_im.save('./mario_link_varying_fid_and_channel.bmp')
                                PROCESS.append(None)  # to stop simulation add None to process

                            dcdd_mssg = None  # restart variables
                            frame_id = None
                            proto_finished.set()  # protocol finalized
                        else:  # D-NACK ---> EPR-frame received as SDC-frame
                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB  /SDC - recv Payload length({frln}) > eff load({eff}).".format(frln=count-1, 
                                                                                                eff=qmem_itfc.eff_load))
                                print("BOB  /SDC - ===> It was an EPR-Frame")
                                print("BOB  /SDC - dropping decoded C-Info.")
                                print("BOB  /SDC - send D-NACK.")
                            qmem_itfc.finish_SDC(val_C_info=int(0))  # decoded information is invalid
                            dcdd_mssg = None
                            frame_id = None

                            # TODO: possible timming problems
                            clsicPipeChann.put(0)  # D-NACK = [0]
                            if on_demand.is_set():  # manage events to coordinate simulation
                                epr_demand_end.set()
                            else:
                                proto_finished.set()                        
                        count = 0 
                        continue  # Go back to receive next quantum frame

                    else:  # EPR-Frame
                        if count == (qmem_itfc.eff_load + 1):  # SDC-frame was received as EPR-frame, correcting it
                            dcdd_mssg = ""
                            uID = qmem_itfc.nuID_CORRECT_epr_as_sdc_EPR_PHASE_2()
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB/EPR---> BOB/SDC - recv EPR-Frame len:{fl} ==> SDC-Frame nuID:{id_u}".format(fl=(count-1), id_u=uID))

                            while qmem_itfc.in_process:
                                recv_qbit, rtrv_qbit = qmem_itfc.pop_sync_SDC_END_PHASE()  # get EPR-Pair
                                decoded_string = QuTils.dense_decode(rtrv_qbit, recv_qbit)  # decode information
                                dcdd_mssg += decoded_string

                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB  /SDC - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                            # TODO: change dcdd_mssgs for c_mem
                            dcdd_mssgs += dcdd_mssg
                                    
                            # IMAGE was completely sent ---> STOP SIMULATION
                            if len(im_2_send) == 0:
                                # TODO: create function performing this repeated steps and replace it where needed
                                finish_time = time.time()
                                print("SIMULATION END: Image was completely sent!")
                                received_indexes = []
                                c1 = 0 
                                c2 = 0
                                for i in range(int(len(dcdd_mssgs)/2)):
                                    c1, c2 = tuple(map(int, dcdd_mssgs[i*2:i*2 +2]))
                                    rec_index = integerFromBinaryTuple(c1, c2)
                                    received_indexes.append(rec_index)

                                for row in range(rows):
                                    for column in range(coloumns):
                                        received_hash = hashes[received_indexes[0]]
                                        received_color = palette[received_hash]
                                        received_mario[column, row, :] = received_color
                                        received_indexes.pop(0)

                                received_im = Image.fromarray(received_mario)

                                if SAVE_MARIO:
                                    received_im.save('./mario_link_varying_fid_and_channel.bmp')
                                # PROCESS=[None] stops simulation
                                PROCESS.append(None)
                            f_est_ideal = 0
                            dcdd_mssg = None
                            frame_id = None
                            proto_finished.set()  # set event

                        else:  # EPR-Frame
                            f_est_ideal = (f_est_ideal /(count -1))  # (ideally estimated) average Fidelity of EPR-Frame
                            print("BOB  /EPR - Ideal fidelity estimation: {}".format(f_est_ideal))

                            # set ideally estimated EPR-Frame's fidelity to entanglement manager
                            qmem_itfc.set_F_EPR_END_PHASE(F_est=f_est_ideal, to_history=True)
                            ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()  # in process (ip) qubit identifiers (qids)

                            meas_amount = len(ip_qids) - qmem_itfc.eff_load  # always divisible by 3 (X-, Y- & Z-Meas)
                            
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /EPR - {} qubits in payload".format(count -1))
                                print("BOB  /EPR - {} random measurements to be performed".format(meas_amount))

                            meas_qids = rRandom.sample(ip_qids, int(meas_amount))  # random qubits to be measured
                            
                            # Composition of EPR-Feedback: [1, F_thres_Bob ,mx , my, mz]
                            epr_feed = [1]
                            epr_feed.extend(F_thres)

                            #TODO: perform measurements in a single loop to simplify code

                            # X-MEASUREMENT
                            x_qids = rRandom.sample(meas_qids, int(meas_amount/3))  # random qubits for X-Measurement
                            for idq in x_qids:  # perform X-Measurement and add it to epr feedback
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit)
                                meas_qids.remove(idq)  # remove measured qubit from list of qubits to measure

                            # Y-MEASUREMENT
                            s_dagger = np.array([[1, 0], [0, -1j]])
                            y_qids = rRandom.sample(meas_qids, int(len(meas_qids)/2))  # random qubits for Y-Measurement
                            for idq in y_qids:  # perform Y-Measurement and add it to epr feedback
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                mq.custom_gate(s_dagger)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit) 
                                meas_qids.remove(idq)  # remove measured qubit from list of qubits to measure

                            # Z-MEASUREMENT
                            # left qubits in "meas_qids" were already randomly chosen
                            for idq in meas_qids:  # perform Z-Measurement and add it to epr feedback
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                bit = mq.measure()
                                epr_feed.append(bit)
                            # TODO: delete "meas_qids" to free memory

                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB  /EPR - send EPR-Feedback")

                            for bit in epr_feed:  # send EPR-FEEDBACK [1, F_thres_Bob, mx, my, mz]
                                clsicPipeChann.put(bit)
                                time.sleep(INTER_CBIT_TIME)  # give time to Alice to process the received cbit

                            clsicPipeChann.snd_feedback_in_trans.wait()  # When Alice send response feedback, continue

                            F_bits = ""
                            cbit_counter = 0
                            while True:
                                try:
                                    bit = clsicPipeChann.out_socket.pop()  # receive feedback bits from alice
                                except IndexError:
                                    continue
                                else:
                                    cbit_counter += 1
                                    if cbit_counter == 1:
                                        if bit == 0:  # NACK
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB  /EPR - recv NACK")
                                            qmem_itfc.drop_ip_frame_EPR_END_PHASE()
                                            break

                                        else:  # ACK
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB  /EPR - recv ACK")
                                            continue
                                    else:
                                        F_bits += str(bit)
                                        if clsicPipeChann.snd_feedback_in_trans.is_set():  # feedback is being sent
                                            continue
                                        else:  # feedback was completely sent
                                            F_est = (bin2int(F_bits) + 500)/1000  # estimated fidelity
                                            qmem_itfc.set_F_EPR_END_PHASE(F_est=F_est)
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB  /EPR - uID:{id_u} - Fid:{f}".format(id_u=frame_id, f=F_est))
                                            break
                            if on_demand.is_set():
                                epr_demand_end.set()
                            else:
                                proto_finished.set()   
                        f_est_ideal = 0
                        count=0
                        continue  # protocol finalized-continue with outer while loop

def main():
    start_time = time.time()

    VERBOSE = True  # control console print statements
    SAVE_DATA = False
    FINAL_EXP = True
    EXP_1 = True
    PRINT_HIST = True

    FOR_PORTION_PLOT = False
    LOW_VAR = True  # if standard deviation is high or not
    q_meas = 0.2  # static measuring portion
    NUM = 1
    Job_arrival_prob = 0.5  # probability of sending an SDC-Frame from Alice to Bob

    # Establish network and nodes
    network = Network.get_instance()

    Alice = Host('A')
    Bob = Host('B')

    Alice.add_connection('B')
    Bob.add_connection('A')

    Alice.start()
    Bob.start()

    network.add_hosts([Alice, Bob])
    network.start()

    if FINAL_EXP:  # oscillation parameters passed to EPR-Pair generator and RXY-noise
        if EXP_1:
            # use min_fid = 0.5
            freq_mu = 1/750
            freq = 1/750
            mu_phi = 0
            gamm_mu_phi = np.pi
            phi = np.pi
        else:
            # use min_fid = 0.8
            freq_mu = 1/750
            freq = 1/1500
            mu_phi = np.pi
            gamm_mu_phi = np.pi
            phi = np.pi

    if FOR_PORTION_PLOT:  # special simulation to generate dataset
        # EPR-Pair generator parameters
        if LOW_VAR:
            # amplitude standard deviation as max and min point
            gen_mx_dev = 0.05
            gen_mn_dev = 0.015
        else:
            gen_mx_dev = 0.15
            gen_mn_dev = 0.05

        mean_fid = 0.95  # IN PROCESS: 0.6 0.8 0.95
        mean_mx = mean_fid + 0.04
        mean_mn = mean_fid - 0.04

        Alice_EPR_gen = EPR_generator(host=Alice, max_fid=mean_mx, min_fid=mean_mn,
                                       max_dev=gen_mx_dev, min_dev=gen_mn_dev, 
                                       typ="gaussian")
        Alice_EPR_gen.start()

        # Rotational XY parameters
        mx_rot = 0.2
        mn_rot = 0.02
        if LOW_VAR:
            mx_dev = 0.01
            mn_dev = 0.005
        else:
            mx_dev = 0.08
            mn_dev = 0.01
        
        freq_mu = 1/750
        freq = 1/750
        mu_phi = 0
        gamm_mu_phi = np.pi
        phi = np.pi
        mx_q = q_meas
        mn_q = q_meas - 0.1

    else:  # standard Simulation
        Alice_EPR_gen =  EPR_generator(host=Alice, max_fid=0.99, min_fid=0.5,
                                       max_dev=0.15, min_dev=0.015, f_mu=freq_mu, 
                                       f_sig=freq, mu_phase=mu_phi, sig_phase=phi)
        Alice_EPR_gen.start()

        # parameters for rotational XY error
        mx_rot = 0.45  # mean value
        mn_rot = 0.05  # mean value
        mx_dev = 0.1  # deviation
        mn_dev = 0.03  # deviation
        # parameters for frame length estimator
        mx_q = 0.7
        mn_q = 0.3

    rot_error = Rot_Error(max_rot=mx_rot, min_rot=mn_rot, max_dev=mx_dev, min_dev=mn_dev, 
                          f_mu=freq_mu, f_sig=freq, mu_phase=gamm_mu_phi, 
                          sig_phase=phi)
    rot_error.start_time = Alice_EPR_gen.start_time

    len_est = frm_len_EST(max_q=mx_q, min_q=mn_q, eff_load=EFF_LOAD, freq_q=freq, 
                          phase=phi)
    len_est.start_time = Alice_EPR_gen.start_time

    # pipelined classical and quantum channel
    delay = 4  # 1.2 minimum delay, implyis channel length in seconds
    Qpiped_channel = QuPipe(delay=delay)
    Cpiped_channel = ClassicPipe(delay=delay)

    # TODO: change Qumem ITFCs to entanglement managers
    # Qumem ITFCs

    # Alice
    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False, 
                                 eff_load=EFF_LOAD)
    qumem_itfc_A.start_time = Alice_EPR_gen.start_time

    # Bob
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True, 
                                 eff_load=EFF_LOAD)
    qumem_itfc_B.start_time = Alice_EPR_gen.start_time

    if VERBOSE:
        print("Host Alice and Bob started. Network started.")
        print("Starting communication: Alice sender, Bob receiver.\n")

    # defining necessary events
    frame_comm_finished = Event()  # protocol finalized
    on_demand_comm = Event()  # protocol in "on_demand" modus
    on_demand_epr_finished = Event()
    FINALIZE_simu = Event()

    # to manage random choices in sender and receiver
    send_random = random.Random()
    recv_random = random.Random() 

    # Fidelity thresholds
    F_thres_A = F_THRES
    F_thres_B = F_THRES


    DaemonThread(target=receiver_protocol, args=(Bob, qumem_itfc_B, 
                                Qpiped_channel, Cpiped_channel, 
                                frame_comm_finished, on_demand_comm, 
                                on_demand_epr_finished, recv_random, F_thres_B,
                                0)) 

    DaemonThread(target=put_next_process, args=(frame_comm_finished, Job_arrival_prob))

    DaemonThread(target=sender_protocol, args=(Alice, Bob.host_id, Alice_EPR_gen,
                            qumem_itfc_A, Qpiped_channel, Cpiped_channel, rot_error,
                            frame_comm_finished, on_demand_comm,

                            on_demand_epr_finished, FINALIZE_simu, len_est, 
                            send_random, F_thres_A, 0))


    frame_comm_finished.set()  # force beginning of communication
    FINALIZE_simu.wait()
    global finish_time

    print("\nSimulation time duration in seconds is: {t}".format(
                                                    t=finish_time - start_time))
    print("Alice sent the following classical bitstream:\n"
          "{bstrm}".format(bstrm=sent_mssgs))
    print("\nBob received the following classical bitstream:"
          "\n{rbstrm}".format(rbstrm=dcdd_mssgs))
    
    BASE_PATH = "./Analysis_plots/Proto_experiments_&_Plots/"

    if SAVE_DATA:
        if FINAL_EXP:
            if EXP_1:
                DATA_PATH = BASE_PATH + "final_exps/EXP_1/FT_NIF/data/"
                exp_typ = 1
            else:
                DATA_PATH = BASE_PATH + "final_exps/EXP_2/FT_NIF/data/"
                exp_typ = 2
        else:
            DATA_PATH = BASE_PATH + "preliminar_exps/FT_NIF/data/"
            exp_typ = "PRE"

        # QUMEM ITFCs HISTORY
        epr_hist_alice = DATA_PATH + "alice_epr_frame_history_exp_" + str(exp_typ) + ".csv"
        epr_hist_bob = DATA_PATH + "bob_epr_frame_history_exp_" + str(exp_typ) + ".csv"
        sdc_hist_alice = DATA_PATH + "alice_sdc_frame_history_exp_" + str(exp_typ) + ".csv"
        sdc_hist_bob = DATA_PATH + "bob_sdc_frame_history_exp_" + str(exp_typ) + ".csv" 
        input_hist_alice = DATA_PATH + "alice_in_halves_history_exp_" + str(exp_typ) + ".csv"
        input_hist_bob = DATA_PATH + "bob_in_halves_history_exp_" + str(exp_typ) + ".csv"

        qumem_itfc_A.EPR_frame_history.to_csv(epr_hist_alice)
        qumem_itfc_B.EPR_frame_history.to_csv(epr_hist_bob)
        qumem_itfc_A.SDC_frame_history.to_csv(sdc_hist_alice)
        qumem_itfc_B.SDC_frame_history.to_csv(sdc_hist_bob)
        qumem_itfc_A.In_halves_history.to_csv(input_hist_alice)
        qumem_itfc_B.In_halves_history.to_csv(input_hist_bob)

        # EPR GEN, APPLIED ERROR & MEASURING PORTION HYSTORIES
        alice_gen_hist = DATA_PATH + "alice_epr_generator_history_exp_" + str(exp_typ) + ".csv"
        error_hist = DATA_PATH + "applied_error_history_exp_" + str(exp_typ) + ".csv"
        meas_portion_hist =  DATA_PATH + "meas_portion_q_exp_" + str(exp_typ) + ".csv"

        Alice_EPR_gen.history.to_csv(alice_gen_hist)
        rot_error.history.to_csv(error_hist)
        len_est.history.to_csv(meas_portion_hist)


    
    if FOR_PORTION_PLOT:
        if LOW_VAR:     
            DATA_PATH = BASE_PATH + "Q_vs_Fid/DATA/L_VAR/" + str(q_meas)
        else:
            DATA_PATH = BASE_PATH + "Q_vs_Fid/DATA/H_VAR/" + str(q_meas)

        epr_hist_alice = DATA_PATH + "/alice_epr_frame_history_" + str(NUM) + ".csv"
        epr_hist_bob = DATA_PATH + "/bob_epr_frame_history_" +  str(NUM) + ".csv"
        meas_portion_hist =  DATA_PATH + "/meas_portion_q_" + str(NUM) + ".csv"

        qumem_itfc_A.EPR_frame_history.to_csv(epr_hist_alice)
        qumem_itfc_B.EPR_frame_history.to_csv(epr_hist_bob)
        len_est.history.to_csv(meas_portion_hist)
    
    
    if PRINT_HIST:
        print(qumem_itfc_A.EPR_frame_history)
        print(qumem_itfc_B.EPR_frame_history)
        print(qumem_itfc_A.SDC_frame_history)
        print(qumem_itfc_B.SDC_frame_history)
        print(qumem_itfc_A.In_halves_history)
        print(qumem_itfc_B.In_halves_history)
    
    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()    

if __name__=='__main__':
    main()