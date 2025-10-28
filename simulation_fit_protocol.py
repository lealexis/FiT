"""An image of 560 bits is sent from Alice to Bob over a quantum channel
using superdense coding after distributing EPR-pairs(a.k.a. entanglement)
between Alice and Bob. For this purpose a protocol is implemented where
EPR-frames and SDC-frames are allowed between Alice and Bob; in the case
of EPR-frame transmission, also the fidelity of the EPR-frame is estimated
and depending on a fidelity threshold the distributed EPR-pairs are stored
or dropped. The stored EPR-pairs halves are then retrieved from Alice's
quantum memory using the entanglement manager. Chunks of the image as bits
are encoded in the EPR-pairs halves at Alice and set over to Bob in a
SDC-frame. When Bob receives an SDC-frame he retrieves also the
corresponding EPR-pairs halves from his quantum memory using his
entanglement manager and decodes the bits from it. When the whole image
is transmitted from Alice to Bob the simulation is finalized. The
communication is half-duplex from Alice to Bob and the protocol also
covers for errors in the header of the quantum-frames, which
differentiates them between EPR-frame and SDC-frame. Classical feedback
channel is ideal and no decoherence nor photon absorption is considered
in this model"""
import math
import random
import time
from threading import Event

import numpy as np
from PIL import Image
from qunetsim.components import Host, Network
from qunetsim.objects import DaemonThread, Qubit
from qutils import superposed_qubit, dens_encode, dense_decode
from epr_gen_class import EPRgenerator, epr_pair_fidelity
from qmem_manager import EntanglementManager as EntMngr
from rotational_error_class import RotError
from pipe_cq_channel import QuPipe as qPipe
from pipe_cq_channel import ClassicPipe as cPipe
from pred_q import PredFrameLen as FrameLenEst

INTER_CBIT_TIME = 0.03  # time between the bits of the feedback signals in ms
INTER_QBIT_TIME = 0.018  # time between the qubits of the quantum frames in ms

# amount of EPR-pairs distributed in an EPR-frame. Use a length compatible with 
# 560 bits, each EPR-pair transmit 2 bits. Ex:(7*40)*2 = 560
EFF_LOAD = 40

# Common knowledge about fidelity for Alice & Bob used to reduce the amount of 
# bits in the protocol.
BASE_FIDELITY = 0.5
SIG = 0.15  # std. dev. for rxy angle applied to header
F_THRES = 0.75  # fidelity threshold, to drop or store epr-frames
  
cmem_alice = ""  # keeps track of the data sent from alice to bob
cmem_bob = ""  # keeps track of decoded data at bob
finish_time = None  # simulation timespan in seconds
job_to_process = []  # "EPR" or "SDC" deciding which quantum frame will be sent

"""Definition of classical data to be sent from Alice to Bob. An image
is used to be sent with entangled-assisted communication."""
IMG_PATH = "./mario_sprite.bmp"
SAVE_IMG_PATH = './received_images/mario_over_qchannel.bmp'

def binary_tuple_from_integer(i):
    """Transforms integer into binary tuple. Used to represent the image
    as binary."""
    return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])


def integer_from_binary_tuple(a, b):
    """Transforms binary into integer. Used to recreate received image
    from binary to int."""
    return a * 2 ** 1 + b * 2 ** 0

im = Image.open(IMG_PATH)
pixels = np.array(im)
im.close()
coloumns, rows, colors = pixels.shape
dtype = pixels.dtype

hashes = []  # used for image's binary preparation
hashes_ = []

palette = {}  # used at receiving mario image
indices = {}  # used to prepare image's binary
for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        hashes.append(hashed)
        hashes_ = hashes.copy()
        palette[hashed] = color
hashes = list(set(hashes))
for i, hashed in enumerate(hashes):  # preparing for binary extraction
    indices[hashed] = i
im_2_send = ""  # holds image to be sent from Alice to Bob as binary
for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        index = indices[hashed]
        b1, b2 = binary_tuple_from_integer(index)  # transform to binary
        im_2_send += str(b1) + str(b2)  # store as binary string

received_image = np.zeros((coloumns, rows, colors), dtype=dtype)


def generate_im(binary):
    """Process received messages to regenerate image from binary data"""
    global received_image
    received_indexes = []
    c1 = 0
    c2 = 0
    for i in range(int(len(binary) / 2)):
        c1, c2 = tuple(map(int, binary[i * 2 : i * 2 + 2]))
        rec_index = integer_from_binary_tuple(c1, c2)
        received_indexes.append(rec_index)

    for row in range(rows):
        for column in range(coloumns):
            received_hash = hashes[received_indexes[0]]
            received_color = palette[received_hash]
            received_image[column, row, :] = received_color
            received_indexes.pop(0)

    return Image.fromarray(received_image)


def int2bin(i, length):
    """transform integer into binary.
    Used to interpret fidelity as binary in classical feedback."""
    return [int(j) for j in list(bin(i)[2:].zfill(length))]


def bin2int(bin_str):
    """transforms binary into integer.
    Used to interpret classical feedback containing fidelity. """
    return int(bin_str, 2)


def epr_proto(host: Host, receiver_id, epr_gen: EPRgenerator, eprMngr: EntMngr,
              q_channel: qPipe, c_channel: cPipe, error_gen: RotError,
              proto_finished: Event, on_demand: Event, epr_demand_end: Event,
              len_est: FrameLenEst, sRand, f_thres=0.5, verbose_level=0):
    """Distributes an EPR-frame from Alice to Bob. Each time this method
    is used, an EPR-frame is generated at Alice and distributed to Bob
    and a fidelity estimation process takes place after this
    distribution"""

    # tracks the fidelity of each generated EPR-pair before their distribution
    # over noisy quantum channel without destroying them.
    f_ideal_bf_chann = 0
    nfID = eprMngr.nfid_epr_start()  # get next frame identifier (nfID)
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/EPR - send EPR-Frame nfID:{id_epr}".format(id_epr=nfID))
    
    head_qbit = superposed_qubit(host, sigma=SIG)  # EPR-frame header
    head_qbit.send_to(receiver_id=receiver_id)  # add receiver id to qubit
    # TODO: integrate noisy channel into q_channel
    error_gen.apply_error(flying_qubit=head_qbit)  # noisy channel
    q_channel.put(head_qbit)  # passing header qubit to channel
    frame_len = len_est.get_frame_len()  # payload length

    for i in range(frame_len):
        q1, q2 = epr_gen.get_epr_pair()
        f_ideal_bf_chann += epr_gen.get_fidelity(half=q1)  # tracks fidelity
        eprMngr.store_epr_phase_1(epr_half=q1)  # store local payload
        if verbose_level == 1:
            print("ALICE/EPR - send EPR halve Nr:{pnum}".format(pnum=i + 1))
        q2.send_to(receiver_id=receiver_id)  # add receiver id
        # TODO: integrate noisy channel into q_channel
        error_gen.apply_error(flying_qubit=q2)
        q_channel.put(q2)  # passing EPR-pair half to channel
        time.sleep(INTER_QBIT_TIME)  # prevent overloading channel

    # Store average frame fidelity into entanglement manager
    f_ideal_bf_chann = (f_ideal_bf_chann / frame_len)
    eprMngr.set_f_epr_end_phase(F_est=f_ideal_bf_chann, to_history=True)
    cbit_counter = 0
    c_channel.fst_feedback_in_trans.wait()  # Waiting for feedback
    ip_qids = None  # in process qubit identifiers (ip_qids)
    recv_meas = []  # to store received Bob's measurement outputs
    d_nack = False
    f_thres_recv = ""  # to store binary fidelity threshold from Bob
    fid_thres = None  # to store decided fidelity Threshold
    THRESHOLD_BITS = 8
    HEADER_BIT = 1
    FID_BITS = 9
    while True:  # persistent hearing of classical feedback channel
        try:
            bit = c_channel.out_socket.pop()
        except IndexError:
            continue
        else:  # a bit was received from the classical feedback channel
            cbit_counter += 1
            if cbit_counter == HEADER_BIT:  # interpret feedback header
                if bit == 0:  # D-NACK
                    # delete corresponding local EPR-pairs halves
                    # in process id (ipid), next used id (nuid)
                    ipid, nuid = eprMngr.drop_dnack_epr_end_phase()
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv D-NACK: dropping (nfID,nuID)="
                        "({ip},{u})".format(ip=ipid, u=nuid))
                    c_channel.feedback_num = 0  # restore channel property
                    d_nack = True
                    break

                else:  # EPR-FEEDBACK
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv EPR-Feedback")
                    # load local ids of EPR-pairs halves to access them
                    ip_qids = eprMngr.get_qids_epr_phase_2()
                    continue
            else:  # receive feedback's payload
                if cbit_counter < THRESHOLD_BITS:  # Bob's fidelity threshold
                    f_thres_recv += str(bit)
                else:  # Bob's measurement output
                    recv_meas.append(bit)

                if c_channel.fst_feedback_in_trans.is_set():
                    continue  # keep receiving feedback
                else:  # first feedback was completely received
                    break

    if d_nack:  # wait accordingly before finalizing epr_proto()
        if on_demand.is_set():
            epr_demand_end.wait()
        else:
            proto_finished.wait()
    else:  # process the feedback's payload
        # Bob's threshold
        f_thres_recv = (bin2int(f_thres_recv) / 100) + BASE_FIDELITY
        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - random measurements")
        
        # raw load - eff load = amount of measurements
        meas_amount = len(ip_qids) - eprMngr.eff_load
        # randomly selected qubits to be measured for fidelity estimation
        meas_qids = sRand.sample(ip_qids, int(meas_amount))
        local_meas = []
        # measurement order [mx , my, mz]
        # random X-Measurements
        x_qids = sRand.sample(meas_qids, int(meas_amount / 3))
        for idq in x_qids:
            mq = eprMngr.get_epr_phase_3(idq)
            mq.H()  # Apply hadamard to implement X-Meas
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output
            meas_qids.remove(idq)  # remove idq from meas_qids

        # random Y-Measurements
        s_dagger_mtrx = np.array([[1, 0], [0, -1j]])  # for Y-Measurement
        y_qids = sRand.sample(meas_qids, int(len(meas_qids) / 2))
        for idq in y_qids:
            mq = eprMngr.get_epr_phase_3(idq)
            # Apply s_dagger and hadamard to implement Y-Measurement
            mq.custom_gate(s_dagger_mtrx)
            mq.H()
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output
            meas_qids.remove(idq)  # remove idq from measuring list

        # random Z-Measurements
        for idq in meas_qids:  # use the rest qids for Z-Measurements
            mq = eprMngr.get_epr_phase_3(idq)
            bit = mq.measure()
            local_meas.append(bit)  # store measurement output

        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - Fidelity estimation")

        # Fidelity estimation process
        meas_count = 0
        mxyz_len = len(meas_qids)  # number of measurement for each X, Y, and Z
        qberx = 0  # initial values of 0 for qubit error rates
        qbery = 0
        qberz = 0

        for idx in range(len(recv_meas)):  # calculate QBERs looping over
                                           # measurement outputs
            meas_count += 1
            if meas_count < (mxyz_len + 1):  # X-Measurements
                if recv_meas[idx] == local_meas[idx]:  # both equal, no error
                    continue
                else:  # error
                    qberx += 1
            elif mxyz_len < meas_count < (2 * mxyz_len + 1):  # Y-Measurements
                if recv_meas[idx] == local_meas[idx]:  # both equal, error
                    qbery += 1
                else:  # no error
                    continue
            else:  # Z-Measurements
                if recv_meas[idx] == local_meas[idx]:  # both equal, no error
                    continue
                else:  # error
                    qberz += 1
        # divide by amount of measurements and QBERs are calculated
        qberx = qberx / mxyz_len
        qbery = qbery / mxyz_len
        qberz = qberz / mxyz_len

        # Fidelity calculated as in paper of Dahlberg
        f_dahl = 1 - (qberx + qbery + qberz) / 2
        print("ALICE/EPR - Dahlberg estimated Fidelity: {}".format(f_dahl))
        f_send = math.trunc(f_dahl * 1000)  # take first three decimals
        f_est = f_send / 1000
        f_send = f_send - (BASE_FIDELITY * 1000)  # to be sent to BOB

        if f_thres > f_thres_recv:  # use Alice's fidelity threshold
            fid_thres = f_thres
        else:
            fid_thres = f_thres_recv  # use Bob's fidelity threshold

        if fid_thres <= f_est:  # send ACK feedback
            eprMngr.set_f_epr_end_phase(F_est=f_est)  # set fidelity
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - uID:{id_u} - Fid:{f}".format(id_u=nfID, 
                                                                f=f_est))
                print("ALICE/EPR - send EPR-ACK")
            # The epr-ack is composed as:  [1, F_est]
            epr_ack = [1]
            epr_ack.extend(int2bin(f_send, FID_BITS)) 
            for bit in epr_ack:  # send epr-ack over channel bit by bit
                c_channel.put(bit)
                time.sleep(INTER_CBIT_TIME)

        else:  # send NACK feedback
            # drop remaining EPR-pairs because of low fidelity
            eprMngr.drop_ip_id_epr_end_phase(fest=f_est)
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - F_est < F_thres:{fest} < {fth}".format(
                    fest=f_est, fth=fid_thres))
                print("ALICE/EPR - send NACK")
            c_channel.put(0)  # send NACK to Bob

        if on_demand.is_set():  # wait before finalizing epr_proto()
            epr_demand_end.wait()
        else:
            proto_finished.wait()


def sdc_proto(host: Host, receiver_id, eprMngr: EntMngr, q_channel: qPipe,
              c_channel: cPipe, error_gen: RotError, proto_finished: Event, 
              verbose_level=0):
    """Send an SDC-frame from Alice to Bob. The binary information to be
    encoded into Alice's EPR-pairs halves is selected from im_2_send, 2
    bits at a time, and are encoded in a single EPR-pair half."""
    
    global im_2_send  # binary string to be sent
    global cmem_alice
    sent_mssg = ""
    nuID = eprMngr.nuid_sdc_start()  # select EPR-frame used in SDC-encoding
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/SDC - send SDC-Frame nuID:{id_u}".format(id_u=nuID))
    
    head_qbit = superposed_qubit(host, sigma=SIG)  # header qubit in |0>
    head_qbit.X()  # transform header to |1> signaling SDC-Frame
    
    error_gen.apply_error(flying_qubit=head_qbit)  # channel noise
    # just adds receiver id to qubit, not sending.
    head_qbit.send_to(receiver_id=receiver_id)
    q_channel.put(head_qbit)  # sending qubit to Bob
    # prevent overloading channel, due to receiver processing time
    time.sleep(INTER_QBIT_TIME)
    for mi in range(eprMngr.eff_load):  # generate SDC-payload
        msg = im_2_send[0:2]  # take the first two bits
        im_2_send = im_2_send[2:]  # delete them from im_2_send
        sent_mssg += msg
        q_sdc = eprMngr.pop_sdc_end_phase()
        dens_encode(q_sdc, msg)  # SDC-encoding
        if verbose_level == 1:
            print("ALICE/SDC - send SDC-encoded epr half Nr:{q_i}".format(
                q_i=mi))
        q_sdc.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q_sdc)  # channel noise
        q_channel.put(q_sdc)  # sending qubit through channel
        time.sleep(INTER_QBIT_TIME)  # prevent channel overloading
    cmem_alice += sent_mssg  # actualize sent messages
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/SDC - send C-Info: {cmsg}".format(cmsg=sent_mssg))
    proto_finished.wait()  # wait for receiver to finish
    eprMngr.finish_sdc()  # finish process in entanglement manager


def sender_protocol(host: Host, receiver_id, epr_gen: EPRgenerator, 
                    eprMngr: EntMngr, q_channel: qPipe, c_channel: cPipe, 
                    error_gen: RotError, proto_finished: Event, 
                    on_demand: Event, epr_demand_end: Event, 
                    finish_simu: Event, len_est: FrameLenEst, sRandom, 
                    f_thres=0.5, verbose_level=0):
    """Alice protocol to transmit EPR- and SDC-frames depending on which
    kind of Job is in job_to_process"""

    if (f_thres < 0.5) or (f_thres > 1):
        raise ValueError("f_thres must live in region 0.5 <= f_thres < 1")
    
    global job_to_process  # stores Jobs, either EPR, SDC, or stop simulation

    # seed is used in Alice & Bob for random processes in fidelity estimation
    sRandom.seed(1100)
    while True:
        try:
            process_type = job_to_process.pop()
        except IndexError:
            continue
        else:
            if process_type == "EPR":
                if eprMngr.is_full:
                    print("ALICE/EPR - Memory is full.\n")
                    continue # get next element from job_to_process
                else:
                    proto_finished.clear()  # event is set back to 0
                    print("\nALICE/EPR - Starting EPR-Frame distribution.")
                    epr_proto(host, receiver_id, epr_gen, eprMngr,
                               q_channel, c_channel, error_gen,
                               proto_finished, on_demand, epr_demand_end,
                               len_est, sRandom, f_thres, verbose_level)
                    continue  # go back to get next element from job_to_process
            elif process_type == "SDC":
                if eprMngr.is_empty:  # ON DEMAND EPR THEN SDC
                    proto_finished.clear()  # event is set back to 0
                    on_demand.set()  # event set to 1
                    att_nr = 0  # count attempts to distribute an EPR-frame

                    while True:  # send EPR-frame
                        att_nr += 1
                        print("\nALICE/SDC - ON DEMAND EPR distribution "
                              "attempt {}\n".format(att_nr))
                        epr_proto(host, receiver_id, epr_gen, eprMngr,
                                   q_channel, c_channel, error_gen,
                                   proto_finished, on_demand, epr_demand_end,
                                   len_est, sRandom, f_thres, verbose_level)
                        epr_demand_end.clear()  # set to 0
                        if not eprMngr.is_empty:  # EPR-frame was distributed
                            break
                        else:  # keep trying to distribute an EPR-frame
                            continue

                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    # Send SDC-frame
                    sdc_proto(host, receiver_id, eprMngr, q_channel,
                               c_channel, error_gen, proto_finished,
                               verbose_level)
                    on_demand.clear()
                    continue  # get next element from job_to_process
                else:  # send SDC-Frame
                    proto_finished.clear()
                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sdc_proto(host, receiver_id, eprMngr, q_channel,
                               c_channel, error_gen, proto_finished,
                               verbose_level)
                    continue  # get next element from job_to_process
            else:  # Finish simulation
                finish_simu.set()
                break
                

def put_next_process(proto_finished: Event, job_prob=0.5):
    """Whenever the protocol finishes the ditribution of an EPR- or 
    SDC-frame, signaled by the proto_finished event, either SDC or EPR 
    will be appended to job_to_process as the next Job to be 
    accomplished. SDC is chosen accordingly to the job arrival 
    probability job_prob."""
    global job_to_process
    while True:
        proto_finished.wait()
        if len(job_to_process) == 0:
            job_prob_var = random.random()
            if job_prob_var > (1 - job_prob):
                job_to_process.append("SDC")
            else:
                job_to_process.append("EPR")
        continue


def receiver_protocol(host: Host, eprMngr: EntMngr, q_channel: qPipe,
                      c_channel: cPipe, proto_finished: Event, 
                      on_demand: Event, epr_demand_end: Event, rRandom, 
                      f_thres=0.5, verbose_level=0):
    """A quantum frame is received from the q_channel and the header is
    measured to decide if the quantum frame is an EPR- or SDC-frame. 
    Then the protocol runs correspondingly interpreting the quantum 
    payload as EPR or SDC and sending classical feedback's between Alice 
    and Bob until the protocol is finalized and the complete quantum 
    payload is validated."""

    if (f_thres < 0.5) or (f_thres > 1):
        raise ValueError("f_thres must live in region 0.5 <= f_thres =< 1")

    # for epr-feedback
    f_thres = math.trunc(f_thres * 100) - BASE_FIDELITY * 100 
    f_thres = int2bin(f_thres, 6)

    global im_2_send
    global cmem_bob  # stores all the SDC-decoded messages at Bob
    global finish_time  # tracks whole simulation time span
    global job_to_process  # used to enforce finalization of the simulation

    rRandom.seed(1100)  # same seed used in Alice & Bob for random processes
    count = 0  # track number of received qubits in a quantum frame
    frame_typ = "epr"
    dcdd_mssg = None
    f_est_ideal = 0 # track epr-frame average fidelity
    frame_id = None
    qbit = None

    while True: # continuous hearing of quantum channel
        try:
            qbit = q_channel.out_socket.pop()
        except IndexError:
            continue
        else:  # qubit was received, proceed to interpret it
            count += 1
            if count == 1:
                header_qbit = qbit.measure()
                if header_qbit == 1:  # SDC-Frame
                    if eprMngr.is_empty:  # Header is corrupted, EPR-frame
                        frame_typ = "epr"
                        frame_id = eprMngr.nfid_epr_start()  # EPR-frame id
                        if (verbose_level == 0) or (verbose_level == 1):
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}"
                                  .format(id_u=frame_id))
                        continue
                    else:  # interpret payload as SDC-Frame
                        frame_typ = "sdc"
                        dcdd_mssg = ""  # stores SDC-decoded message
                        frame_id = eprMngr.nuid_sdc_start()  # EPR-frame id
                        if (verbose_level == 0) or (verbose_level == 1):
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}"
                                  .format(id_u=frame_id))
                        continue
                else:  # EPR-Frame
                    if eprMngr.is_full:  # Header is corrupted, SDC-frame
                        frame_typ = "sdc"
                        dcdd_mssg = ""  # stores SDC-decoded message
                        frame_id = eprMngr.nuid_sdc_start()  # EPR-frame id
                        if (verbose_level == 0) or (verbose_level == 1):
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}"
                                  .format(id_u=frame_id))
                        continue
                    else:  # interpret payload as EPR-Frame
                        frame_typ = "epr"
                        frame_id = eprMngr.nfid_epr_start()  # EPR-frame id
                        if (verbose_level == 0) or (verbose_level == 1):
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}"
                                  .format(id_u=frame_id))
                        continue
            else:  # receive quantum payload
                if frame_typ == "epr":  # receive EPR-frame
                    if verbose_level == 1:
                        print("BOB  /EPR - recv EPR halve Nr: {hnum}".format(
                            hnum=(count - 1)))
                    f_est_ideal += epr_pair_fidelity(epr_halve=qbit)
                    eprMngr.store_epr_phase_1(epr_half=qbit)
                else:  # receive SDC-frame
                    if verbose_level == 1:
                        print("BOB  /SDC - recv SDC-encoded epr half Nr:{q_i}"
                              .format(q_i=(count - 1)))
                    if eprMngr.in_process:  # EPR-halves in qmem
                        retrieved_epr_half = eprMngr.pop_sdc_end_phase()
                        decoded_string = dense_decode(retrieved_epr_half, qbit)
                        dcdd_mssg += decoded_string
                    else:  # no EPR-halves, destroy qubits by measuring
                        qbit.measure()
                
                if q_channel.Qframe_in_transmission.is_set():
                    continue
                else:  # interpret quantum payload
                    if frame_typ == "sdc":  # SDC-frame
                        if count == (eprMngr.eff_load + 1):  # validity check
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /SDC - recv C-Info: {cmsg}".format(
                                    cmsg=dcdd_mssg))
                            
                            # decoded information is valid
                            eprMngr.finish_sdc(val_c_info=int(1))
                            cmem_bob += dcdd_mssg
                            if len(im_2_send) == 0:  # process simulation's end
                                finish_time = time.time()
                                print("SIMULATION END: Image was completely " 
                                      "sent!")
                                received_im = generate_im(cmem_bob)
                                received_im.save(SAVE_IMG_PATH)
                                job_to_process.append(None)  # stop simulation
                            
                            # restart variables
                            dcdd_mssg = None
                            frame_id = None
                            proto_finished.set()

                        else:  # D-NACK: EPR-frame received as SDC-frame
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /SDC - recv Payload length({frln})"
                                "> eff load({eff}).".format(frln=count - 1, eff
                                                            =eprMngr.eff_load))
                                print("BOB  /SDC - ===> It was an EPR-Frame")
                                print("BOB  /SDC - dropping decoded C-Info.")
                                print("BOB  /SDC - send D-NACK.")
                            
                            # invalid info
                            eprMngr.finish_sdc(val_c_info=int(0))
                            # restart variables
                            dcdd_mssg = None
                            frame_id = None

                            c_channel.put(0)  # send D-NACK = [0]
                            # manage events to coordinate simulation
                            if on_demand.is_set():
                                epr_demand_end.set()
                            else:
                                proto_finished.set()
                        count = 0
                        continue  # Go back to receive next quantum frame

                    else:  # EPR-Frame
                        # SDC-frame was received as EPR-frame, correcting it
                        if count == (eprMngr.eff_load + 1):
                            dcdd_mssg = ""
                            uID = eprMngr.correct_epr_as_sdc_epr_phase_2()
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB/EPR---> BOB/SDC - recv EPR-Frame "
                                      "len:{fl} ==> SDC-Frame nuID:{id_u}"
                                      .format(fl=(count - 1), id_u=uID))

                            while eprMngr.in_process:
                                recv_qbit, rtrv_qbit = eprMngr\
                                    .pop_sync_sdc_end_phase()  # get EPR-Pair
                                decoded_string = dense_decode(rtrv_qbit, 
                                                              recv_qbit)
                                dcdd_mssg += decoded_string

                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /SDC - recv C-Info: {cmsg}".format(
                                    cmsg=dcdd_mssg))
                            cmem_bob += dcdd_mssg
                                    
                            if len(im_2_send) == 0:  # simulation's end
                                finish_time = time.time()
                                print("SIMULATION END: Image was sent!")
                                received_im = generate_im(cmem_bob)
                                received_im.save(SAVE_IMG_PATH)
                                job_to_process.append(None) # stop simulation

                            # restart variables
                            f_est_ideal = 0
                            dcdd_mssg = None
                            frame_id = None
                            proto_finished.set()

                        else:  # EPR-Frame
                            f_est_ideal = (f_est_ideal / (count -1))
                            print("BOB  /EPR - Ideal fidelity estimation: {}"
                                  .format(f_est_ideal))

                            # store average EPR-Frame's fidelity
                            eprMngr.set_f_epr_end_phase(F_est=f_est_ideal, 
                                                        to_history=True)
                            
                            # in process (ip) qubit identifiers (qids)
                            ip_qids = eprMngr.get_qids_epr_phase_2()

                            meas_amount = len(ip_qids) - eprMngr.eff_load
                            
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /EPR - {} qubits in payload"
                                      .format(count -1))
                                print("BOB  /EPR - {} random measurements to "
                                      "be performed".format(meas_amount))
                            
                            # random selection of qubits to be measured
                            meas_qids = rRandom.sample(ip_qids, 
                                                       int(meas_amount))
                            
                            # Composition of EPR-Feedback:
                            # [1, F_thres_Bob ,mx , my, mz]
                            epr_feed = [1]
                            epr_feed.extend(f_thres)

                            # X-MEASUREMENT
                            x_qids = rRandom.sample(meas_qids, 
                                                    int(meas_amount / 3))
                            for idq in x_qids:
                                mq = eprMngr.get_epr_phase_3(idq)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit)
                                meas_qids.remove(idq)

                            # Y-MEASUREMENT
                            s_dagger = np.array([[1, 0], [0, -1j]])
                            y_qids = rRandom.sample(meas_qids, 
                                                    int(len(meas_qids) / 2))
                            for idq in y_qids:
                                mq = eprMngr.get_epr_phase_3(idq)
                                mq.custom_gate(s_dagger)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit)
                                meas_qids.remove(idq)

                            # Z-MEASUREMENT
                            # left qubits in "meas_qids" were implicitly
                            # randomly chosen
                            for idq in meas_qids:
                                mq = eprMngr.get_epr_phase_3(idq)
                                bit = mq.measure()
                                epr_feed.append(bit)

                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB  /EPR - send EPR-Feedback")

                            for bit in epr_feed:  # send EPR-FEEDBACK
                                c_channel.put(bit)
                                time.sleep(INTER_CBIT_TIME)

                            # Wait for Alice to send response feedback
                            c_channel.snd_feedback_in_trans.wait()

                            F_bits = ""
                            cbit_counter = 0
                            while True:
                                try:  # receive feedback bits from alice
                                    bit = c_channel.out_socket.pop()
                                except IndexError:
                                    continue
                                else:
                                    cbit_counter += 1
                                    if cbit_counter == 1:
                                        if bit == 0:  # NACK
                                            if ((verbose_level == 0) 
                                                or (verbose_level == 1)):
                                                print("BOB  /EPR - recv NACK")
                                            eprMngr.drop_ip_id_epr_end_phase()
                                            break

                                        else:  # ACK
                                            if ((verbose_level == 0) 
                                                or (verbose_level == 1)):
                                                print("BOB  /EPR - recv ACK")
                                            continue
                                    else:
                                        F_bits += str(bit)  # fidelity bits
                                        if (c_channel.snd_feedback_in_trans
                                            .is_set()):
                                            continue
                                        else:  # feedback was completely sent
                                            F_est = ((bin2int(F_bits) / 1000) 
                                                     + BASE_FIDELITY)
                                            eprMngr.set_f_epr_end_phase(
                                                F_est=F_est)
                                            if (verbose_level == 0) or (
                                                verbose_level == 1):
                                                print("BOB  /EPR - uID:{id_u} "
                                                      "- Fid:{f}".format(
                                                          id_u=frame_id, 
                                                          f=F_est))
                                            break
                            if on_demand.is_set():
                                epr_demand_end.set()
                            else:
                                proto_finished.set()
                        f_est_ideal = 0
                        count=0
                        continue  # continue receiving frames


def main():
    start_time = time.time()

    VERBOSE = True  # control console print statements
    SAVE_DATA = False
    PRINT_HIST = True
    Job_arrival_prob = 0.5  # probability of sending an SDC-Frame

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
    
    # oscillation parameters passed to EPR-Pair generator and RXY-noise
    freq_mu = 1 / 750
    freq = 1 / 750
    mu_phi = 0
    gamm_mu_phi = np.pi
    phi = np.pi
    
    Alice_EPR_gen = EPRgenerator(host=Alice, max_fid=0.99, min_fid=0.5,
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

    rot_error = RotError(max_rot=mx_rot, min_rot=mn_rot, max_dev=mx_dev,
                          min_dev=mn_dev, f_mu=freq_mu, f_sig=freq,
                          mu_phase=gamm_mu_phi, sig_phase=phi)
    rot_error.start_time = Alice_EPR_gen.start_time

    len_est = FrameLenEst(max_q=mx_q, min_q=mn_q, eff_load=EFF_LOAD,
                          freq_q=freq, phase=phi)
    len_est.start_time = Alice_EPR_gen.start_time

    # pipelined classical and quantum channel
    delay = 4  # 1.2 minimum delay, imply channel length in seconds
    Qpiped_channel = qPipe(delay=delay)
    Cpiped_channel = cPipe(delay=delay)

    # Entanglement Managers

    # Alice
    EntMngr_a = EntMngr(Alice, Bob.host_id, is_receiver=False,
                                 eff_load=EFF_LOAD)
    EntMngr_a.start_time = Alice_EPR_gen.start_time

    # Bob
    EntMngr_b = EntMngr(Bob, Alice.host_id, is_receiver=True,
                                 eff_load=EFF_LOAD)
    EntMngr_b.start_time = Alice_EPR_gen.start_time

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
    f_thres_a = F_THRES
    f_thres_b = F_THRES


    DaemonThread(target=receiver_protocol,
                 args=(Bob, EntMngr_b, Qpiped_channel, Cpiped_channel,
                       frame_comm_finished, on_demand_comm,
                       on_demand_epr_finished, recv_random, f_thres_b, 0))

    DaemonThread(target=put_next_process, args=(frame_comm_finished,
                                                Job_arrival_prob))

    DaemonThread(target=sender_protocol,
                 args=(Alice, Bob.host_id, Alice_EPR_gen, EntMngr_a,
                       Qpiped_channel, Cpiped_channel, rot_error,
                       frame_comm_finished, on_demand_comm,
                       on_demand_epr_finished, FINALIZE_simu, len_est,
                       send_random, f_thres_a, 0))


    frame_comm_finished.set()  # force beginning of communication
    FINALIZE_simu.wait()
    global finish_time
    global cmem_alice
    global cmem_bob

    print("\nSimulation time duration in seconds is: {t}"
          .format(t=finish_time - start_time))
    print("Alice sent the following classical bitstream:\n"
          "{bstrm}".format(bstrm=cmem_alice))
    print("\nBob received the following classical bitstream:"
          "\n{rbstrm}".format(rbstrm=cmem_bob))
    
    BASE_PATH = "./Analysis_plots/Proto_experiments_&_Plots/"

    if SAVE_DATA:
            
        DATA_PATH = BASE_PATH + "final_exps/EXP_1/FT_NIF/data/"
        exp_typ = 1

        # QUMEM ITFCs HISTORY
        epr_hist_alice = (DATA_PATH + "alice_epr_frame_history_exp_" 
                          + str(exp_typ) + ".csv")
        epr_hist_bob = (DATA_PATH + "bob_epr_frame_history_exp_" 
                        + str(exp_typ) + ".csv")
        sdc_hist_alice = (DATA_PATH + "alice_sdc_frame_history_exp_" 
                          + str(exp_typ) + ".csv")
        sdc_hist_bob = (DATA_PATH + "bob_sdc_frame_history_exp_" 
                        + str(exp_typ) + ".csv") 
        input_hist_alice = (DATA_PATH + "alice_in_halves_history_exp_" 
                            + str(exp_typ) + ".csv")
        input_hist_bob = (DATA_PATH + "bob_in_halves_history_exp_" 
                          + str(exp_typ) + ".csv")

        EntMngr_a.EPR_frame_history.to_csv(epr_hist_alice)
        EntMngr_b.EPR_frame_history.to_csv(epr_hist_bob)
        EntMngr_a.SDC_frame_history.to_csv(sdc_hist_alice)
        EntMngr_b.SDC_frame_history.to_csv(sdc_hist_bob)
        EntMngr_a.In_halves_history.to_csv(input_hist_alice)
        EntMngr_b.In_halves_history.to_csv(input_hist_bob)

        # EPR GEN, APPLIED ERROR & MEASURING PORTION HYSTORIES
        alice_gen_hist = (DATA_PATH + "alice_epr_generator_history_exp_" 
                          + str(exp_typ) + ".csv")
        error_hist = (DATA_PATH + "applied_error_history_exp_" + str(exp_typ) 
                      + ".csv")
        meas_portion_hist = (DATA_PATH + "meas_portion_q_exp_" + str(exp_typ) 
                             + ".csv")

        Alice_EPR_gen.history.to_csv(alice_gen_hist)
        rot_error.history.to_csv(error_hist)
        len_est.history.to_csv(meas_portion_hist)

    if PRINT_HIST:
        print(EntMngr_a.EPR_frame_history)
        print(EntMngr_b.EPR_frame_history)
        print(EntMngr_a.SDC_frame_history)
        print(EntMngr_b.SDC_frame_history)
        print(EntMngr_a.In_halves_history)
        print(EntMngr_b.In_halves_history)
    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()

if __name__ == '__main__':
    main()
