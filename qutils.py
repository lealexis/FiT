import numpy as np

from qunetsim.objects import Qubit
from qunetsim.components import Host


def get_epr_fidelity(epr_halve: Qubit):
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
    

def superposed_qubit(host_ins: Host, mu=0, sigma=0.05):
    """Generate a qubit in a superposition state being mostly at the ground
    state by applying rotational X and Y gate. Rotational angles are drawn as
    a normal distribution with mean value 0 and sigma 0.05. The output is
    multiplied with the angle in radians np.pi/2."""

    mixed_qubit = Qubit(host_ins)  # qubit in ket(0)
    rx_rad, ry_rad = np.random.normal(loc=mu, scale=sigma, size=2)
    rx_rad = rx_rad * (np.pi / 2)
    ry_rad = ry_rad * (np.pi / 2)
    #ry_rad = np.random.normal(loc=mu, scale=sigma) * (np.pi / 2)

    mixed_qubit.rx(phi=rx_rad)
    mixed_qubit.ry(phi=ry_rad)

    return mixed_qubit


def dens_encode(q: Qubit, bits: str):
    """super dense coding"""
    if bits == "00":
        q.I()
    elif bits == "10":
        q.Z()
    elif bits == "01":
        q.X()
    elif bits == "11":
        q.X()
        q.Z()
    else:
        raise Exception('Bad input')
    return


def dense_decode(stored_epr_half: Qubit, received_qubit: Qubit):
    """super dense decoding"""
    received_qubit.cnot(stored_epr_half)
    received_qubit.H()
    meas = [None, None]
    meas[0] = received_qubit.measure()
    meas[1] = stored_epr_half.measure()
    return str(meas[0]) + str(meas[1])


# The functions bellow were programmed in the development process
def gen_epr_fidelity_rand_rot(gen_host: Host, F=0.5, mu=0, sigma=0.2, 
                              full_return=False):
    """Generator of EPR-Pair with a given fidelity F. Qubits in superposition
    are the input to the entanglement circuit. Those input qubits are mostly
    in the zero ket state, i.e. a zero ket qubit will be x-y rotated with
    random choosen angles. The last rotation angle is calculated to satisfy the
    requested Fidelity F of the pair.
    If full_return is set True the rotation angles for the first and second
    qubit around their x and y axes is returned, so that exactly the same
    EPR-Pair can be replicated."""
    
    rx1 = 0
    ry1 = 0
    rx2 = 0
    ry2 = 0
    while True:
        q1, rx1, ry1 = superposed_qubit(host_ins=gen_host, mu=mu, sigma=sigma)
        A = ((np.cos(rx1 / 2) * np.cos(ry1 / 2))**2 + (np.sin(rx1 / 2) * 
                                                       np.sin(ry1 / 2))**2)
    
        rx2 = np.random.normal(loc=mu, scale=sigma) * (np.pi / 2)
        C1 = np.cos(rx2 / 2)**2
        C2 = np.sin(rx2 / 2)**2

        A1 = F / (A * (C2 - C1))
        A2 = F / (A * (C1 - C2))
        B1 = C1 / (C2 - C1)
        B2 = C2 / (C1 - C2)

        #arcsin and arccos domain -1 to +1 
        if (A1 < B1) and (A2 < B2):
            q1.release()
            continue
        elif (A1 > B1):
            arg_arcsin = np.sqrt(A1 - B1)
            if (arg_arcsin > 1) or (arg_arcsin < -1):
                q1.release()
                continue
            else:
                ry2 = 2 * np.arcsin(arg_arcsin)
                break
        else:
            arg_arccos = np.sqrt(A2 - B2)
            if (arg_arccos > 1) or (arg_arccos < -1):
                q1.release()
                continue
            else:
                ry2 = 2 * np.arccos(arg_arccos)
                break

    q2 = Qubit(gen_host, q_id=q1.id)
    q2.rx(phi=rx2)
    q2.ry(phi=ry2)

    # Entangling the quibits
    q1.H()
    q1.cnot(q2)

    if full_return:
        return q1, rx1, ry1, q2, rx2, ry2
    else:
        return q1, q2


def gen_epr_non_random(gen_host: Host, rx1=0, ry1=0, rx2=0, ry2=0):
    """Generate an EPR-Pair by applying x and y rotational gates with angles
    rx1, ry1, rx2, and ry2 to the first and second qubit before passing them
    trough the entanglement circuit.
    If default rotation angles are used a perfect EPR-Pair is generated."""

    q1 = Qubit(gen_host)
    q2 = Qubit(gen_host, q_id=q1.id)

    # apply rotational gates
    q1.rx(phi=rx1)
    q1.ry(phi=ry1)
    q2.rx(phi=rx2)
    q2.ry(phi=ry2)

    # entanglement circuit
    q1.H()
    q1.cnot(q2)

    return q1, q2


def alpha_epr_gen_fid(gen_host: Host, F=0.5, spread=0.05, exact=False, 
                      full_return=False, verbose=False):
    """ EPR-Pairs are generated by applying X and Y rotation gates to the zero
    ket qubits before entering the entanglement circuit. The rotation angles
    are taked from a gaussian distribution with mean value matching the angle
    alpha which would generate a EPR-Pair with fidelity F. """

    if (F < 0.25) or (F > 1):
        raise InputError("Invalid F passed in. F must live in the interval"
                         " [0.25, 1]")
    else:
        alpha = 0.5 * np.arccos(4 * np.sqrt(F) - 3)

        if exact:
            qa, qb = gen_epr_non_random(gen_host=gen_host, rx1=alpha, 
                                        ry1=alpha, rx2=alpha, ry2=alpha)
            if verbose:
                print("Pair with F = " + str(get_epr_fidelity(epr_halve=qa))
                      +  " was generated.")
            if full_return:
                return qa, qb, alpha
            else:
                return qa, qb
            
        else:
            sigma = spread * alpha
            rxa, rya, rxb, ryb = np.random.normal(loc=alpha, scale=sigma, 
                                                  size=4)
 
            qa, qb = gen_epr_non_random(gen_host=gen_host, rx1=rxa, ry1=rya, 
                                        rx2=rxb, ry2=ryb)
            if verbose:
                print("Pair with F = " + str(get_epr_fidelity(epr_halve=qa))
                      +  " was generated.")
            if full_return:
                return qa, rxa, rya, qb, rxb, ryb
            else:
                return qa, qb


def gaussian_xy_rotation_channel(flying_qubit: Qubit, mu=(np.pi)/5, spread=0.3, 
                                 single_angle=True):
    sigma = spread * mu
 
    if single_angle:
        gamma = np.random.normal(loc=mu, scale=sigma)
        flying_qubit.rx(phi=gamma)
        flying_qubit.ry(phi=gamma)
    else:
        gamma_x, gamma_y = np.random.normal(loc=mu, scale=sigma, size=2)

        flying_qubit.rx(phi=gamma_x)
        flying_qubit.ry(phi=gamma_y)
    return


def fidelity_dropping_channel(flying_qubit: Qubit, f_drop_percent=0.1, 
                              rot_type="xy", mu=0, sigma=0.25, 
                              full_return=False):
    """Channel model implemented by a sequential application of two rotational
    gates xy, xz or yz. The rotation angles are choosen in such a way that the
    Fidelity of the qubit, with respect to a corresponding pure state, is
    dropped percentualy by f_drop_percent.
    Solely single qubit systems or double qubit systems(EPR-Pair) are allowed
    to use this channel model. """
    
    if np.shape(flying_qubit.statevector()[1])[0] == 2:
        # Single qubit system - applyable to the header!
        if rot_type == "xy":
            pass
        elif rot_type == "xz":
            pass
        elif rot_type == "yz":
            pass
        else:
            raise InputError("Invalid rot_type passed in.")

    elif np.shape(flying_qubit.statevector()[1])[0] == 4:
        # Double qubit system - EPR Pair
        ket_state = flying_qubit.statevector()[1]
        a1 = ket_state[0].real
        b1 = ket_state[0].imag
        
        a2 = ket_state[1].real
        b2 = ket_state[1].imag
        
        a3 = ket_state[2].real
        b3 = ket_state[2].imag

        a4 = ket_state[3].real
        b4 = ket_state[3].imag
        # first only for xy

        F = get_epr_fidelity(epr_halve=flying_qubit)
        Delta_F = -1 * F * f_drop_percent

        if rot_type == "xy":
            K1 = 0.5*((b1 - b4)**2 + (a1 - a4)**2)
            K2 = 0.5*((a2 - a3)**2 + (b2 - b3)**2)
            K3 = 0.5*((a2 + a3)**2 + (b2 + b3)**2)
            K4 = 2*(a4*b1 - a1*b4)
            K5 = 2*(a2*b3 - a3*b2)
            K6 = a1*a2 - a3*a4 + b1*b2 - b3*b4
            K7 = a2*a4 - a1*a3 - b1*b3 + b2*b4
            K8 = a2*b1 + a3*b4 - a1*b2 - a4*b3
            K9 = a1*b3 + a4*b2 - a3*b1 - a2*b4
            gamma = 0
            phi = 0
            while True:
                gamma = np.random.normal(loc=mu, scale=sigma) * (np.pi / 2)
            
                r = np.cos(gamma / 2)
                s = np.sin(gamma / 2)

                kappa_1 = F + Delta_F - r * s * K9
                kappa_2 = 0.5 * ((F + K2) * r**2 + (K3 + K1) * s**2)
                m = 0.5 * ((F - K2) * r**2 - 2 * r * s * K8 + (K3 - K1) * s**2)
                n = 0.5 * (r * s * (K4 + K5) + K6 + K7 * (r**2 - s**2))

                arc_arg = (kappa_1 - kappa_2)/np.sqrt(m**2 + n**2)
                
                if (arc_arg > 1) or (arc_arg < -1):
                    continue
                else:
                    phi=0
                    if n > m :
                        phi = (np.arcsin(arc_arg) - np.arctan(m / n))
                        break
                    else:
                        phi = (np.arccos(arc_arg) + np.arctan(n / m))
                        break

            flying_qubit.rx(phi=gamma)
            flying_qubit.ry(phi=phi)
            
            if full_return:
                return rot_type, gamma, phi
            else:
                return

        elif rot_type == "xz":
            pass
        elif rot_type == "yz":
            pass
        else:
            raise InputError("Invalid rot_type passed in.")
    else:
        raise InputError("Invalid qubit passed: Only single or double qubit "
                         "systems are allowed.")


def rot_channel(flying_qubit: Qubit, rot_typ="xy", phi_1=0, phi_2=0):
    if rot_typ == "xy":
        flying_qubit.rx(phi=phi_1)
        flying_qubit.ry(phi=phi_2)
        return
    elif rot_typ == "xz":
        flying_qubit.rx(phi=phi_1)
        flying_qubit.rz(phi=phi_2)
        return
    elif rot_typ == "yz":
        flying_qubit.ry(phi=phi_1)
        flying_qubit.rz(phi=phi_2)
        return
    else:
        raise InputError("Invalid rot_type passed in.")