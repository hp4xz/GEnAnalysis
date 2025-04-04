
def spherical_to_cartesian(theta_deg, phi_deg):
    import numpy as np
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def compute_Px_Pz(trPx, trPy, trPz, ebeam, theta_deg, phi_deg):
    import numpy as np
    """
    Compute Px and Pz (spin projections) for all events given:
    - trP{xyz}: arrays of scattered electron momenta
    - ebeam: array of beam energies
    - theta_deg, phi_deg: spin orientation angles (degrees)

    Returns:
    - Px: np.ndarray
    - Pz: np.ndarray
    """

    # Convert spin direction to unit vector
    S = spherical_to_cartesian(theta_deg, phi_deg)
    S /= np.linalg.norm(S)

    # Beam vector (along z)
    k_in_z = ebeam
    k_in_x = np.zeros_like(trPx)
    k_in_y = np.zeros_like(trPy)

    # q = k_in - k_out
    q_x = k_in_x - trPx
    q_y = k_in_y - trPy
    q_z = k_in_z - trPz

    q_mag = np.sqrt(q_x**2 + q_y**2 + q_z**2)
    valid_q = q_mag > 0
    q_x[~valid_q] = 1
    q_y[~valid_q] = 0
    q_z[~valid_q] = 0
    q_hat_x = q_x / q_mag
    q_hat_y = q_y / q_mag
    q_hat_z = q_z / q_mag

    # n = k_in × k_out
    n_x = k_in_y * trPz - k_in_z * trPy
    n_y = k_in_z * trPx - k_in_x * trPz
    n_z = k_in_x * trPy - k_in_y * trPx

    n_mag = np.sqrt(n_x**2 + n_y**2 + n_z**2)
    valid_n = n_mag > 0
    n_x[~valid_n] = 1
    n_y[~valid_n] = 0
    n_z[~valid_n] = 0
    n_hat_x = n_x / n_mag
    n_hat_y = n_y / n_mag
    n_hat_z = n_z / n_mag

    # q̂ × S
    qxS_x = q_hat_y * S[2] - q_hat_z * S[1]
    qxS_y = q_hat_z * S[0] - q_hat_x * S[2]
    qxS_z = q_hat_x * S[1] - q_hat_y * S[0]

    # Px = n̂ ⋅ (q̂ × S)
    Px = n_hat_x * qxS_x + n_hat_y * qxS_y + n_hat_z * qxS_z

    # Pz = q̂ ⋅ S
    Pz = q_hat_x * S[0] + q_hat_y * S[1] + q_hat_z * S[2]

    # Mask invalid
    valid = valid_q & valid_n
    Px[~valid] = np.nan
    Pz[~valid] = np.nan

    return Px, Pz
