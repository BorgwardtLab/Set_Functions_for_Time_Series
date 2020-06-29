"""Computation of dt tensor used in GRUSimple and PhasedLSTM."""
import tensorflow as tf


def get_delta_t(times, measurements, measurement_indicators):
    """Add a delta t tensor which contains time since previous measurement.

    Args:
        times: The times of the measurements (tp,)
        measurements: The measured values (tp, measure_dim)
        measurement_indicators: Indicators if the variables was measured or not
            (tp, measure_dim)

    Returns:
        delta t tensor of shape (tp, measure_dim)

    """
    scattered_times = times * tf.cast(measurement_indicators, tf.float32)
    dt_array = tf.TensorArray(tf.float32, tf.shape(measurement_indicators)[0])
    # First observation has dt = 0
    first_dt = tf.zeros(tf.shape(measurement_indicators)[1:])
    dt_array = dt_array.write(0, first_dt)

    def compute_dt_timestep(i, last_dt, dt_array):
        last_dt = tf.where(
            measurement_indicators[i-1],
            tf.fill(tf.shape(last_dt), tf.squeeze(times[i] - times[i-1])),
            times[i] - times[i-1] + last_dt
        )
        dt_array = dt_array.write(i, last_dt)
        return i+1, last_dt, dt_array

    n_observations = tf.shape(scattered_times)[0]
    _, last_dt, dt_array = tf.while_loop(
        lambda i, a, b: i < n_observations,
        compute_dt_timestep,
        loop_vars=[tf.constant(1), first_dt, dt_array]
    )
    dt_tensor = dt_array.stack()
    dt_tensor.set_shape(measurements.get_shape())
    return dt_tensor
