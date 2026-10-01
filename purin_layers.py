import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='PurinMRI')
class PurinConv2D(tf.keras.layers.Layer):
    """Convolution with learned output modulation for each feature channel."""

    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.convolution = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False
        )

    def build(self, input_shape):
        self.convolution.build(input_shape)
        self.channel_weights = self.add_weight(
            name='channel_weights', shape=(self.filters,), initializer='ones', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        features = self.convolution(inputs)
        features = features * self.channel_weights[None, None, None, :]
        return tf.nn.relu(features)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package='PurinMRI')
class STPModulation(tf.keras.layers.Layer):
    """Activation-driven short-term state with bounded values and recovery to one."""

    def __init__(self, recovery=0.9, lower_bound=0.8, upper_bound=1.2, **kwargs):
        super().__init__(**kwargs)
        self.recovery = recovery
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.stp_state = self.add_weight(
            name='stp_state', shape=(channels,), initializer='ones', trainable=False
        )
        self.sensitivity = self.add_weight(
            name='sensitivity', shape=(channels,),
            initializer=tf.keras.initializers.Constant(0.05), trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        channel_activity = tf.reduce_mean(inputs, axis=(0, 1, 2))
        activity_scale = tf.reduce_mean(tf.abs(channel_activity)) + tf.keras.backend.epsilon()
        centered_activity = (channel_activity - tf.reduce_mean(channel_activity)) / activity_scale
        target_state = tf.clip_by_value(
            1.0 + self.sensitivity * centered_activity, self.lower_bound, self.upper_bound
        )
        next_state = tf.clip_by_value(
            1.0 + self.recovery * (self.stp_state - 1.0)
            + (1.0 - self.recovery) * (target_state - 1.0),
            self.lower_bound, self.upper_bound,
        )

        def update_state():
            self.stp_state.assign(next_state)
            return tf.identity(self.stp_state)

        if training is True:
            modulation = update_state()
        elif training is None or training is False:
            modulation = tf.identity(self.stp_state)
        else:
            modulation = tf.cond(
                tf.cast(training, tf.bool), update_state, lambda: tf.identity(self.stp_state)
            )
        return inputs * modulation[None, None, None, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'recovery': self.recovery,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
        })
        return config
