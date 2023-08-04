import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)))
        ])
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfp.layers.MultivariateNormalTriL(n),
        ])
    return posterior_model


def bnn_model(hidden_units, input_shape, train_size, activation):
    # Create the input layer
    input_layer = layers.Input(shape=input_shape)
    features = input_layer

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for i, units in enumerate(hidden_units):
        if i!= len(hidden_units)-1:
            features = layers.Dense(units, activation=activation)(features)
        else:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation=activation,
            )(features)

    # create output - mean and std
    distribution_params = layers.Dense(units=tfp.layers.IndependentNormal.params_size(2))(features)
    outputs = tfp.layers.IndependentNormal(2)(distribution_params)

    model = keras.Model(inputs=input_layer, outputs=outputs)

    return model


def fit_bnn_model(model, loss, num_epochs, learning_rate, X_train, y_train):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=loss,
        metrics=[keras.metrics.MeanSquaredError()],
    )

    model.fit(x=X_train, y=y_train, epochs=num_epochs)


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)
