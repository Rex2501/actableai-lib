from functools import lru_cache
from time import time
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import shap
import tensorflow.keras.backend as K
from sklearn.cluster import KMeans as KMeansSK
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Input, Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from actableai.clustering import metrics
from actableai.clustering.models.base import BaseClusteringModel
from actableai.clustering.models.base import Model as ClusteringModelEnum
from actableai.clustering.models.kmeans import KMeans
from actableai.parameters.numeric import (
    IntegerListParameter,
    FloatParameter,
    IntegerParameter,
)
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    Example:
        model.add(ClusteringLayer(n_clusters=10))

    Input shape:
        2D tensor with shape: `(n_samples, n_features)`.
    Output shape:
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        """Constructor

        Args:
            n_clusters: number of clusters.
            weights: (n_clusters, n_features) represents the initial cluster centers.
            alpha: parameter in Student's t-distribution. Default to 1.0.
        """
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (
            1.0
            + (
                K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = q / K.sum(q, axis=1, keepdims=True)
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class _DEC:
    @staticmethod
    def _autoencoder(
        dims: List[int],
        act: str = "relu",
        init: str = "glorot_uniform",
    ) -> Tuple[Model, Model, Model]:
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim,
                dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the
                    auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
            init: Initialization for weights.
                See https://keras.io/initializers/
        return:
            Tuple [Model, Model, Model]:
                - Auto-Encoder model
                - Encoder model. To get latent representation of input data.
                - Decoder model. To reconstruct input data from latent representation.
        """
        n_stacks = len(dims) - 1
        # input
        x = Input(shape=(dims[0],), name="input")
        h = x

        # internal layers in encoder
        for i in range(n_stacks - 1):
            h = Dense(
                dims[i + 1],
                activation=act,
                kernel_initializer=init,
                name="encoder_%d" % i,
            )(h)

        # hidden layer
        h = Dense(
            dims[-1], kernel_initializer=init, name="encoder_%d" % (n_stacks - 1)
        )(
            h
        )  # hidden layer, features are extracted from here

        y = h
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            y = Dense(
                dims[i], activation=act, kernel_initializer=init, name="decoder_%d" % i
            )(y)

        # output
        y = Dense(dims[0], kernel_initializer=init, name="decoder_0")(y)

        # build decoder model
        decoder_in = Input(shape=(dims[-1],), name="projected_input")
        decoder_y = decoder_in
        for i in range(n_stacks - 1, 0, -1):
            decoder_y = Dense(
                dims[i], activation=act, kernel_initializer=init, name="%d" % i
            )(decoder_y)
        decoder_y = Dense(dims[0], kernel_initializer=init, name="0")(decoder_y)

        return (
            Model(inputs=x, outputs=y, name="AE"),
            Model(inputs=x, outputs=h, name="encoder"),
            Model(inputs=decoder_in, outputs=decoder_y, name="decoder"),
        )

    def __init__(
        self,
        dims: List,
        n_clusters: Union[str, int] = "auto",
        alpha: float = 1.0,
        init: str = "glorot_uniform",
        auto_num_clusters_min: int = 2,
        auto_num_clusters_max: int = 20,
        alpha_k: float = 0.01,
    ):
        """Deep Embedded Clustering implementation.

        Args:
            dims: list of number of units in each layer of encoder.
            n_clusters: Number of clusters. If n_clusters is set to be 'auto',
                then it will be determined automatically by the algorithm.
            alpha: Parameter in Student's t-distribution. Default to 1.0.
            init: Initialization for weights.
                See https://keras.io/initializers/
            auto_num_clusters_min: The minimum number of clusters to be determined
                automatically. Default to 2.
            auto_num_clusters_max: The maximum number of clusters to be determined
                automatically. Default to 20.
            alpha_k: The factor to control the penalty term of the number of clusters.
                Default to 0.01.
        """
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder, self.decoder = self._autoencoder(
            self.dims, init=init
        )

        self.auto_num_clusters_min = auto_num_clusters_min
        self.auto_num_clusters_max = auto_num_clusters_max
        self.alpha_k = alpha_k

    def pretrain(
        self,
        x: np.ndarray,
        optimizer: str = "adam",
        epochs: int = 200,
        batch_size: int = 256,
    ) -> None:
        """Pretrain the autoencoder.

        Args:
            x: Input data.
            optimizer: Optimizer for training the autoencoder. Default to 'adam'.
            epochs: Number of epochs. Default to 200.
            batch_size: Batch size. Default to 256.
        """
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx]

        print("Training Auto-encoder...")
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.decoder.compile(optimizer=optimizer, loss="mse")
        es = callbacks.EarlyStopping(
            monitor="loss", patience=5, min_delta=1e-3, mode="min"
        )
        cb = [es]
        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)

        print("Initializing cluster centers with k-means...")
        if self.n_clusters == "auto":
            self.n_clusters = KMeans.KMeans_pick_k(
                x,
                self.alpha_k,
                range(self.auto_num_clusters_min, self.auto_num_clusters_max + 1),
            )
            print("Found number of clusters: ", self.n_clusters)

        kmeans = KMeansSK(n_clusters=self.n_clusters, n_init=20)
        self.y_pred_last = kmeans.fit_predict(self.encoder.predict(x))

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(
            self.encoder.output
        )
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        print("Pretraining time: %ds" % round(time() - t0))
        self.pretrained = True

    def load_weights(self, weights: str) -> None:
        """Load weights of DEC model.

        Args:
            weights: Path to the weights file.
        """
        self.model.load_weights(weights)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project data into the latent space.

        Args:
            x: Input data.

        Returns:
            _type_: _description_
        """
        return self.encoder.predict(x)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct data from the latent space.

        Args:
            x: Input data.

        Returns:
            np.ndarray: Reconstructed data.
        """
        return self.decoder.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict cluster labels probabilities using the output of clustering layer.

        Args:
            x: Input data.

        Returns:
            np.ndarray: Predicted cluster label probabilities.
        """
        return self.model.predict(x, verbose=0)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        """Predict cluster labels using the output of clustering layer.

        Args:
            x: Input data.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        """Target distribution for the K-means objective."""
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer: str = "adam", loss: str = "kld") -> None:
        """Compile the DEC model.

        Args:
            optimizer: Optimizer for training the model. Default to 'adam'.
            loss: Loss function. Default to 'kld'.
        """
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(
        self,
        x,
        y=None,
        maxiter=2e4,
        batch_size=256,
        tol=1e-3,
        update_interval=140,
    ):
        """Train the model.

        Args:
            x: Input data.
            y: Target data. Default to None.
            maxiter: Maximum number of iterationsf for training. Default to 2e4.
            batch_size: Batch size. Default to 256.
            tol: Tolerance for the stopping criterion. Default to 1e-3.
            update_interval: The interval to check the stopping criterion and update the
                cluster centers. Default to 140.
        """
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx]
        if y is not None:
            y = y[idx]

        # save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs

        # Step 1: initialize cluster centers using k-means
        # Step 2: deep clustering

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        print("Training DEC model...")
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(
                    q
                )  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print(
                        "Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f"
                        % (ite, acc, nmi, ari),
                        " ; loss=",
                        loss,
                    )

                # check stop criterion
                delta_label = (
                    np.sum(y_pred != self.y_pred_last).astype(np.float32)
                    / y_pred.shape[0]
                )
                self.y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print("delta_label ", delta_label, "< tol ", tol)
                    print("Reached tolerance threshold. Stopping training.")
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[
                index * batch_size : min((index + 1) * batch_size, x.shape[0])
            ]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            ite += 1

        self.encoded_cluster_centers = self.model.get_layer("clustering").get_weights()[
            0
        ]

    def fine_tune_decoder(self, x, maxiter=2e4):
        """Fine-tune decoder after encoder and centroids are updated. This is useful for reconstructing
        encoded values which can be used to explain for clusters using reconstructed centroids or
        explain for axes."""
        # copy model weights to decoder model
        for i in range(len(self.dims) - 2, -1, -1):
            self.decoder.get_layer("%d" % i).set_weights(
                self.autoencoder.get_layer("decoder_%d" % i).get_weights()
            )

        # fine-tune decoder to catch up with fine-tuned encoder
        self.decoder.fit(
            x=self.encoder.predict(x),
            y=x,
            epochs=int(maxiter),
            callbacks=[callbacks.EarlyStopping(monitor="loss", patience=3, mode="min")],
        )

        return self.decoder


class DEC(BaseClusteringModel):
    """
    TODO write documentation
    """

    has_explanations = True
    handle_categorical = True

    @staticmethod
    @lru_cache(maxsize=None)
    def get_parameters() -> Parameters:
        """Returns the parameters of the model.

        Returns:
            The parameters.
        """
        parameters = [
            IntegerListParameter(
                name="dims",
                display_name="Dimensions",
                description="List of number of units in each internal layer of encoder.",
                default=[500, 500, 2000, 10],
                min_len=1,
                min=1,
                # TODO check constraints, maybe add max_len and max
            ),
            OptionsParameter[str](
                name="optimizer",
                display_name="Optimizer",
                description="Optimizer to use for backpropagation.",
                default="sgd",
                is_multi=False,
                options={
                    "sgd": {
                        "display_name": "Stochastic Gradient Descent",
                        "value": "sgd",
                    },
                    "adam": {"display_name": "Adam", "value": "adam"},
                },
            ),
            FloatParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Learning rate used during training.",
                default=0.01,
                min=0.0001,
                # TODO check constraints
            ),
            FloatParameter(
                name="momentum",
                display_name="Momentum",
                description="Momentum used during training.",
                default=0.9,
                min=0,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="loss",
                display_name="Loss",
                description="Training loss.",
                default="kld",
                is_multi=False,
                options={
                    "kld": {
                        "display_name": "Kullback-Leibler",
                        "value": "kld",
                    },
                    "mae": {"display_name": "Mean Absolute Error", "value": "mae"},
                    "mape": {
                        "display_name": "Mean Absolute Percentage Error",
                        "value": "mape",
                    },
                    "mse": {"display_name": "Mean Squared Error", "value": "mse"},
                    # TODO add more losses, see https://www.tensorflow.org/api_docs/python/tf/keras/losses
                },
            ),
            IntegerParameter(
                name="max_iteration",
                display_name="Max Iteration",
                description="Maximum training iteration",
                default=2e4,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="batch_size",
                display_name="Batch Size",
                description="Batch size to use when training.",
                default=256,
                min=1,
                # TODO check constraints
            ),
            FloatParameter(
                name="tol",
                display_name="Tolerance",
                description="Tolerance for the stopping criterion.",
                default=1e-3,
                min=0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="update_interval",
                display_name="Update Interval",
                description="The interval to check the stopping criterion and update the cluster centers.",
                default=140,
                min=1,
                # TODO check constraints
            ),
            IntegerParameter(
                name="pretrain_epochs",
                display_name="Pretrain Epochs",
                description="Number of epochs to use when pre-training.",
                default=200,
                min=1,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="pretrain_optimizer",
                display_name="Pre-train Optimizer",
                description="Optimizer to use for backpropagation when pre-training.",
                default="adam",
                is_multi=False,
                options={
                    "sgd": {
                        "display_name": "Stochastic Gradient Descent",
                        "value": "sgd",
                    },
                    "adam": {"display_name": "Adam", "value": "adam"},
                },
            ),
            FloatParameter(
                name="pretrain_learning_rate",
                display_name="Pre-train Learning Rate",
                description="Learning rate used during pre-training.",
                default=0.001,
                min=0.0001,
                # TODO check constraints
            ),
            FloatParameter(
                name="pretrain_momentum",
                display_name="Pre-train Momentum",
                description="Momentum used during pre-training.",
                default=0,
                min=0,
                # TODO check constraints
            ),
            IntegerParameter(
                name="pretrain_batch_size",
                display_name="Pre-train Batch Size",
                description="Batch size to use when pre-training.",
                default=256,
                min=1,
                # TODO check constraints
            ),
            OptionsParameter[str](
                name="init",
                display_name="Weight Initialization",
                description="Initialization for weights.",
                default="glorot_uniform",
                is_multi=False,
                options={
                    "glorot_uniform": {
                        "display_name": "Glorot Uniform",
                        "value": "glorot_uniform",
                    },
                    "random_normal": {
                        "display_name": "Random Normal",
                        "value": "random_normal",
                    },
                    "radom_uniform": {
                        "display_name": "Random Uniform",
                        "value": "radom_uniform",
                    },
                    # TODO add more initializers
                },
            ),
            FloatParameter(
                name="alpha",
                display_name="Alpha",
                description="Parameter in Student's t-distribution.",
                default=1.0,
                # TODO check constraints
            ),
        ]

        return Parameters(
            name=ClusteringModelEnum.dec,
            display_name="Deep Embedding for Clustering (DEC) Model",
            parameters=parameters,
        )

    def __init__(
        self,
        input_size: int,
        num_clusters: int,
        parameters: Dict[str, Any] = None,
        process_parameters: bool = True,
        verbosity: bool = 1,
    ):
        super().__init__(
            input_size=input_size,
            num_clusters=num_clusters,
            parameters=parameters,
            process_parameters=process_parameters,
            verbosity=verbosity,
        )

        self.pretrain_epochs = self.parameters["pretrain_epochs"]
        self.pretrain_batch_size = self.parameters["pretrain_batch_size"]
        self.loss = self.parameters["loss"]
        self.max_iteration = self.parameters["max_iteration"]
        self.batch_size = self.parameters["batch_size"]
        self.tol = self.parameters["tol"]
        self.update_interval = self.parameters["update_interval"]

        pretrain_optimizer = self.parameters["pretrain_optimizer"]
        pretrain_learning_rate = self.parameters["pretrain_learning_rate"]
        pretrain_momentum = self.parameters["pretrain_momentum"]
        optimizer = self.parameters["optimizer"]
        learning_rate = self.parameters["learning_rate"]
        momentum = self.parameters["momentum"]

        self.model = _DEC(
            dims=[self.input_size, *self.parameters["dims"]],
            n_clusters=self.num_clusters,
            alpha=self.parameters["alpha"],
            init=self.parameters["init"],
        )

        if pretrain_optimizer == "adam":
            self.pretrain_optimizer = Adam(learning_rate=pretrain_learning_rate)
        elif pretrain_optimizer == "sgd":
            self.pretrain_optimizer = SGD(
                learning_rate=pretrain_learning_rate,
                momentum=pretrain_momentum,
            )
        else:
            self.pretrain_optimizer = pretrain_optimizer

        if optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = SGD(
                learning_rate=learning_rate,
                momentum=momentum,
            )
        else:
            self.optimizer = optimizer

    def _fit(self, data: np.ndarray, target: np.ndarray = None) -> bool:
        """
        TODO write documentation
        """
        self.model.pretrain(
            x=data,
            optimizer=self.pretrain_optimizer,
            epochs=self.pretrain_epochs,
            batch_size=self.pretrain_batch_size,
        )
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        self.model.fit(
            x=data,
            maxiter=self.max_iteration,
            batch_size=self.batch_size,
            tol=self.tol,
            update_interval=self.update_interval,
        )

        return True

    def _predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)

    def _project(self, data: np.ndarray) -> np.ndarray:
        return self.model.project(data)

    def _explain_samples(self, data: np.ndarray) -> np.ndarray:
        background_samples = 100
        if len(data) < 100:
            background_samples = int(len(data) * 0.1)

        background = data[
            np.random.choice(
                data.shape[0],
                background_samples,
                replace=False,
            )
        ]

        explainer = shap.DeepExplainer(self.model.model, np.array(background))

        shap_values = explainer.shap_values(np.array(data), check_additivity=False)

        return np.array(shap_values)
