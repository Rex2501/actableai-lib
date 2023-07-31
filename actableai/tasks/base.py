import logging
import ray
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

from actableai.parameters.models import ModelSpace
from actableai.parameters.parameters import Parameters
from actableai.tasks import TaskType
from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.resources.profile import ResourceProfilerType


class AAITask(ABC):
    """
    Base abstract class to represent a Actable AI Task
    """

    def __init__(
        self,
        use_ray: bool = False,
        ray_params: Optional[dict] = None,
        optimize_memory_allocation: bool = False,
        collect_memory_usage: bool = False,
        optimize_memory_allocation_nrmse_threshold: float = 0.2,
        max_memory_offset: float = 0.1,
        optimize_gpu_memory_allocation: bool = False,
        collect_gpu_memory_usage: bool = False,
        optimize_gpu_memory_allocation_nrmse_threshold: float = 0.2,
        max_gpu_memory_offset: float = 0.1,
        resources_predictors_actor: Optional[ray.actor.ActorHandle] = None,
        cpu_percent_interval: float = 1.0,
        return_model: bool = True,
        upload_model: bool = False,
        s3_models_bucket: Optional[str] = None,
        s3_models_prefix: Optional[str] = None,
        seed=None,
    ):
        """
        AAITask Constructor

        Parameters
        ----------
        use_ray:
            If true the task will be run using ray (in a remote worker), default: False
        ray_params:
            Parameters to be passed to the ray remote function
        optimize_memory_allocation:
            If True will try to predict the memory allocation of the function and use it with ray, default: False
        collect_memory_usage:
            If True will monitor the memory usage of the task and collect that data for future prediction,
            default: False
        optimize_memory_allocation_nrmse_threshold:
            If the Normalized RMSE of the memory allocation model is greater than this threshold then the prediction
            will not be used, default: 0.2
        max_memory_offset:
            When prediction the memory allocation this offset will be added to the max memory usage prediction,
            (offset in percentage of the prediction), default: 0.1
        optimize_gpu_memory_allocation:
            If True will try to predict the GPU memory allocation of the function and use it with ray, default: False
        collect_gpu_memory_usage:
            If True will monitor the GPU memory usage of the task and collect that data for future prediction,
            default: False
        optimize_gpu_memory_allocation_nrmse_threshold:
            If the Normalized RMSE of the GPU memory allocation model is greater than this threshold then the prediction
            will not be used, default: 0.2
        max_gpu_memory_offset:
            When prediction the memory allocation this offset will be added to the max GPU memory usage prediction,
            (offset in percentage of the prediction), default: 0.1
        resources_predictors_actor:
            The actor used to predict the resources usage
        cpu_percent_interval:
            Compare cpu usage before and after interval in seconds
        return_model:
            If False the model field will be deleted from the data if it exists
        upload_model:
            If True it will upload the pickled model with the name `model.p` at the AWS location given
        s3_models_bucket:
            The AWS S3 bucket name where the models will be stored
        s3_models_prefix:
            The AWS S3 prefix to use when saving the models
        seed:
            The seed to use to control randomness and reproducibility of results
        """
        self.use_ray = use_ray
        self.ray_params = ray_params if ray_params is not None else {}

        self.optimize_memory_allocation = optimize_memory_allocation
        self.collect_memory_usage = collect_memory_usage
        self.optimize_memory_allocation_nrmse_threshold = (
            optimize_memory_allocation_nrmse_threshold
        )
        self.max_memory_offset = max_memory_offset

        self.optimize_gpu_memory_allocation = optimize_gpu_memory_allocation
        self.collect_gpu_memory_usage = collect_gpu_memory_usage
        self.optimize_gpu_memory_allocation_nrmse_threshold = (
            optimize_gpu_memory_allocation_nrmse_threshold
        )
        self.max_gpu_memory_offset = max_gpu_memory_offset

        self.resources_predictors_actor = resources_predictors_actor

        self.cpu_percent_interval = cpu_percent_interval

        self.return_model = return_model
        self.upload_model = upload_model
        self.s3_models_bucket = s3_models_bucket
        self.s3_models_prefix = s3_models_prefix

        self.seed = seed

        if self.s3_models_prefix is None:
            self.s3_models_prefix = ""

        if self.optimize_memory_allocation and not self.use_ray:
            self.optimize_memory_allocation = False
            logging.warning(
                "`optimize_memory_allocation` is set to False: `use_ray` is False"
            )
        if self.collect_memory_usage and not self.use_ray:
            self.collect_memory_usage = False
            logging.warning(
                "`collect_memory_usage` is set to False: `use_ray` is False"
            )
        if self.optimize_gpu_memory_allocation and not self.use_ray:
            self.optimize_gpu_memory_allocation = False
            logging.warning(
                "`optimize_gpu_memory_allocation` is set to False: `use_ray` is False"
            )
        if self.collect_gpu_memory_usage and not self.use_ray:
            self.collect_gpu_memory_usage = False
            logging.warning(
                "`collect_gpu_memory_usage` is set to False: `use_ray` is False"
            )

        if self.optimize_memory_allocation and self.resources_predictors_actor is None:
            self.optimize_memory_allocation = False
            logging.warning(
                "`optimize_memory_allocation` is set to False: `resources_predictors_actor` is None"
            )
        if self.collect_memory_usage and self.resources_predictors_actor is None:
            self.collect_memory_usage = False
            logging.warning(
                "`collect_memory_usage` is set to False: `resources_predictors_actor` is None"
            )
        if (
            self.optimize_gpu_memory_allocation
            and self.resources_predictors_actor is None
        ):
            self.optimize_gpu_memory_allocation = False
            logging.warning(
                "`optimize_gpu_memory_allocation` is set to False: `resources_predictors_actor` is None"
            )
        if self.collect_gpu_memory_usage and self.resources_predictors_actor is None:
            self.collect_gpu_memory_usage = False
            logging.warning(
                "`collect_gpu_memory_usage` is set to False: `resources_predictors_actor` is None"
            )

        if self.upload_model and self.s3_models_bucket is None:
            self.upload_model = False
            logging.warning("`upload_model` is set to False: `s3_model_bucket` is None")

    def _predict_resource(
        self, resource_predicted: ResourcePredictorType, task: TaskType, features: dict
    ) -> Tuple[Union[List[Any], Optional[float]], Optional[float]]:
        """
        Call the resources predictors actor to get a prediction for a specific resource and a specific task

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        features:
            The features to use when calling the model

        Returns
        -------
        The predicted resource raw, and the predicted resource ready to use (offset)
        """
        model_metrics = ray.get(
            self.resources_predictors_actor.get_model_metrics.remote(
                resource_predicted, task
            )
        )

        nrmse_threshold = 1.0
        prediction_offset = 0.0
        if resource_predicted == ResourcePredictorType.MAX_MEMORY:
            nrmse_threshold = self.optimize_memory_allocation_nrmse_threshold
            prediction_offset = self.max_memory_offset
        elif resource_predicted == ResourcePredictorType.MAX_GPU_MEMORY:
            nrmse_threshold = self.optimize_gpu_memory_allocation_nrmse_threshold
            prediction_offset = self.max_gpu_memory_offset

        if model_metrics["NRMSE"] < nrmse_threshold:
            predicted_resource = ray.get(
                self.resources_predictors_actor.predict.remote(
                    resource_predicted, task, features
                )
            )

            return predicted_resource, predicted_resource * (1.0 + prediction_offset)

        return None, None

    @staticmethod
    def run_with_ray_remote(task: TaskType) -> Callable:
        """
        Method to run a specific task with ray remote (used as a decorator)

        Args:
            task: The task type that will be run

        Returns:
            The decorator
        """

        def decorator(function: Callable) -> Callable:
            """
            The decorator used to run a task with ray remote

            Args:
                function: The function to run

            Returns:
                The wrapper running the function
            """

            class FunctionWrapperActor:
                @staticmethod
                def run(function_to_run, *args, **kwargs):
                    return function_to_run(*args, **kwargs)

            def _func(task_object, *args, **kwargs):
                """
                The wrapper of a function that run within Ray cluster.

                Args:
                    task_object:
                        The AAITask object (self)
                    args:
                        The arguments to pass to the function
                    kwargs:
                        The named arguments to pass to the function

                Returns:
                    The result of the function
                """
                import logging
                import tensorflow as tf

                if task_object.seed is not None:
                    # Note: This may increase computation time. See more
                    # information at https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
                    tf.compat.v1.disable_eager_execution()  # Necessary to set seed for tf on first run
                    tf.config.experimental.enable_op_determinism()
                    tf.keras.utils.set_random_seed(
                        task_object.seed
                    )  # Equivalent to calling random.seed(), np.random.seed(), and tf.random.set_seed()

                gpus = tf.config.list_physical_devices("GPU")
                if gpus is not None:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logging.warning(
                            f"Visible devices must be set before GPUs have been initialized ({e})"
                        )

                process = None
                cpu_affinity = None

                if task_object.use_ray:
                    import math
                    import psutil
                    import numpy as np

                    num_cpus = task_object.ray_params.get("num_cpus")
                    if num_cpus is not None:
                        num_cpus = math.ceil(num_cpus)
                        process = psutil.Process()
                        cpu_affinity = process.cpu_affinity()

                        cpu_count = len(cpu_affinity)
                        if num_cpus > cpu_count:
                            num_cpus = cpu_count

                        cpu_percent = psutil.cpu_percent(
                            interval=task_object.cpu_percent_interval, percpu=True
                        )

                        new_cpu_affinity = (
                            np.array(cpu_percent).argsort()[:num_cpus].tolist()
                        )
                        process.cpu_affinity(new_cpu_affinity)

                data = function(task_object, *args, **kwargs)

                # Restore default affinity
                if cpu_affinity is not None:
                    process.cpu_affinity(cpu_affinity)

                if type(data) == dict and "model" in data and data["model"] is not None:
                    if task_object.upload_model:
                        # Upload model
                        import os
                        import boto3
                        import dill as pickle

                        s3 = boto3.resource("s3")
                        bucket = s3.Bucket(task_object.s3_models_bucket)
                        path = os.path.join(task_object.s3_models_prefix, "model.p")
                        bucket.put_object(Key=path, Body=pickle.dumps(data["model"]))

                    # Delete model if needed
                    if not task_object.return_model:
                        del data["model"]

                return data

            @wraps(function)
            def wrapper(task_object: AAITask, *args, **kwargs):
                """
                The wrapper running the function with ray remote

                Args:
                    task_object:
                        The AAITask object (self)
                    args:
                        The arguments to pass to the function
                    kwargs:
                        The named arguments to pass to the function

                Returns:
                    The result of the function
                """
                from actableai.utils.resources.profile import profile_function
                from actableai.utils.resources.predict.features import extract_features

                if task_object.use_ray:
                    ray_params = deepcopy(task_object.ray_params)

                    (
                        max_memory_prediction_features,
                        max_memory_prediction_features_full,
                    ) = (None, None)
                    (
                        max_gpu_memory_prediction_features,
                        max_gpu_memory_prediction_features_full,
                    ) = (None, None)
                    predicted_max_memory_raw = None
                    predicted_max_gpu_memory_raw = None

                    # Extract features from the function arguments
                    if (
                        task_object.optimize_memory_allocation
                        or task_object.collect_memory_usage
                    ):
                        (
                            max_memory_prediction_features,
                            max_memory_prediction_features_full,
                        ) = extract_features(
                            ResourcePredictorType.MAX_MEMORY,
                            task,
                            function,
                            task_object,
                            *args,
                            **kwargs,
                        )
                    if (
                        task_object.optimize_gpu_memory_allocation
                        or task_object.collect_gpu_memory_usage
                    ):
                        (
                            max_gpu_memory_prediction_features,
                            max_gpu_memory_prediction_features_full,
                        ) = extract_features(
                            ResourcePredictorType.MAX_GPU_MEMORY,
                            task,
                            function,
                            task_object,
                            *args,
                            **kwargs,
                        )

                    # Optimize memory allocation
                    if task_object.optimize_memory_allocation:
                        (
                            predicted_max_memory_raw,
                            predicted_max_memory,
                        ) = task_object._predict_resource(
                            ResourcePredictorType.MAX_MEMORY,
                            task,
                            max_memory_prediction_features,
                        )

                        if (
                            predicted_max_memory is not None
                            and predicted_max_memory > 0
                        ):
                            ray_params["memory"] = int(predicted_max_memory)

                    # Optimize GPU memory allocation
                    if task_object.optimize_gpu_memory_allocation:
                        (
                            predicted_max_gpu_memory_raw,
                            predicted_max_gpu_memory,
                        ) = task_object._predict_resource(
                            ResourcePredictorType.MAX_GPU_MEMORY,
                            task,
                            max_gpu_memory_prediction_features,
                        )

                        if (
                            predicted_max_gpu_memory is not None
                            and predicted_max_gpu_memory > 0
                        ):
                            if "resources" not in ray_params:
                                ray_params["resources"] = {}
                            ray_params["resources"]["gpu_memory"] = int(
                                predicted_max_gpu_memory
                            )
                            # FIXME should it take into account the num_gpus given by the user?
                            # TODO add num_gpus to the ray params

                    resource_to_profile = 0
                    if task_object.collect_memory_usage:
                        resource_to_profile |= (
                            ResourceProfilerType.RSS_MEMORY
                            | ResourceProfilerType.SWAP_MEMORY
                        )
                    if task_object.collect_gpu_memory_usage:
                        resource_to_profile |= ResourceProfilerType.GPU_MEMORY

                    # Launch the function/task with ray
                    if (
                        task_object.collect_memory_usage
                        or task_object.collect_gpu_memory_usage
                    ):
                        actor_handle = (
                            ray.remote(FunctionWrapperActor)
                            .options(**ray_params)
                            .remote()
                        )
                        profiling_results, data = ray.get(
                            actor_handle.run.options(name=task).remote(
                                profile_function,
                                resource_to_profile,
                                True,
                                _func,
                                task_object,
                                *args,
                                **kwargs,
                            )
                        )

                        # If the task succeeded add the collected data to the respective models
                        if (
                            type(data) is not dict
                            or data.get("status", "SUCCESS") == "SUCCESS"
                        ):
                            if task_object.collect_memory_usage:
                                max_memory = int(
                                    profiling_results.get_max_profiled(
                                        ResourceProfilerType.RSS_MEMORY
                                        | ResourceProfilerType.SWAP_MEMORY
                                    )
                                )

                                logging.info(
                                    f"Collect max memory usage for task: {task}"
                                )
                                logging.info(
                                    f"    predicted: {predicted_max_memory_raw}"
                                )
                                logging.info(f"    observed: {max_memory}")

                                task_object.resources_predictors_actor.add_data.remote(
                                    ResourcePredictorType.MAX_MEMORY,
                                    task,
                                    max_memory_prediction_features,
                                    max_memory,
                                    prediction=predicted_max_memory_raw,
                                    full_features=max_memory_prediction_features_full,
                                )

                            if task_object.collect_gpu_memory_usage:
                                max_gpu_memory = int(
                                    profiling_results.get_max_profiled(
                                        ResourceProfilerType.GPU_MEMORY
                                    )
                                )

                                logging.info(
                                    f"Collect max GPU memory usage for task: {task}"
                                )
                                logging.info(
                                    f"    predicted: {predicted_max_gpu_memory_raw}"
                                )
                                logging.info(f"    observed: {max_gpu_memory}")

                                task_object.resources_predictors_actor.add_data.remote(
                                    ResourcePredictorType.MAX_GPU_MEMORY,
                                    task,
                                    max_gpu_memory_prediction_features,
                                    max_gpu_memory,
                                    prediction=predicted_max_gpu_memory_raw,
                                    full_features=max_gpu_memory_prediction_features_full,
                                )
                    else:
                        actor_handle = (
                            ray.remote(FunctionWrapperActor)
                            .options(**ray_params)
                            .remote()
                        )
                        data = ray.get(
                            actor_handle.run.options(name=task).remote(
                                _func, task_object, *args, **kwargs
                            )
                        )

                    ray.kill(actor_handle)

                    return data

                return _func(task_object, *args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def get_parameters(cls) -> Parameters:
        return Parameters(
            name="undefined",
            display_name="Undefined",
            parameters=[],
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Abstract method called to run the task
        """
        raise NotImplementedError()


class AAITunableTask(AAITask, ABC):
    """Base abstract class to represent a Tunable Actable AI Task."""

    @staticmethod
    @abstractmethod
    def get_hyperparameters_space(*args, **kwargs) -> ModelSpace:
        """Return the hyperparameters space oof the task.

        Returns:
            Hyperparameters space represented as a ModelSpace.
        """
        raise NotImplementedError()
