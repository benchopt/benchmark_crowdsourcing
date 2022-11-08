from benchopt import BaseObjective, safe_import_context

import os
import sys

from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import joblib
    import torch
    import tensorflow as tf
    from tqdm import tqdm
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities.seed import seed_everything

    BenchPLModule = import_ctx.import_from("lightning_helper", "BenchPLModule")
    AugmentedDataset = import_ctx.import_from(
        "lightning_helper", "AugmentedDataset"
    )
    change_classification_head_tf = import_ctx.import_from(
        "tf_vgg", "change_classification_head"
    )
    remove_initial_downsample = import_ctx.import_from(
        "torch_resnets", "remove_initial_downsample"
    )
    change_classification_head_torch = import_ctx.import_from(
        "torch_vgg", "change_classification_head"
    )
    TFResNet18 = import_ctx.import_from("tf_resnets", "ResNet18")
    TFResNet34 = import_ctx.import_from("tf_resnets", "ResNet34")
    TFResNet50 = import_ctx.import_from("tf_resnets", "ResNet50")

    TF_MODEL_MAP = {
        "resnet": {
            "18": TFResNet18,
            "34": TFResNet34,
            "50": TFResNet50,
        },
        "vgg": {
            "16": tf.keras.applications.vgg16.VGG16,
        },
    }

    TORCH_MODEL_MAP = {
        "resnet": {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
        },
        "vgg": {
            "16": models.vgg16,
        },
    }


class Objective(BaseObjective):
    name = "Crowdsourcing"

    def skip(
        self,
        dataset,
        val_dataset,
        test_dataset,
        votes,
        n_samples_train,
        n_samples_val,
        n_samples_test,
        image_width,
        n_classes,
        framework,
        normalization,
        symmetry,
        y_train_truth,
        n_workers,
        **kwargs,
    ):
        if framework == "tensorflow" and image_width < 32:
            return True, "images too small for TF networks"
        return False, None

    def get_tf_model_init_fn(self):
        model_klass = TF_MODEL_MAP[self.model_type][str(self.model_size)]
        add_kwargs = {}
        input_width = self.width
        if self.model_type == "resnet":
            add_kwargs["use_bias"] = False
            add_kwargs["dense_init"] = "torch"

            # For now 128 is an arbitrary number
            # to differentiate big and small images
            if self.width < self.image_width_cutout:
                input_width = 4 * self.width
                add_kwargs["no_initial_downsample"] = True
            else:
                add_kwargs["no_initial_downsample"] = False

        def _model_init_fn():
            is_vgg = self.model_type == "vgg"
            is_small_images = self.width < self.image_width_cutout
            model = model_klass(
                weights=None,
                classes=self.n_classes,
                classifier_activation="softmax",
                input_shape=(input_width, input_width, 3),
                **add_kwargs,
            )
            if is_vgg and is_small_images:
                model = change_classification_head_tf(model)
            return model

        return _model_init_fn

    def get_torch_model_init_fn(self):
        model_klass = TORCH_MODEL_MAP[self.model_type][str(self.model_size)]

        def _model_init_fn():
            model = model_klass(num_classes=self.n_classes)
            is_resnet = self.model_type == "resnet"
            is_vgg = self.model_type == "vgg"
            is_small_images = self.width < self.image_width_cutout
            if is_resnet and is_small_images:
                model = remove_initial_downsample(model)
            if is_vgg and is_small_images:
                model = change_classification_head_torch(model)
            if torch.cuda.is_available():
                model = model.cuda()
            return model

        return _model_init_fn

    def get_lightning_model_init_fn(self):
        torch_model_init_fn = self.get_torch_model_init_fn()

        def _model_init_fn():
            model = torch_model_init_fn()
            return BenchPLModule(model)

        return _model_init_fn

    def get_model_init_fn(self, framework):
        if framework == "tensorflow":
            return self.get_tf_model_init_fn()
        elif framework == "lightning":
            return self.get_lightning_model_init_fn()
        elif framework == "pytorch":
            return self.get_torch_model_init_fn()
        else:
            raise ValueError(f"No framework named {framework}")

    def set_data(
        self,
        dataset,
        val_dataset,
        test_dataset,
        votes,
        n_samples_train,
        n_samples_val,
        n_samples_test,
        image_width,
        n_classes,
        framework,
        normalization,
        symmetry,
        y_train_truth,
        n_workers,
        **kwargs,
    ):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.with_validation = val_dataset is not None
        self.test_dataset = test_dataset
        self.n_samples_train = n_samples_train
        self.n_samples_val = n_samples_val
        self.n_samples_test = n_samples_test
        self.width = image_width
        self.n_classes = n_classes
        self.framework = framework
        self.normalization = normalization
        self.symmetry = symmetry
        self.votes = votes
        self.y_train_truth = y_train_truth
        self.n_workers = n_workers

        # Get the model initializer
        self.get_one_solution = {
            "yhat": np.zeros(len(self.votes)),
            "model": self.get_model_init_fn(framework),
        }

    def compute(self, res):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        yhat = res["yhat"]
        results = dict()
        if yhat.ndim > 1:
            top1 = np.argmax(yhat, axis=1)
        else:
            top1 = yhat
        train_accuracy = np.mean(top1 == self.y_train_truth)
        results["Train Accuracy"] = train_accuracy
        results["value"] = train_accuracy
        return results

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(
            train=self.train,
            val=self.val,
            test=self.test,
            votes=self.votes,
            y_train_truth=self.y_train_truth,
            n_classes=self.n_classes,
            n_workers=self.n_workers,
        )
