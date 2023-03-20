from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional
import pandas as pd
import numpy as np
import dtlpy as dl
import torch.nn
import logging
import torch
import time
import copy
import tqdm
import os

from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

import random

logger = logging.getLogger('cnn-adapter')


class ModelAdapter(dl.BaseModelAdapter):
    """
    cnn Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self, local_path, **kwargs):
        logger.info("Loaded model successfully :P")

    def save(self, local_path, **kwargs):
        logger.info("Saved model successfully :D")

    def train(self, data_path, output_path, **kwargs):
        logger.info("Entering fake training XD")

        for i in range(11):
            logger.warning("My Progress: {} %".format(i * 10))

        logger.info("Completed fake training :)")

    def predict(self, batch, **kwargs):
        batch_annotations = list()

        for img in batch:
            collection = dl.AnnotationCollection()
            result = random.randint(1, 10)
            for index in range(result):
                collection.add(annotation_definition=dl.Point(label=str(index), x=index * 10, y=index * 10),
                               model_info={'name': "Faker",
                                           'confidence': index * 0.1,
                                           'model_id': self.model_entity.id,
                                           'snapshot_id': self.snapshot.id})
                logger.debug("Predicted {} ({})".format(str(index), index * 0.1))
            batch_annotations.append(collection)
        return batch_annotations


def _get_imagenet_label_json():
    import json
    with open('imagenet_labels.json', 'r') as fh:
        labels = json.load(fh)
    return list(labels.values())


def model_creation(project_name, env: str = 'prod'):
    dl.setenv(env)
    project = dl.projects.get(project_name)

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                              git_tag='master')
    model = project.models.create(model_name='Dummy',
                                  description='Dummy model implemented in pytorch',
                                  output_type=dl.AnnotationType.POINT,
                                  scope='public',
                                  codebase=codebase,
                                  tags=['torch'],
                                  default_configuration={
                                      'weights_filename': 'model.pth',
                                      'input_size': 256,
                                  },
                                  default_runtime=dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_HIGHMEM_L,
                                                                       runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.6',
                                                                       autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                           min_replicas=0,
                                                                           max_replicas=1),
                                                                       concurrency=1),
                                  entry_point='dummy_adapter.py')
    return model


# def snapshot_creation(project_name, model: dl.Model, env: str = 'prod', resnet_ver='50'):
#     dl.setenv(env)
#     project = dl.projects.get(project_name)
#     bucket = dl.buckets.create(dl.BucketType.GCS,
#                                gcs_project_name='viewo-main',
#                                gcs_bucket_name='model-mgmt-snapshots',
#                                gcs_prefix='ResNet{}'.format(resnet_ver))
#     snapshot = model.snapshots.create(snapshot_name='pretrained-resnet{}'.format(resnet_ver),
#                                       description='resnset{} pretrained on imagenet'.format(resnet_ver),
#                                       tags=['pretrained', 'imagenet'],
#                                       dataset_id=None,
#                                       scope='public',
#                                       # status='trained',
#                                       configuration={'weights_filename': 'model.pth',
#                                                      'classes_filename': 'classes.json'},
#                                       project_id=project.id,
#                                       bucket=bucket,
#                                       labels=_get_imagenet_label_json()
#                                       )
#     return snapshot


def model_and_snapshot_creation(project_name, env: str = 'prod'):
    model = model_creation(project_name, env=env)
    print("Model : {} - {} created".format(model.name, model.id))
    # snapshot = snapshot_creation(project_name, model=model, env=env, resnet_ver=resnet_ver)
    # print("Snapshot : {} - {} created".format(snapshot.name, snapshot.id))


if __name__ == "__main__":
    model_and_snapshot_creation("Abeer N Ofir Project")
