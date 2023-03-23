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

logger = logging.getLogger('dummy-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for Dummy Model',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """
    Dummy Model adapter using pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity=None):
        if not isinstance(model_entity, dl.Model):
            # pending fix DAT-31398
            if isinstance(model_entity, str):
                model_entity = dl.models.get(model_id=model_entity)
            if isinstance(model_entity, dict) and 'model_id' in model_entity:
                model_entity = dl.models.get(model_id=model_entity['model_id'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ModelAdapter, self).__init__(model_entity=model_entity)

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
                               model_info={'name': "Dummy Model",
                                           'confidence': index * 0.1,
                                           'model_id': self.model_entity.id,
                                           'dataset_id': self.model_entity.dataset_id})
                logger.debug("Predicted {} ({})".format(str(index), index * 0.1))
            batch_annotations.append(collection)

        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured
            Virtual method - need to implement
            e.g. take dlp dir structure and construct annotation file
        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'model.pth',
                                                                 'input_size': 256},
                                          output_type=dl.AnnotationType.POINT,
                                          )
    module = dl.PackageModule.from_entry_point(entry_point='dummy_adapter.py')
    package = project.packages.push(package_name='dummy',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop ResNet implemented in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(
                                        git_url='https://github.com/OfirDataloopAI/dummy_model_adapter',
                                        git_tag='master'),
                                    modules=[module],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_HIGHMEM_L,
                                                                        runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.7',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    # package.metadata = {'system': {'ml': {'defaultConfiguration': {'weights_filename': 'model.pth',
    #                                                                'input_size': 256},
    #                                       'outputType': dl.AnnotationType.CLASSIFICATION,
    #                                       'tags': ['torch'], }}}
    # package = package.update()
    s = package.services.list().items[0]
    s.package_revision = package.version
    s.versions['dtlpy'] = '1.74.9'
    s.update(True)
    return package


def model_creation(package: dl.Package, project: dl.Project):
    labels = list()
    for i in range(10):
        labels.append(str(i))

    model = package.models.create(model_name='dummy-model',
                                  description='dummy-model for testing',
                                  tags=['pretrained', 'no-data'],
                                  dataset_id=None,
                                  scope='public',
                                  # scope='project',
                                  model_artifacts=[dl.LinkArtifact(
                                      type=dl.PackageCodebaseType.GIT,
                                      url='https://storage.googleapis.com/model-mgmt-snapshots/ResNet50/model.pth',
                                      filename='model.pth')],
                                  status='trained',
                                  configuration={'weights_filename': 'model.pth',
                                                 'batch_size': 16,
                                                 'num_epochs': 10},
                                  project_id=project.id,
                                  labels=labels,
                                  )
    return model


if __name__ == "__main__":
    env = 'rc'
    project_name = 'QA_MODELS_V3'
    dl.setenv(env)
    project = dl.projects.get(project_name)
    package_creation(project)
    package = project.packages.get('dummy')
    # package.artifacts.list()
    model_creation(package=package, project=project)

    # Useful:
    #https://github.com/dataloop-ai/pytorch_adapters/blob/mgmt3/resnet_adapter.py
