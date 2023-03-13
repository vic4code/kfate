#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse

import torch as t
from torch import nn
from pipeline import fate_torch_hook
from pipeline.component import HomoNN
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, Evaluation, DataTransform
from pipeline.interface import Data, Model
from pipeline.utils.tools import load_job_config
import sys
import os
import torch as t
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

sys.path.append('/data/projects/fate/persistence/fate/python/fate_client')
sys.path.append('/data/projects/fate/persistence/fate/python')

os.chdir("/data/projects/fate/persistence/fate")
os.environ['FATE_PROJECT_BASE'] = '/data/projects/fate/persistence/fate'
    
fate_torch_hook(t)

class Resnet(nn.Module):

    def __init__(self, ):
        super(Resnet, self).__init__()
        self.resnet = resnet18()
        self.classifier = t.nn.Linear(1000, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.training:
            return self.classifier(self.resnet(x))
        else:
            return self.softmax(self.classifier(self.resnet(x)))

def main(config="/data/projects/fate/persistence/fate/examples/config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
        
    fate_project_path = os.path.abspath('/data/projects/fate')
    guest = 9999
    host = 10000
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host,
                                                                                arbiter=host)
    data_0 = {"name": "cifar10", "namespace": "experiment"}
    data_path = fate_project_path + '/examples/data/toy_cifar10/train'
    pipeline.bind_table(name=data_0['name'], namespace=data_0['namespace'], path=data_path)
    pipeline.bind_table(name=data_0['name'], namespace=data_0['namespace'], path=data_path)
    
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=data_0)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=data_0)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=data_0)
    reader_1.get_party_instance(role='host', party_id=host).component_param(table=data_0)
    
    from pipeline.component.homo_nn import DatasetParam, TrainerParam

    model = t.nn.Sequential(
    t.nn.CustModel(module_name='resnet', class_name='Resnet')
)

    nn_component = HomoNN(name='nn_0',
                          model=model, 
                          loss=t.nn.CrossEntropyLoss(),
                          optimizer = t.optim.Adam(lr=0.001, weight_decay=0.001),
                          dataset=DatasetParam(dataset_name='image'),  # 使用自定义的dataset
                          trainer=TrainerParam(trainer_name='fedavg_trainer', epochs=10, batch_size=1024, data_loader_worker=8),
                          torch_seed=100
                          )
    
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(nn_component, data=Data(train_data=reader_0.output.data, validate_data=reader_1.output.data))
    pipeline.add_component(Evaluation(name='eval_0', eval_type='multi'), data=Data(data=nn_component.output.data))
    
    pipeline.compile()
    pipeline.fit() # submit pipeline here

    print(pipeline.get_component("nn_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RESNET DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
