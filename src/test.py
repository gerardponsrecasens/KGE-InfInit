import torch
from .utils import *
from .model.model_process import *
from .model.model_process_incDE import *


class Tester():
    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg 
        self.model = model
        if args.lifelong_name == 'incDE':
            self.test_processor = DevBatchProcessorincDE(args, kg)
        else:
            self.test_processor = DevBatchProcessor(args, kg)


    def test(self):
        self.args.valid = False
        res = self.test_processor.process_epoch(self.model)
        return res