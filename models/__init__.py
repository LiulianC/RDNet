import importlib

from models.arch import *

from models.cls_model_eval_nocls_reg import ClsModel


def make_model(name: str):

    model = ClsModel()
    return model
