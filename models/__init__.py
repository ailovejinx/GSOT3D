import os.path as osp
import importlib
import inspect
import glob
from .base_model import BaseModel


def create_model(cfg, log):

    filenames = glob.glob(osp.join(osp.dirname(__file__), '*', 'model.py'))
    filenames = [fn.split(osp.sep)[-2] for fn in filenames]
    type2model = dict()
    for filename in filenames:
        module = importlib.import_module('models.%s.model' % filename)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for clsmember in clsmembers:
            is_model = False
            for base_cls in inspect.getmro(clsmember[1]):
                if base_cls is BaseModel or isinstance(base_cls, BaseModel):
                    is_model = True
                    break
            if is_model:
                type2model[clsmember[0]] = clsmember[1]

    # print(type2model)
    return type2model[cfg.model_type](cfg, log)
