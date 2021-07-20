import os
from koogu.model.trained_model import TrainedModel
from koogu.model import architectures
from koogu.model.architectures import BaseArchitecture


def add_architectures():
    def adder(cls):
        archs_dir = os.path.join(os.path.dirname(__file__), 'architectures')
        for name in (model[:-3] for model in os.listdir(archs_dir)
                     if model.endswith('.py') and model != '__init__.py'):
            module = getattr(architectures, name, None)
            if module is not None:
                arch_cls = getattr(module, 'Architecture', None)
                if arch_cls is not None and \
                        issubclass(arch_cls,
                                   architectures.KooguArchitectureBase):
                    setattr(cls, name, arch_cls)
        return cls
    return adder


@add_architectures()
class Architectures:
    pass


__all__ = ['TrainedModel', 'BaseArchitecture', 'Architectures']
