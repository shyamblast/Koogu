import os
import warnings
from koogu.model.trained_model import TrainedModel
from koogu.model import architectures


def add_architectures():
    # warnings.showwarning(
    #     'The interface koogu.model.Architectures is deprecated and will be ' +
    #     'removed in a future release. Instead, please use the interface ' +
    #     'koogu.model.architectures to create models.',
    #     DeprecationWarning, __name__, '')

    def subcls_interface_depr(old_name, new_name):
        def wrapped(subcls):
            def final(*args, **kwargs):
                warnings.showwarning(
                    'The interface koogu.model.Architectures is deprecated ' +
                    'and will be removed in a future release. Instead of ' +
                    'koogu.model.Architectures.{} please use'.format(old_name) +
                    'koogu.model.architectures.{} to '.format(new_name) +
                    'create models.',
                    DeprecationWarning, __name__, '')

                return subcls(*args, **kwargs)

            return final

        return wrapped

    def adder(cls):
        archs_dir = os.path.join(os.path.dirname(__file__), 'architectures')
        for name in (model[:-3] for model in os.listdir(archs_dir)
                     if model.endswith('.py') and model != '__init__.py'):
            module = getattr(architectures, name, None)
            if module is not None:
                arch_cls = getattr(module, module.__all__[0], None)
                if arch_cls is not None and \
                        issubclass(arch_cls,
                                   architectures.KooguArchitectureBase):
                    setattr(cls, name,  # wrap for deprecation
                            subcls_interface_depr(name,
                                                  module.__all__[0])(arch_cls))
        return cls
    return adder


@add_architectures()
class Architectures:
    pass


__all__ = ['TrainedModel', 'Architectures']
