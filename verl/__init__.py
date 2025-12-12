import importlib.util
import os
import sys
from pkgutil import extend_path

_inner_pkg_dir = os.path.join(os.path.dirname(__file__), 'verl')

if os.path.isdir(_inner_pkg_dir):
    spec = importlib.util.spec_from_file_location(__name__, os.path.join(_inner_pkg_dir, '__init__.py'))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = __name__
    module.__spec__ = spec
    module.__path__ = [_inner_pkg_dir]
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
else:
    __path__ = extend_path(__path__, __name__)
