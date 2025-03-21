

import os
import sys

from os.path import join
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False


def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    
    config = Configuration('shallow_water', parent_package, top_path)

    config.add_data_dir('tests')
    config.add_data_dir(join('tests', 'data'))

    util_dir = join('..', 'utilities')

    config.add_extension('sw_domain_orig_ext',
                         sources = ['sw_domain_orig_ext.pyx'],
                         include_dirs = [util_dir])

    config.ext_modules = cythonize(config.ext_modules, annotate=False)

    return config
    
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
