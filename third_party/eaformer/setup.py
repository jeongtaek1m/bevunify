from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='eaformer',
    version=__version__,
    author='Brady Zhou',
    author_email='brady.zhou@utexas.edu',
    description='EAFormer (WACV25, arXiv:2412.01595) reimplementation — fork of cross_view_transformers',
    url='https://github.com/bradyz/cross_view_transformers',
    license='MIT',
    packages=find_packages(include=['eaformer', 'eaformer.*']),
    zip_safe=False,
)
