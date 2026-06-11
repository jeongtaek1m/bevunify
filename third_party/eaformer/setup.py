from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='eaformer',
    version=__version__,
    author='Brady Zhou',
    author_email='brady.zhou@utexas.edu',
    url='https://github.com/bradyz/eaformers',
    license='MIT',
    packages=find_packages(include=['eaformer', 'eaformer.*']),
    zip_safe=False,
)
