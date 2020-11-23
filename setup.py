from setuptools import setup, find_packages

setup(
    name='mlsummary',
    version='0.1.0',
    install_requires=['numpy', 'pandas', 'scikit-learn', 'mlxtend'],
    packages= find_packages(),
    url='https://github.com/serafinialessio/mlsummary',
    license='"BSD"',
    author='Alessio Serafini',
    author_email='srf.alessio@gmail.com',
    description='Summary package for machinne learning algorithms.',
    #keywords = []
    platforms='any'
)
