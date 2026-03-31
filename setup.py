from setuptools import setup
from scresid import __version__

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
      name='scresid',  # pkg_name
      packages=['scresid',],
      version=__version__,  # version number
      description="Single-cell perturbation prediction with residual connections across cell types.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/scresid',
      download_url='https://github.com/linjing-lab/scresid/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/scresid/tree/main/scresid/',
            'Tracker': 'https://github.com/linjing-lab/scresid/issues',
      },
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=[
            'adjustText>=1.3.0',
            'matplotlib>=3.10.5',
            'numpy>=1.26.4', # numpy==1.26.4
            'pandas>=2.3.1',
            'POT>=0.9.5',
            'scanpy>=1.10.4',
            'scikit-learn>=1.7.1',
            'seaborn>=0.13.2',
            'tqdm>=4.67.1',
      ],
      # extras_require=[]
)