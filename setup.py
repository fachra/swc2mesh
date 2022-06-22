from setuptools import setup

version = {}
with open('swc2mesh/_version.py') as fp:
    exec(fp.read(), version)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='swc2mesh',
    version=version['__version__'],
    description='Convert neuronal morphological reconstructions to watertight surface meshes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fachra/swc2mesh/',
    author='Chengran Fang',
    author_email='victor.fachra@gmail.com',
    license='GPLv3',

    packages=['swc2mesh'],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'trimesh',
        'pymeshlab',
    ],
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    keywords='diffusion mri, simulation, meshes, neuron reconstructions',
    project_urls={
    'Source': 'https://github.com/fachra/swc2mesh',
    'Tracker': 'https://github.com/fachra/swc2mesh/issues',
    },
)
