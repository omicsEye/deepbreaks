try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")

import os
import urllib

try:
    from urllib.request import urlretrieve

    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
except ImportError:
    from urllib import urlretrieve

    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]

VERSION = "1.1.1"
AUTHOR = "Ali Rahnavard"
AUTHOR_EMAIL = "gholamali.rahnavard@gmail.com"
MAINTAINER = "Ali Rahnavard"
MAINTAINER_EMAIL = "gholamali.rahnavard@gmail.com"

# try to download the bitbucket counter file to count downloads
# this has been added since PyPI has turned off the download stats
# this will be removed when PyPI Warehouse is production as it
# will have download stats
COUNTER_URL = "https://github.com/omicsEye/deepBreaks/blob/master/README.md"
counter_file = "README.md"
if not os.path.isfile(counter_file):
    print("Downloading counter file to track deepBreaks downloads" +
          " since the global PyPI download stats are currently turned off.")
    try:
        pass  # file, headers = urlretrieve(COUNTER_URL,counter_file)
    except EnvironmentError:
        print("Unable to download counter")

setup(
    name="deepBreaks",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version=VERSION,
    license="MIT",
    description="deepBreaks: multi-resolution clustering",
    long_description="deepBreaks provides an elegant clustering approach " + \
                     "to find clusters in data sets with different density and resolution.",
    url="http://github.com/omicsEye/deepBreaks",
    keywords=['clustering', 'multi-resolution', 'hierarchically'],
    platforms=['Linux', 'MacOS', "Windows"],
    classifiers=classifiers,
    # long_description=open('readme.md').read(),
    install_requires=[
        "latex >= 0.0.1",
        "Cython >= 0.29.2",
        "Numpy >= 1.9.2",
        "Scipy >= 0.17.0",
        "Matplotlib >= 1.5.1",
        "Scikit-learn >= 0.14.1",
        "pandas >= 0.18.1",
        "Community == 1.0.0b1",
        "networkx >= 0.2.5"
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'deepBreaks = deepBreaks.deepBreaks:main',
            'deepBreaksviz = deepBreaks.viz:main',
            'deepBreaks_test = deepBreaks.tests.deepBreaks_test:main'
        ]},
    test_suite='deepBreaks.tests.deepBreaks_test',
    zip_safe=False
)