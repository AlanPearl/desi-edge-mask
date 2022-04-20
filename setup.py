from setuptools import setup, find_packages

setup(name="edgemask",
      version="0.0.1.dev1",
      description="For masking data near the edges of the DESI SV3 bright galaxy catalog",
      url="https://github.com/AlanPearl/galtab",
      author="Alan Pearl",
      author_email="alanpearl@pitt.edu",
      license="MIT",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=[
          "numpy",
          "matplotlib",
          "opencv-python",
      ],
      zip_safe=True,
      test_suite="nose.collector",
      tests_require=["nose"]
      )
