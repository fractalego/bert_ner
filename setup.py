from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='bert_ner',
      version='0.0.1',
      url='http://github.com/fractalego/bert_ner',
      author='Alberto Cetoli',
      author_email='alberto@nlulite.com',
      description="A test for NER with BERT",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['bert_ner'],
      install_requires=[
          'numpy==1.18.1',
          'transformers==2.5.1',
          'pytorch-transformers==1.2.0',
      ],
      classifiers=[
          'License :: OSI Approved :: MIT License',
      ],
      include_package_data=True,
      zip_safe=False)
