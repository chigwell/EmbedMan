from setuptools import setup, find_packages

setup(
    name='EmbedMan',
    version='0.0.2',
    author='Eugene Evstafev',
    author_email='chigwel@gmail.com',
    description='A tool for managing embeddings for code analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/chigwell/EmbedMan',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'langchain-community',
        'gpt4all',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
