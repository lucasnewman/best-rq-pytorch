from setuptools import setup, find_packages

setup(
    name = 'best-rq-pytorch',
    packages = find_packages(exclude=[]),
    version = '0.0.1',
    license='MIT',
    description = 'BEST-RQ - Pytorch',
    author = 'Lucas Newman',
    author_email = 'lucasnewman@me.com',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/lucasnewman/best-rq-pytorch',
    keywords = [
        'artificial intelligence',
        'asr',
        'audio-generation,'
        'deep learning',
        'transformers',
        'text-to-speech'
    ],
    install_requires=[
        'beartype',
        'einops>=0.6.1',
        'rotary-embedding-torch>=0.3.0',
        'torch>=2.0',
        'torchaudio>=2.0',
        'tqdm',
        'vector-quantize-pytorch>=1.7.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
