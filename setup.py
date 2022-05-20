from setuptools import setup, find_packages
#setup(

    #name='testdummy',
    #version='0.0.1',
    #install_requires=[
        #'importlib-metadata; python_version == "3.8"',
    #],

    #packages=find_packages("contrib"),  # include all packages under src
    #package_dir={"": "contrib"},   # tell distutils packages are under src

    #package_data={
        ## If any package contains *.txt files, include them:
        #"": ["*.txt"],
        ## And include any *.dat files found in the "data" subdirectory
        ## of the "mypkg" package, also:
        #"": ["data/*.npz"],
    #}
#)




setup(
    name='inversiontestproblems',
    version='0.0.1',
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ],
    include_package_data=True,
    #packages=find_packages(exclude=("*.egg_info")),  
    package_dir={"inversiontestproblems": "contrib"},   # tell distutils packages are under src
)

