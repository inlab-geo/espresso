import importlib

#import os
#path_to_current_file = os.path.realpath(__file__)
#current_directory = os.path.split(path_to_current_file)[0]
#current_directory=current_directory.replace("/", ".")
#current_directory=current_directory + ".Ex1_dir"
#print(current_directory)


# Get the object ‘Ex1’ from ‘first-example-directory’ and rename it ‘ExampleOne'
gravityforward = getattr(importlib.import_module(".gravityforward", package='inversiontestproblems'), "gravity")

## Get the object ‘MyTestProblem’ from ‘another-directory’ and rename it ‘ExampleTwo'
#ExampleTwo = getattr(importlib.import_module(".Ex2_dir", package='inversiontestproblems'), "hello")

## More imported objects
#ExampleThree = getattr(importlib.import_module(".Ex3_dir", package='inversiontestproblems'), "hello")
#ExampleFour = getattr(importlib.import_module(".Ex4_dir", package='inversiontestproblems'), "hello")

