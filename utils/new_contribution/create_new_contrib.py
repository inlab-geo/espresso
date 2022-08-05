# Command to create a new example folder:
# In ROOT, execute:
# python utils/new_contribution/create_new_contrib.py <example-name>
# Replacing <example_name> with the new example name.

from glob import glob
import sys
import os
from shutil import copyfile

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


CONTRIB_FOLDER = "contrib"
TEMPLATE_FOLDER = "utils/new_contribution/_template"

if __name__ == '__main__':
    print("🥰 Thanks for contributing! \nWe are generating new contribution component from template...\n")

    # validate example name
    example_name = sys.argv[-1]
    existing_examples = glob(CONTRIB_FOLDER+"/*/")
    existing_examples = [e for e in existing_examples]
    existing_examples = [e.split("/")[1] for e in existing_examples]
    # print(existing_examples)
    if example_name in existing_examples:
        raise ValueError("The example name provided already exists, please choose another name")
    
    new_subfolder = CONTRIB_FOLDER+"/"+example_name
    os.makedirs(new_subfolder)    
    os.makedirs(new_subfolder+"/data")    

    template_files=getListOfFiles(TEMPLATE_FOLDER)
    new_files = [f.replace(TEMPLATE_FOLDER, new_subfolder) for f in template_files]
    new_files = [f.replace("example_name", example_name) for f in new_files]

    print("🗂  Copying files...")
    for template_file, new_file in zip(template_files, new_files):
        print("file: "+template_file+" -> "+new_file)
        copyfile(template_file, new_file)
    print("\n🎉 OK. Please navigate to " + new_subfolder + " to write your own example. ")
    