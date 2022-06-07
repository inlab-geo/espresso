"""This script tests running all Python files under notebooks/ folder.
"""


import subprocess
from glob import glob
import os
import shutil
import filecmp
import re
import logging
import numpy

NOTEBOOKS_FOLDER = "notebooks"
OUTPUT_FOLDER = "utils/validation/_output"
VALIDATION_FOLDER = "utils/validation/_validation_output"

PYTHON = "python"

class bcolors:
    PASSED   = '\033[92m'
    WARNING  = '\033[95m'
    FAILED   = '\033[91m'
    MISSING = '\033[96m'
    ENDC     = '\033[0m'
    BOLD     = '\033[1m'

def listdir_nohidden():
    for dir in glob(f"{NOTEBOOKS_FOLDER}/*/"):
        if not dir.startswith("."):
            yield dir

def listfiles_nohidden(dir):
    for f in glob(f"{dir}/*"):
        if not f.startswith("."):
            yield f

def listpy_nohidden_notlib(dir):
    for f in listfiles_nohidden(dir):
        if f.endswith(".py") and "lib.py" not in f:
            yield f
    

def approximate_diff(file_name_1, file_name_2, tolerance=1.0e-03):
    """
    Takes the "approximate diff" of two files, allowing for
    differences in real data up to the specified tolerance.
    This isn't as sophisticated as a true diff in terms of
    matching largest common subsections.  It's just meant
    to speed the user's analysis if changing the code results
    in structural or numerical differences.
    Originally written by Jonathan Claridge.
    """

    def approximatelyEqual(x,y,tolerance):
        if x==y:
            return True

        try:
            if abs(float(x) - float(y)) < tolerance:
                return True
        except:
            return False

    file_1 = open(file_name_1, 'r')
    file_2 = open(file_name_2, 'r')
    lines_1 = file_1.readlines()
    lines_2 = file_2.readlines()
    difference = False
    any_difference = False
    max_absolute_difference = 0.
    max_relative_difference =0.
    max_i = 0
    
    diff_output = []
  
    #==== Check file lengths ====
    #--------------------------------------------------------------------
    # A problem here will indicate that something is structurally wrong.
    #--------------------------------------------------------------------
    if not(len(lines_1) == len(lines_2)):
        diff_output.append("Files are of different length")

    #==== Check line by line ====
    #----------------------------------------------------------------------
    # This is where numerical differences will be highlighted.  Also, if
    # the files are comparable up to a certain point, this will show where
    # they begin to diverge.
    #----------------------------------------------------------------------
    for i in range(min(len(lines_1), len(lines_2))):
        #split_1 = lines_1[i].split();
        #split_2 = lines_2[i].split();
        if lines_1[i].startswith('#'):
            continue
        if lines_1[i].startswith('>'):
            continue
        split_1=re.split(' |,',lines_1[i])
        split_2=re.split(' |,',lines_2[i])
        
        if len(split_1) == len(split_2):
            #-----------------------------------------------------------
            # If lines have the same number of elements, then check for
            # numerical differences.
            #-----------------------------------------------------------
            for j in range(len(split_1)):
                if not(approximatelyEqual(split_1[j],split_2[j],tolerance)):
                    diff_output.append("  Line " +  str(i+1) + ", element " \
                        + str(j+1) + " differs")
                    diff_output.append("  " + file_name_1.rjust(40) + ": " + split_1[j])
                    diff_output.append("  " + file_name_2.rjust(40) + ": " + split_2[j])
                if not(split_1[j] == split_2[j]):
                    try:
                        x1 = float(split_1[j])
                        x2 = float(split_2[j])
                        max_absolute_difference = max(abs(x1-x2), max_absolute_difference)
                        max_relative_difference = max(abs(x1-x2)/numpy.sqrt(x1**2+x2**2), max_relative_difference)
                        max_i = i+1
                    except:
                        max_absolute_difference = numpy.nan
        else:
            #-----------------------------------------------------------
            # If lines have a different number of elements, then print
            # their contents.
            #-----------------------------------------------------------
            diff_output.append("  Line " + str(i+1) + ": number of elements differs")
            diff_output.append("  " + file_name_1.rjust(40) + ": " + lines_1[i])
            diff_output.append("  " + file_name_2.rjust(40) + ": " + lines_2[i])

    return diff_output,max_absolute_difference, max_relative_difference,max_i


def analyse_cmp_res(match, mismatch, errors, outdir, regdir, files):
    if (len(mismatch)==0 and len(errors)==0):
        logging.critical(bcolors.PASSED +bcolors.BOLD+ '[PASSED] '+ bcolors.ENDC+regdir.replace(f"{VALIDATION_FOLDER}/",''))
    elif (len(mismatch)>0 and len(errors)==0):
        logging.warning(bcolors.WARNING +bcolors.BOLD+'[WARNING] '+regdir.replace(f"{VALIDATION_FOLDER}/",'') +bcolors.ENDC)
    elif (len(errors)==len(files)):
        logging.error(bcolors.MISSING +bcolors.BOLD+  '[MISSING] '+regdir.replace(f"{VALIDATION_FOLDER}/",'') +bcolors.ENDC)
    elif (len(errors)>0):
        logging.error(bcolors.FAILED +bcolors.BOLD+  '[FAILED] '+regdir.replace(f"{VALIDATION_FOLDER}/",'') +bcolors.ENDC)
    
    if (len(errors)==len(files)):
        logging.error('\tno output files found')
    else:
        if len(mismatch)>0:
            logging.critical(bcolors.WARNING+bcolors.BOLD+'\twith following mismatched files:'+ bcolors.ENDC)
            for f in mismatch:
                fn=f.replace('./','')
                logging.critical(bcolors.BOLD+'\t\t'+fn + bcolors.ENDC)
                f1=regdir+'/'+f
                f2=outdir+'/'+f
                fn=f.replace('./','')
                diff_output,abs_max,rel_max,i_max=approximate_diff(f1,f2)
                if f.endswith('png'):
                    logging.warning('\t\t\timage files are not identical')
                elif abs_max==0 and rel_max==0:
                    logging.critical('\t\t\tno numerical difference detected')
                else:
                    logging.critical('\t\t\tnumber of differences      : {:d}'.format(i_max))
                    logging.critical('\t\t\tmaximum absolute difference: {:.5f}'.format(abs_max))
                    logging.critical('\t\t\tmaximum relative difference: {:.5f}%'.format(rel_max*100.0))
        if len(errors)>0 and len(match)>1:
            logging.error(bcolors.FAILED+bcolors.BOLD+'\tmissing files:'+ bcolors.ENDC)
            for f in errors:
                fn=f.replace('./','')
                logging.error(bcolors.BOLD+'\t\t'+fn + bcolors.ENDC)
        

def main():
    print("Validation starts.\n")
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.mkdir(OUTPUT_FOLDER)

    # iterate through each example
    for example_dir in listdir_nohidden():
        logging.info(f"Testing example - {example_dir} ...")
        out_dir = example_dir.replace(NOTEBOOKS_FOLDER, OUTPUT_FOLDER)
        os.mkdir(out_dir)
        val_dir = example_dir.replace(NOTEBOOKS_FOLDER, VALIDATION_FOLDER)
    
        # iterate through each script under current example
        for script in listpy_nohidden_notlib(example_dir):
            # create new subfolder for current script
            script_name = script.split("/")[-1]
            out_subdir = f"{out_dir}{script_name[:-3]}"
            os.mkdir(out_subdir)
            logging.info(f"script: {script}")
            # run the script
            logging.info(f"\tsaving outputs to folder {out_subdir}")
            res = subprocess.run([
                PYTHON, 
                script, 
                "--no-show-plot", 
                "--save-plot", 
                "--show-summary",
                "--output-dir",
                out_subdir,
            ], stdout=subprocess.PIPE, text=True)
            # write log file
            with open(f"{out_subdir}/log.txt", "w") as log_file:
                log_file.write(res.stdout)
            # compare output with validation standards
            val_subdir = f"{val_dir}{script_name[:-3]}"
            logging.info(f"\tcomparing outputs with folder {val_subdir}")
            files = list(listfiles_nohidden(out_subdir))
            files = [f.split("/")[-1] for f in files]
            logging.info(f"\t\twith following files: {files}")
            match, mismatch, errors = filecmp.cmpfiles(out_subdir, val_subdir, files)
            analyse_cmp_res(match, mismatch, errors, out_subdir, val_subdir, files)

        # remove empty folders (for the examples with no script)
        try:
            os.rmdir(out_dir)
        except:
            pass

    print("\nAll done - check ./debug.log file for full report.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    main()
