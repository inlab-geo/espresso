from shutil import copytree

OUTPUT_FOLDER = "utils/validation/_output"
VALIDATION_FOLDER = "utils/validation/_validation_output"

if __name__ == "__main__":
    copytree(OUTPUT_FOLDER, VALIDATION_FOLDER, dirs_exist_ok=True)
