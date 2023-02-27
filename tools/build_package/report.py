"""Generate API compliance report 

"""

import run_examples
import criteria

def _metadata_compliance(prob_class, prob_class_name, prob_path, parent_mod):
    # preparation
    names_in_folder, paths_in_folder = run_examples.get_folder_content(prob_path)
    prob_name = prob_path.split("/")[-1]

    # checking
    criteria._check_folder_file_names(prob_name, prob_path, names_in_folder)
    criteria._check_required_files(prob_path, names_in_folder)
    criteria._check_licence_nonempty(prob_path)
    criteria._check_init_all(parent_mod, prob_name, prob_class_name)
    criteria._check_subclass(prob_class, prob_class_name)
    criteria._check_metadata(prob_class, prob_class_name)

def _example_compliance(all_results):
    for attr_check in criteria.attributes_to_check:
        attr_key, attr_str, required, to_check = attr_check
        obj = all_results[attr_key]
        obj_str = f"{all_results['prob_instance_str']}.{attr_str}"
        if isinstance(obj, Exception):
            raise obj
        if obj is None and required:
            raise NotImplementedError(f"{obj_str} is required but you haven't implemented it")
        if obj is not None:
            for check_func in to_check:
                check_func(all_results, obj, obj_str)

def api_compliance_report(print=True):
    return dict()
