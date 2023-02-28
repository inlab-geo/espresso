"""Generate API compliance report 

"""

import run_examples
import criteria


def _init_attr_report():
    _report = {
        "required": dict(),
        "optional": dict(),
        "additional": set(),
    }
    for attr_check in criteria.attributes_to_check:
        _, attr_str, required, _ = attr_check
        _report_key = "required" if required else "optional"
        _attr_key_name = attr_str.split("(")[0]
        _report[_report_key][_attr_key_name] = []
    return _report

def _collect_compliance_info(all_results, report):
    for attr_check in criteria.attributes_to_check:
        attr_key, attr_str, required, to_check = attr_check
        obj = all_results[attr_key]
        obj_str = f"{all_results['prob_instance_str']}.{attr_str}"
        _report_key = "required" if required else "optional"
        _attr_key_name = attr_str.split("(")[0]
        _to_update = report[_report_key][_attr_key_name]
        if isinstance(obj, Exception) or obj is None:
            _to_update.append(obj)
        else:
            try:
                for check_func in to_check:
                    check_func(all_results, obj, obj_str)
            except Exception as e:
                _to_update.append(e)
            else:
                _to_update.append(True)

def _example_standard(example_instance, report):
    _standard_attr = set(report["required"].keys()) 
    _standard_attr.update(report["optional"].keys())
    _standard_attr.update(example_instance.__abstract_metadata_keys__)
    _standard_attr.add("metadata")
    _standard_attr.add("params")
    _standard_attr.add("example_number")
    return _standard_attr

def _collect_additional_attr(all_results, report):
    p = all_results["prob_instance"]
    p_dir = [attr for attr in dir(p) if not attr.startswith("_")]
    _standard_attr = _example_standard(p, report)
    _additional_attr = {a for a in p_dir if a not in _standard_attr}
    report["additional"].update(_additional_attr)

def raw_api_compliance_report(problems_to_check=None, pre_build=True):
    report = dict()
    problems = run_examples.problems_to_run(problems_specified=problems_to_check)
    results = run_examples.run_problems(problems, pre_build=pre_build)
    for res in results:
        _report_for_problem = dict()
        # problem level report
        try:
            criteria.criteria_for_problem(
                res["problem class"],
                res["problem class str"],
                res["problem path"],
                res["parent module"], 
            )
        except Exception as e:
            _report_for_problem["metadata"] = e
        else:
            _report_for_problem["metadata"] = True
            # example level report
            _report_for_problem["attributes"] = _init_attr_report()
            for prob_out_i in res["problem results generator"]:
                # required / optional attributes
                _collect_compliance_info(prob_out_i, _report_for_problem["attributes"])
                # additional attributes
                _collect_additional_attr(prob_out_i, _report_for_problem["attributes"])
        report[res["problem class str"]] = _report_for_problem
    return report
