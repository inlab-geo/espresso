"""Generate API compliance report 

"""

import run_examples
import criteria


def _init_attr_report():
    _report = {
        "required": {},
        "optional": {},
        "additional": {},
    }
    for attr_check in criteria.attributes_to_check:
        _, attr_str, required, _ = attr_check
        _report_key = "required" if required else "optional"
        _attr_key_name = attr_str.split("(")[0]
        _report[_report_key][_attr_key_name] = []
    return _report

def _example_compliance(all_results, _report):
    for attr_check in criteria.attributes_to_check:
        attr_key, attr_str, required, to_check = attr_check
        obj = all_results[attr_key]
        obj_str = f"{all_results['prob_instance_str']}.{attr_str}"
        _report_key = "required" if required else "optional"
        _attr_key_name = attr_str.split("(")[0]
        _to_update = _report[_report_key][_attr_key_name]
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

def raw_api_compliance_report(problems_to_check=None, pre_build=True):
    report = dict()
    problems = run_examples.problems_to_run(problems_specified=problems_to_check)
    results = run_examples.run_problems(problems, pre_build=pre_build)
    for res in results:
        _report_for_problem = dict()
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
            _report_for_problem["attributes"] = _init_attr_report()
            for prob_out_i in res["problem results generator"]:
                _example_compliance(prob_out_i, _report_for_problem["attributes"])
        report[res["problem class str"]] = _report_for_problem
    return report
