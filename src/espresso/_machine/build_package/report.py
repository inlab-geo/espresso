"""Generate API compliance report 

Usage:
- To generate raw report, raw_compliance_report(problems_to_check=None, pre_build=True)
- To generate report, compliance_report(problems_to_check=None, pre_build=True)
- To print report, print_compliance_report(report)
"""
import run_examples
import criteria
import _utils


def _init_attr_report():
    import criteria
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
    import criteria
    _has_init_error = isinstance(all_results["prob_instance"], Exception)
    if _has_init_error:
        _init_error = all_results["prob_instance"]
    for attr_check in criteria.attributes_to_check:
        attr_key, attr_str, required, to_check = attr_check
        _report_key = "required" if required else "optional"
        _attr_key_name = attr_str.split("(")[0]
        _to_update = report[_report_key][_attr_key_name]
        if _has_init_error:
            _to_update.append(_init_error)
            continue
        obj = all_results[attr_key]
        obj_str = f"{all_results['prob_instance_str']}.{attr_str}"
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
    if isinstance(p, Exception):
        return
    p_dir = [attr for attr in dir(p) if not attr.startswith("_")]
    _standard_attr = _example_standard(p, report)
    _additional_attr = {a for a in p_dir if a not in _standard_attr}
    report["additional"].update(_additional_attr)

def raw_compliance_report(problems_to_check=None, pre_build=True, timeout=None):
    """Run all problems and generate a raw compliance report
    
    A typical raw report looks like:
    {
        'FmmTomography': {
            'metadata': True, 
            'attributes': {
                'required': {
                    'model_size': [True], 'data_size': [True], 'good_model': [True], 'starting_model': [True], 'data': [True], 'forward': [True]
                }, 
                'optional': {
                    'description': [None], 'covariance_matrix': [None], 'inverse_covariance_matrix': [None], 'jacobian': [True], 'forward': [True, True], 'plot_model': [True], 'plot_data': [None], 'misfit': [None], 'log_likelihood': [None], 'log_prior': [None]
                }, 
                'additional': {
                    'exe_fm2dss', 'clean_tmp_files', 'tmp_paths', 'tmp_files'
                }
            }
        }
    }
    """
    report = dict()
    problems = _utils.problems_to_run(problems_to_check)
    results = run_examples.run_problems(problems, pre_build=pre_build, timeout=timeout)
    for res in results:
        _report_for_problem = dict()
        # problem level report
        if isinstance(res["parent module"], Exception):
            _report_for_problem["metadata"] = res["parent module"]
        else:
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

def _analyse_report_dict(sub_report):
    _new_report = dict()
    _new_count = {
        "implemented": 0,
        "not_implemented": 0,
        "error": 0,
        "total": 0,
    }
    for item_name, item_report in sub_report.items():
        _done = [r for r in item_report if r == True]
        _has_error = [r for r in item_report if isinstance(r, Exception)]
        if len(_done) == len(item_report):
            _new_count["implemented"] += 1
            _new_report[item_name] = "OK"
        elif len(_has_error) > 0:
            _new_count["error"] += 1
            _new_report[item_name] = _has_error[0]
        else:
            _new_count["not_implemented"] += 1
            _new_report[item_name] = "Not implemented"
        _new_count["total"] += 1
    return _new_report, _new_count

def _analyse_compliance(new_report):
    _metadata_ok = new_report["metadata"] == "OK"
    _required_count = new_report["required_count"]
    _required_ok = _required_count["implemented"] == _required_count["total"]
    _optional_ok = new_report["optional_count"]["error"] == 0
    return _metadata_ok and _required_ok and _optional_ok

def compliance_report(problems_to_check=None, pre_build=True, timeout=60):
    """Generate a readable compliance report based on running raw report
    
    A typical compliance report looks like:
    {
        'FmmTomography': {
            'metadata': 'OK', 
            'required': {'model_size': 'OK', 'data_size': 'OK', 'good_model': 'OK', 'starting_model': 'OK', 'data': 'OK', 'forward': 'OK'}, 
            'required_count': {'implemented': 6, 'not_implemented': 0, 'error': 0, 'total': 6}, 
            'optional': {'description': 'Not implemented', 'covariance_matrix': 'Not implemented', 'inverse_covariance_matrix': 'Not implemented', 'jacobian': 'OK', 'forward': 'OK', 'plot_model': 'OK', 'plot_data': 'Not implemented', 'misfit': 'Not implemented', 'log_likelihood': 'Not implemented', 'log_prior': 'Not implemented'}, 
            'optional_count': {'implemented': 3, 'not_implemented': 7, 'error': 0, 'total': 10}, 
            'additional': {'clean_tmp_files', 'tmp_paths', 'exe_fm2dss', 'tmp_files'}, 
            'additional_count': 4, 
            'api_compliance': True,
        }
    }
    """
    raw_report = raw_compliance_report(problems_to_check, pre_build, timeout)
    new_report = dict()
    for prob_name, prob_report in raw_report.items():
        _new_report = dict()
        # metadata
        _metadata = prob_report["metadata"]
        _new_report["metadata"] = "OK" if _metadata == True else _metadata
        # check possibility to continue
        if "attributes" not in prob_report:
            _new_report["api_compliance"] = _metadata
            new_report[prob_name] = _new_report
            continue
        # required
        _res = _analyse_report_dict(prob_report["attributes"]["required"])
        _new_report["required"] = _res[0]
        _new_report["required_count"] = _res[1]
        # optional
        _res = _analyse_report_dict(prob_report["attributes"]["optional"])
        _new_report["optional"] = _res[0]
        _new_report["optional_count"] = _res[1]
        # additional
        _new_report["additional"] = prob_report["attributes"]["additional"]
        _new_report["additional_count"] = len(_new_report["additional"])
        # sum up
        _new_report["api_compliance"] = _analyse_compliance(_new_report)
        new_report[prob_name] = _new_report
    return new_report

# from SO: https://stackoverflow.com/a/287944
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cformat(style, content):
    return f"{style}{content}{bcolors.ENDC}"

def pprint_compliance_report(report):
    """Pretty print a compliance report (i.e. output of compliance_report())
    
    The console output typically looks like:
    ```
    ----------------------------------------
    ESPRESSO Machine - API Compliance Report
    ----------------------------------------

    FmmTomography
    Metadata: OK
    Required attributes: 6/6 implemented, 0 errors, 0 not implemented yet
            .model_size     - OK
            .data_size      - OK
            .good_model     - OK
            .starting_model - OK
            .data   - OK
            .forward        - OK
    Optional attributes: 3/10 implemented, 0 errors, 7 not implemented yet
            .description    - Not implemented
            .covariance_matrix      - Not implemented
            .inverse_covariance_matrix      - Not implemented
            .jacobian       - OK
            .forward        - OK
            .plot_model     - OK
            .plot_data      - Not implemented
            .misfit - Not implemented
            .log_likelihood - Not implemented
            .log_prior      - Not implemented
    Additional attributes: 4 detected
            .clean_tmp_files
            .tmp_paths
            .exe_fm2dss
            .tmp_files

    FmmTomography is API-compliance. Cheers!
    ```
    """
    # title
    _title = "ESPRESSO Machine - API Compliance Report"
    print("-" * len(_title))
    print(cformat(bcolors.HEADER, _title))
    print("-" * len(_title))
    for prob, r in report.items():
        print()
        #### problem name
        print(cformat(bcolors.OKCYAN, cformat(bcolors.BOLD, prob)))
        #### metadata
        _metadata = cformat(bcolors.UNDERLINE, "Metadata") + ": "
        if r["metadata"] == "OK":
            _metadata += cformat(bcolors.OKGREEN, "OK")
        else:
            _metadata += cformat(bcolors.FAIL, r["metadata"])
        print(_metadata)
        #### check possibility to continue
        if "required" not in r:
            continue
        #### required
        _required_title = cformat(bcolors.UNDERLINE, "Required attributes") + ": "
        _required_count = r["required_count"]
        _required_title += (
            f"{_required_count['implemented']}/{_required_count['implemented']} "
            f"implemented, {_required_count['error']} errors, "
            f"{_required_count['not_implemented']} not implemented yet"
        )
        print(_required_title)
        for attr_name, attr_res in r["required"].items():
            if attr_res == "OK": attr_res_str = cformat(bcolors.OKGREEN, "OK")
            else: attr_res_str = cformat(bcolors.FAIL, attr_res.__repr__())
            print(f"\t.{attr_name}\t- {attr_res_str}")
        #### optional
        _optional_title = cformat(bcolors.UNDERLINE, "Optional attributes") + ": "
        _optional_count = r["optional_count"]
        _optional_title += (
            f"{_optional_count['implemented']}/{_optional_count['total']} "
            f"implemented, {_optional_count['error']} errors, "
            f"{_optional_count['not_implemented']} not implemented yet"
        )
        print(_optional_title)
        for attr_name, attr_res in r["optional"].items():
            if attr_res == "OK": attr_res_str = cformat(bcolors.OKGREEN, "OK")
            elif isinstance(attr_res, Exception): 
                attr_res_str = cformat(bcolors.FAIL, attr_res.__repr__())
            else:
                attr_res_str = cformat(bcolors.WARNING, attr_res)
            print(f"\t.{attr_name}\t- {attr_res_str}")
        #### additional
        _additional_title = cformat(bcolors.UNDERLINE, "Additional attributes") + ": "
        _additional_title += f"{r['additional_count']} detected"
        print(_additional_title)
        for attr_name in r["additional"]:
            print(f"\t.{attr_name}")
        #### sum up
        _api_compliance = r["api_compliance"]
        if _api_compliance:
            print(cformat(bcolors.OKCYAN, f"\n{prob} is API-compliance. Cheers!"))
        else:
            print(cformat(bcolors.FAIL, f"\n{prob} is not API-compliant."))

def capability_report(problems_to_check=None, timeout=1):
    # those with TimeoutError will be approximated to be OK
    _compliance_report = compliance_report(problems_to_check, True, timeout)
    _capability_report = dict()
    for prob, res in _compliance_report.items():
        _new_report = dict()
        if "required" not in res:
            raise RuntimeError(res)
        for attr, attr_res in res["required"].items():
            _new_report[attr] = int(attr_res == "OK" or isinstance(attr_res, TimeoutError))
        for attr, attr_res in res["optional"].items():
            _new_report[attr] = int(attr_res == "OK" or isinstance(attr_res, TimeoutError))
        for additional in res["additional"]:
            _new_report[additional] = 1
        _capability_report[prob] = _new_report
    return _capability_report
