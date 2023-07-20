"""Generate API compliance report 

Usage:
- To generate raw report, raw_compliance_report(problems_to_check=None, pre_build=True)
- To generate report, compliance_report(problems_to_check=None, pre_build=True)
- To print report, print_compliance_report(report)
"""
import dataclasses
import typing

import run_examples
import criteria
import _utils


@dataclasses.dataclass
class ProblemRawReport:
    problem_name: str
    problem_class_name: str
    metadata: typing.Union[bool, Exception]
    attributes_required: dict[str, list[typing.Optional[bool]]]
    attributes_optional: dict[str, list[typing.Optional[bool]]]
    attributes_additional: list[str]

    def __init__(self, results: run_examples.ResultsFromProblem):
        self.problem_name = results.problem_name
        self.problem_class_name = results.problem_class_str
        self.metadata = self._collect_metadata_report(results)
        if self.metadata_ok():
            self.attributes_required = dict()
            self.attributes_optional = dict()
            self.attributes_additional = list()
            self._collect_attributes_report(results)

    def metadata_ok(self) -> bool:
        return self.metadata == True

    def _collect_metadata_report(self, results: run_examples.ResultsFromProblem):
        if isinstance(results.parent_module, Exception):
            return results.parent_module
        else:
            try:
                criteria.criteria_for_problem(
                    results.problem_class,
                    results.problem_class_str,
                    results.problem_path,
                    results.parent_module,
                )
            except Exception as e:
                return e
            else:
                return True

    def _collect_attributes_report(self, results: run_examples.ResultsFromProblem):
        for results_i in results.problem_results_generator:
            self._collect_compliance_info(results_i)
            self._collect_additional_info(results_i)

    def _collect_compliance_info(self, results_i: run_examples.ResultsFromExample):
        _has_init_error = isinstance(results_i.prob_instance, Exception)
        if _has_init_error:
            _init_error = results_i.prob_instance
        for attr_check in criteria.attributes_to_check:
            attr_key, attr_str, required, to_check = attr_check
            _report_key = "attributes_required" if required else "attributes_optional"
            _attr_name = attr_str.split("(")[0]
            if _attr_name not in getattr(self, _report_key):
                getattr(self, _report_key)[_attr_name] = []
            _to_update = getattr(self, _report_key)[_attr_name]
            if _has_init_error:
                _to_update.append(_init_error)
                continue
            obj = getattr(results_i, attr_key)
            obj_str = f"{results_i.prob_instance_str}.{attr_str}"
            if isinstance(obj, Exception) or obj is None:
                _to_update.append(obj)
            else:
                try:
                    for check_func in to_check:
                        check_func(results_i, obj, obj_str)
                except Exception as e:
                    _to_update.append(e)
                else:
                    _to_update.append(True)

    def _get_standard_attr(self, problem_instance):
        standard_attr = set(self.attributes_required.keys())
        standard_attr.update(self.attributes_optional.keys())
        standard_attr.update(problem_instance.__abstract_metadata_keys__)
        standard_attr.update({"metadata", "params", "example_number"})
        return standard_attr

    def _collect_additional_info(self, results_i: run_examples.ResultsFromExample):
        p = results_i.prob_instance
        if isinstance(p, Exception):
            return
        dir_p = [attr for attr in dir(p) if not attr.startswith("_")]
        _standard_attr = self._get_standard_attr(p)
        _additional_attr = {a for a in dir_p if a not in _standard_attr}
        self.attributes_additional = _additional_attr


def raw_compliance_report(problems_to_check=None, pre_build=True, timeout=None) \
    -> dict[str, ProblemRawReport]:
    """Run all problems and generate a raw compliance report

    A typical raw report looks like:
    {
        'FmmTomography': {
            'metadata': True,
            'attributes_required': {
                'model_size': [True], 'data_size': [True], 'good_model': [True], 'starting_model': [True], 'data': [True], 'forward': [True]
            },
            'attributes_optional': {
                'description': [None], 'covariance_matrix': [None], 'inverse_covariance_matrix': [None], 'jacobian': [True], 'forward': [True, True], 'plot_model': [True], 'plot_data': [None], 'misfit': [None], 'log_likelihood': [None], 'log_prior': [None]
            },
            'attributes_additional': {
                'exe_fm2dss', 'clean_tmp_files', 'tmp_paths', 'tmp_files'
            }
        }
    }
    """
    raw_report = dict()
    problems = _utils.problems_to_run(problems_to_check)
    results = run_examples.run_problems(problems, pre_build=pre_build, timeout=timeout)
    for res in results:
        raw_report[res.problem_class_str] = ProblemRawReport(res)
    return raw_report


@dataclasses.dataclass
class ProblemReport:
    problem_name: str
    problem_class_name: str
    metadata: typing.Union[str, Exception]
    required: dict[str, typing.Union[str, Exception]]
    required_count: dict[str, int]
    optional: dict[str, typing.Union[str, Exception]]
    optional_count: dict[str, int]
    additional: list[str]
    additional_count: int
    api_compliance: bool

    def __init__(self, raw_report: ProblemRawReport):
        self.problem_name = raw_report.problem_name
        self.problem_class_name = raw_report.problem_class_name
        self.metadata = self._collect_metadata_report(raw_report)
        if self.metadata_ok():
            self._collect_required_attr_report(raw_report)
            self._collect_optional_attr_report(raw_report)
            self._collect_additional_attr_report(raw_report)
            self._analyse_compliance()
        else:
            self.api_compliance = self.metadata
            self.required = None
            self.optional = None
            self.required_count = None
            self.optional_count = None
            self.additional = None
            self.additional_count = None

    def metadata_ok(self) -> bool:
        return self.metadata == "OK"
    
    def _collect_metadata_report(self, raw_report: ProblemRawReport):
        return "OK" if raw_report.metadata_ok() else raw_report.metadata

    @staticmethod
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
        return {"report": _new_report, "count": _new_count}

    def _collect_required_attr_report(self, raw_report: ProblemRawReport):
        res = self._analyse_report_dict(raw_report.attributes_required)
        self.required = res["report"]
        self.required_count = res["count"]

    def _collect_optional_attr_report(self, raw_report: ProblemRawReport):
        res = self._analyse_report_dict(raw_report.attributes_optional)
        self.optional = res["report"]
        self.optional_count = res["count"]
    
    def _collect_additional_attr_report(self, raw_report: ProblemRawReport):
        self.additional = raw_report.attributes_additional
        self.additional_count = len(raw_report.attributes_additional)
    
    def _analyse_compliance(self):
        _metadata_ok = self.metadata_ok()
        _required_ok = \
            self.required_count["implemented"] == self.required_count["total"]
        _optional_ok = self.optional_count["error"] == 0
        self.api_compliance = _metadata_ok and _required_ok and _optional_ok


def compliance_report(problems_to_check=None, pre_build=True, timeout=_utils.DEFAULT_TIMEOUT) \
    -> dict[str, ProblemReport]:
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
    report = dict()
    for prob_name, prob_raw_report in raw_report.items():
        report[prob_name] = ProblemReport(prob_raw_report)
    return report


# from SO: https://stackoverflow.com/a/287944
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def cformat(style, content):
    return f"{style}{content}{bcolors.ENDC}"


def pprint_compliance_report(report: dict[str, ProblemReport]):
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

    FmmTomography (fmm_tomography) is API-compliance. Cheers!
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
        if r.metadata_ok():
            _metadata += cformat(bcolors.OKGREEN, "OK")
        else:
            _metadata += cformat(bcolors.FAIL, r.metadata)
        print(_metadata)
        #### check possibility to continue
        if not r.metadata_ok():
            continue
        #### required
        _required_title = cformat(bcolors.UNDERLINE, "Required attributes") + ": "
        _required_count = r.required_count
        _required_title += (
            f"{_required_count['implemented']}/{_required_count['implemented']} "
            f"implemented, {_required_count['error']} errors, "
            f"{_required_count['not_implemented']} not implemented yet"
        )
        print(_required_title)
        for attr_name, attr_res in r.required.items():
            if attr_res == "OK":
                attr_res_str = cformat(bcolors.OKGREEN, "OK")
            else:
                attr_res_str = cformat(bcolors.FAIL, attr_res.__repr__())
            print(f"\t.{attr_name}\t- {attr_res_str}")
        #### optional
        _optional_title = cformat(bcolors.UNDERLINE, "Optional attributes") + ": "
        _optional_count = r.optional_count
        _optional_title += (
            f"{_optional_count['implemented']}/{_optional_count['total']} "
            f"implemented, {_optional_count['error']} errors, "
            f"{_optional_count['not_implemented']} not implemented yet"
        )
        print(_optional_title)
        for attr_name, attr_res in r.optional.items():
            if attr_res == "OK":
                attr_res_str = cformat(bcolors.OKGREEN, "OK")
            elif isinstance(attr_res, Exception):
                attr_res_str = cformat(bcolors.FAIL, attr_res.__repr__())
            else:
                attr_res_str = cformat(bcolors.WARNING, attr_res)
            print(f"\t.{attr_name}\t- {attr_res_str}")
        #### additional
        _additional_title = cformat(bcolors.UNDERLINE, "Additional attributes") + ": "
        _additional_title += f"{r.additional_count} detected"
        print(_additional_title)
        for attr_name in r.additional:
            print(f"\t.{attr_name}")
        #### sum up
        _api_compliance = r.api_compliance
        if _api_compliance:
            print(cformat(bcolors.OKCYAN, f"\n{prob} ({r.problem_name}) is API-compliant. Cheers!"))
        else:
            print(cformat(bcolors.FAIL, f"\n{prob} ({r.problem_name}) is not API-compliant."))


def capability_report(problems_to_check=None, timeout=_utils.DEFAULT_TIMEOUT_SHORT):
    # those with TimeoutError will be approximated to be OK
    _compliance_report = compliance_report(problems_to_check, True, timeout)
    _capability_report = dict()
    for prob, res in _compliance_report.items():
        _new_report = dict()
        if not res.metadata_ok():
            raise RuntimeError(res)
        for attr, attr_res in res.required.items():
            _new_report[attr] = int(
                attr_res == "OK" or isinstance(attr_res, TimeoutError)
            )
        for attr, attr_res in res.optional.items():
            _new_report[attr] = int(
                attr_res == "OK" or isinstance(attr_res, TimeoutError)
            )
        for additional in res.additional:
            _new_report[additional] = 1
        _capability_report[prob] = _new_report
    return _capability_report
