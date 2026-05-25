import requests
import time
import re
import os
import tarfile
import docker
import io 
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from collections import defaultdict
import multiprocessing
import subprocess
import shlex
import threading
from pprint import pprint
import numpy as np
from pie.scoring import pie_score
import asyncio
import aiohttp
import json

def parse_submission_result(result: Union[List[Dict[str, Any]], Dict[str, Any]]):
    if isinstance(result, list):
        return [_parse_submission(r) for r in result]
    else:
        return _parse_submission(result)
    
def _parse_submission(result: Dict[str, Any]):
    return PieSingleResult.from_dict(result)

def _parse_single_submission(result: Dict[str, Any]):
    parsed_result = {}
    compilation = result["compile_success"]
    accs = result["accs"]
    mean_acc = np.mean([acc for acc in accs.values()])
    
    parsed_result["compilation"] = compilation
    parsed_result["accs"] = accs
    parsed_result["mean_acc"] = mean_acc
    
    tc2time = {}
    agg_runtime = 0
    tc2success = {} 
    tc2stats = {}
        
    gem5_result = result["gem5"]
    if gem5_result is None or gem5_result == {}:
        agg_runtime = np.inf
    else:
        for tc_no, tc_result in gem5_result.items():
            tc_no = int(tc_no)
            tc2success[tc_no] = tc_result["success"]
            if tc2success[tc_no]: 
                stats =  tc_result["stats"]
                tc2stats[tc_no] = stats
                tc2time[tc_no] = stats["sim_seconds_precise"]
            else: 
                tc2time[tc_no] = np.inf
            agg_runtime += tc2time[tc_no]
    
    parsed_result["agg_runtime"] = agg_runtime
    parsed_result["tc2time"] = tc2time
    parsed_result["tc2success"] = tc2success
    parsed_result["agg_stdev"] = 0 
    parsed_result["tc2stats"] = tc2stats

    parsed_result['score'] = pie_score(compilation, mean_acc, agg_runtime)
    
    return parsed_result

@dataclass
class PieResult: 
    def to_json(self):
        return json.dumps(self.to_dict())
        

@dataclass
class PieSingleResult(PieResult):
    compilation: bool
    accs: Dict[str, float]
    mean_acc: float
    agg_runtime: float
    agg_stdev: float
    tc2time: Dict[str, float]
    tc2success: Dict[str, bool]
    tc2stats: Dict[str, List[float]]
    
    agg_runtime_binary: float = None
    agg_stdev_binary: float = None
    tc2time_binary: Dict[str, float] = None
    tc2success_binary: Dict[str, bool] = None
    tc2stats_binary: Dict[str, List[float]] = None
    score: float = 0.0
    
    @staticmethod
    def from_dict(result: Dict[str, Any]):
        parsed_result = _parse_single_submission(result)
        return PieSingleResult(**parsed_result)
    
    def to_dict(self):
        result = {}
        result["compile_success"] = self.compile_success
        result["accs"] = self.accs
        result["mean_acc"] = self.mean_acc
        result["agg_runtime"] = self.agg_runtime
        result["agg_stdev"] = self.agg_stdev
        result["tc2time"] = self.tc2time
        result["tc2success"] = self.tc2success
        result["tc2stats"] = self.tc2stats
        result["score"] = self.score
        return result
    
    @staticmethod
    def from_json(json_str: str):
        return PieSingleResult.from_dict(json.loads(json_str))

    @staticmethod
    def timeout_result():
        result = PieSingleResult(False, None, 0.0, 0.0, 0.0, None, None, None)
        return result

class Gem5Client:

    def __init__(self):
        self.port = 10086
        self.api_key = '!EWQ2131d'
        self.response_patterns = r"```cpp\s*(.*?)```"

    def parse_response(self, response):
        matches = re.findall(self.response_patterns, response, re.DOTALL)
        if matches:
            return matches[0]
        return ""

    async def test(self, batch, timeout):
        code_list = [sample["completion"] for sample in batch]

        test_sorted = [sample['tests'] for sample in batch]
        problem_ids = [sample['problem_id'] for sample in batch]
        # print(f'code_list {code_list}')
        start_time = time.time()
        try:
            # results = await self.submit_multiple_single_submissions(code_list,
            #                                                 test_sorted,
            #                                                 problem_ids,
            #                                                 "gem5", timeout)
            # return results
            tasks = [
                self.query_single(code, test_case, pid, timeout)
                for code, test_case, pid in zip(code_list, test_sorted, problem_ids)
            ]

            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            print(f"gem5 evaluation failed {e}")
            results = [PieSingleResult.timeout_result() for sample in batch]
            return results
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Gem5 execution time: {execution_time} seconds")

        # results = []
        # for code, test_case, problem_id in zip(code_list, test_sorted, problem_ids):
        #     try:
        #         result = await self.submit_single_submission(code, test_case, problem_id, "gem5", timeout)
        #         results.append(result)
        #     except Exception as e:
        #         print(f"gem5 evaluation failed for one submission: {e}")
        #         results.append(PieSingleResult.timeout_result())
        # return results

        

    async def query_single(self, code, test_case, problem_id, timeout):
        try:
            result = await self.submit_single_submission(code, test_case, problem_id, "gem5", timeout)
            return result
        except Exception as e:
            print(f"gem5 evaluation failed for one submission: {e}")
            return PieSingleResult.timeout_result()

    async def submit_single_submission(self, 
                             code: str, 
                             testcases: List[str], 
                             problem_id: str, 
                             timing_env: str, timeout,
                             override_flags: str = None):
        
        # print(f"Submitting single submission to port {self.port}")
        
        aio_timeout = aiohttp.ClientTimeout(total = timeout)
        async with aiohttp.ClientSession(timeout = aio_timeout) as session:
            async with session.get(f"http://localhost:{self.port}/gem5/single_submission", 
                json={"code": code, 
                              "testcases": testcases, 
                              "problem_id": problem_id, 
                              "timing_env": timing_env, 
                              "override_flags": override_flags, 
                              "api_key": self.api_key}) as resp:
                text = await resp.text()
                return parse_submission_result(json.loads(text))

    async def submit_multiple_single_submissions(self, code_list: List[str],
                                            testcases_list: List[List[str]],
                                            problem_id_list: List[str],
                                            timing_env: str, timeout,
                                            override_flags_list: List[str] = None):
        print(f"Submitting multiple single submissions to port {self.port}")
        if override_flags_list is None:
            override_flags_list = [None] * len(code_list)
        submissions = [{"code": code,
                        "testcases": testcases,
                        "problem_id": problem_id,
                        "override_flags": override_flags} 
                        for code, testcases, problem_id, override_flags 
                        in zip(code_list, testcases_list, problem_id_list, override_flags_list)]
        return await self._get_multiple_single_submissions(submissions, timing_env, timeout)

    async def _get_multiple_single_submissions(self, submissions: List[Dict[str, str]], 
                                        timing_env: str, timeout):
        
        # self.start_stream_thread()
        print(f'Sending data to http://localhost:{self.port}/gem5/multiple_single_submissions')
        aio_timeout = aiohttp.ClientTimeout(total = timeout)
        async with aiohttp.ClientSession(timeout = aio_timeout) as session:
            async with session.post(f"http://localhost:{self.port}/gem5/multiple_single_submissions", json = {"submissions": submissions, 
                    "timing_env": timing_env, 
                    "api_key": self.api_key}) as resp:
                text = await resp.text()
                return parse_submission_result(json.loads(text))
                    
        # req = requests.post(f"http://localhost:{self.port}/gem5/multiple_single_submissions", 
        #                     json={"submissions": submissions, 
        #                         "timing_env": timing_env, 
        #                         "api_key": self.api_key})

        # self.stop_stream_thread()
        # print('=====simulator.py::finish request.get======')
        # return parse_submission_result(req.json())


async def submit_gem5_task(client, data, timeout = 350):
    return await client.test(data, timeout = timeout)