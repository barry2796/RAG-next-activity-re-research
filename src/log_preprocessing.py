import csv
import os
import random
import re


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    with open(file_path, 'r') as file:
        return file.read()


def extract_traces(log_content):
    pattern_traces = re.findall(r"<trace>(.*?)</trace>", log_content, re.DOTALL)
    traces_result = []
    c = 1
    for single_trace in pattern_traces:
        events = re.findall(r"<event>(.*?)</event>", single_trace, re.DOTALL)
        if len(events) >= 2:
            traces_result.append(single_trace)
            print(f"Processed trace: {c}")
            c += 1
    print(f"Total traces: {c}")
    return traces_result


def build_prefixes(traces, base=1, gap=3):
    prefixes = []
    for trace in traces:
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        if len(events) <= 2:
            continue
        else:
            for i in range(base, len(events), gap):
                i += 1
                if i <= len(events):
                    prefix = events[:i]
                    prefixes.append(''.join(prefix))
    return prefixes


def process_prefixes(traces, log_name=None):
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"\s*/>', re.DOTALL)
    lifecycle_transition_pattern = re.compile(r'<string key="lifecycle:transition" value="(.*?)"\s*/>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"\s*/>')
    seen_prefixes = []
    results = {}
    keys = {}
    activities = set()
    is_bpic_special = log_name and any(bpic_log in log_name.lower() for bpic_log in ['bpic_2013'])
    
    for i in range(0, len(traces)):
        trace = traces[i]
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        cl_list = []
        for event in events:
            concept_name_match = concept_name_pattern.search(event)
            lifecycle_transition_match = lifecycle_transition_pattern.search(event)
            
            if concept_name_match:
                if is_bpic_special:  # For BPIC logs, include more attributes for uniqueness
                    event_name = concept_name_match.group(1)
                    resource_pattern = re.compile(r'<string key="org:resource" value="(.*?)"\s*/>')
                    role_pattern = re.compile(r'<string key="org:role" value="(.*?)"\s*/>')
        
                    resource_match = resource_pattern.search(event)
                    role_match = role_pattern.search(event)
                    
                    event_identifier = event_name
                    if lifecycle_transition_match:
                        event_identifier += f"+{lifecycle_transition_match.group(1)}"
                    if resource_match:
                        event_identifier += f"+{resource_match.group(1)}"
                    if role_match:
                        event_identifier += f"+{role_match.group(1)}"
                    
                    cl_list.append(event_identifier)
                    activities.add(event_identifier)
                else:
                    if lifecycle_transition_match:
                        if lifecycle_transition_match.group(1).lower() == "complete":
                            event_name = concept_name_match.group(1)
                            cl_list.append(event_name)
                            activities.add(event_name)
                    elif not lifecycle_transition_match:
                        event_name = concept_name_match.group(1)
                        cl_list.append(event_name)
                        activities.add(event_name)
        if not cl_list:
            continue
        cl_list_string = ','.join(cl_list)
        if cl_list_string in seen_prefixes:
            continue
        else:
            seen_prefixes.append(cl_list_string)
        last_evt = cl_list[-1]
        attr_vals = {}
        for evt in events:
            for attribute_match in attribute_pattern.findall(evt):
                key, value = attribute_match
                if key != "lifecycle:transition" and key != "concept:name":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attr_vals[key_initial] = value
                    keys[key_initial] = key

        if cl_list_string not in results:
            results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
        print(f"Processed prefix {i}/{len(traces)}")
    print(f"Total unique prefixes: {len(results)}")
    return results, keys, activities


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set


def generate_csv_from_test_set(test_set, test_path, size=300, log_name=None):
    test_set = build_prefixes(test_set)
    test_set, attr_keys, act_list = process_prefixes(test_set, log_name)
    if size > len(test_set):
        size = len(test_set)
    test_set = dict(random.sample(list(test_set.items()), size))
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for prefix, prediction in test_set.items():
            attributes = prediction.split('|')[0].strip()
            next_activity = prediction.split('| Next activity: ')[1].strip()
            csvwriter.writerow([f'{prefix} - Values: {attributes}', next_activity])
