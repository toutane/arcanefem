import json

def find_key(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            # Recurse if value is a dictionary or list
            result = find_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                result = find_key(item, target_key)
                if result is not None:
                    return result
            elif isinstance(item, str):
                if item == target_key and len(data) >= idx + 2:
                    return data[idx + 1];


file_path="output/listing/time_stats.json"

with open(file_path, 'r') as file:
    content = file.read()

obj = json.loads(content)

formats = ["Legacy",
           "Coo",
           "CooSort",
           "Coo_Gpu",
           "CooSort_Gpu",
           "Csr",
           "Csr_Gpu",
           "CsrNodeWise",
           "CsrBuildLess"]

output = {}

for format in formats:
    metric_name = "AssembleBilinearOperator_" + format
    res = find_key(obj, metric_name)

    tmp = {} 
    if res is not None:
        tmp["AssembleBilinearOperator"] = res
        # tmp["AssembleBinilearOperator"] = {"Local": res["Local"], "Cumulative": res["Cumulative"]}
        # res = find_key(res, "BuildMatrix")
        # if res is not None:
        #     tmp["BuildMatrix"] = {"Local": res["Local"], "Cumulative": res["Cumulative"]}

    output[format] = tmp

print(json.dumps(output, indent = 4))

