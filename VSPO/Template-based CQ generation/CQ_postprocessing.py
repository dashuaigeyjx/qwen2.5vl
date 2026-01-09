import os
import json

batchoutput_directory = "CQ_Batchoutput"
jsonl_files = [file for file in os.listdir(batchoutput_directory) if file.endswith(".jsonl")]

for jsonl_file in jsonl_files:
    loaded_data = []
    file_path = os.path.join(batchoutput_directory, jsonl_file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            temp_data = json.loads(line)
            if "http://" in temp_data["response"]["body"]["choices"][0]["message"]["content"]: CQ_list = temp_data["response"]["body"]["choices"][0]["message"]["content"].split("// ")
            else: CQ_list = temp_data["response"]["body"]["choices"][0]["message"]["content"].split("//")
            temp_list = []
            for CQ in CQ_list:
                CQ = CQ.strip()
                if CQ == "": continue
                if CQ[:2]=="\n": CQ = CQ[2:]
                if CQ[0]=="-": CQ = CQ[1:]
                temp_list.append(CQ.strip())
            temp_out = {"axiom": temp_data["custom_id"].split("_",1)[1], "CQ": temp_list}
            loaded_data.append(temp_out)
    # Save the loaded data to a new jsonl file
    output_file_path = os.path.join("Generated CQ", f"{jsonl_file}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in loaded_data:
            json_string = json.dumps(item, ensure_ascii=False)
            f.write(json_string + '\n')
    print(f"Loaded {len(loaded_data)} records from {len(jsonl_files)} JSONL files.")