'''
v1.1.0 -- 22 Nov 2023 -- Yori Jonkman
Takes a labelled JSONL file from IDRISI-RE as input.
Converts the given JSONL to the label format read by Doccano.
Which is to say, from:

"location_mentions": [{"text": "EU", "type": "LOCATION", "start_offset": 0, "end_offset": 2}]

to:

"label": [[0, 2, "LOCATION"]]

Changes in v1.1:
Saves a unique copy of each sentence for each respective label.
Can convert every JSONL file in a folder if given a folder.
'''
import json, sys, os, re

def reformat_file(filepath, location=""):
	filename = filepath
	if "\\" in filepath:
		filename = filepath[filepath.rindex('\\')+1:]
	if location:
		location += "\\"
	if not re.search("\.[\w\d]+$", filename):	# Folders
		os.mkdir(f"{location}re_{filename}")
		for subfile in os.listdir(filepath):
			reformat_file(f"{filepath}\\{subfile}", location=f"{location}re_{filename}")
	elif re.search("\.jsonl$", filename):		# JSONL files
		with open(filepath, "r") as old_file:
			first_line = old_file.readline()
			old_dict = json.loads(first_line)
			if "location_mentions" not in old_dict:
				return
			if "text" not in old_dict:
				return
		with open(f"{location}re_{filename}", "w") as new_file:
			new_file.write("")
		with open(f"{location}re_{filename}", "a") as new_file:
			with open(filepath, "r") as old_file:
				for old_line in old_file:
					old_dict = json.loads(old_line)
					labels = []
					for mention in old_dict["location_mentions"]:
						if "start_offset" in mention and "end_offset" in mention:
							labels.append([mention["start_offset"], mention["end_offset"], "LOCATION"])
						elif "startIdx" in mention and "endIdx" in mention:
							labels.append([mention["startIdx"], mention["endIdx"], "LOCATION"])
					if not labels:
						new_dict = {"text": old_dict["text"], "label": []}
						new_line = json.dumps(new_dict) + "\n"
						new_file.write(new_line)
					for label in labels:
						new_dict = {"text": old_dict["text"], "label": [label]}
						new_line = json.dumps(new_dict) + "\n"
						new_file.write(new_line)

try:
	filepath = sys.argv[1]
	reformat_file(filepath)
	print(f"Operation complete.")
except Exception as e:
	print(e)
