'''
v1.0.0 -- 12 Oct 2023 -- Yori Jonkman
Takes a labelled JSONL file from IDRISI-RE as input.
Converts the given JSONL to the label format read by Doccano.
Which is to say, from:

"location_mentions": [{"text": "EU", "type": "LOCATION", "start_offset": 0, "end_offset": 2}]

to:

"label": [[0, 2, "LOCATION"]]
'''
import json
import sys

try:
	filepath = sys.argv[1]
	filename = filepath
	if "\\" in filepath:
		filename = filepath[filepath.rindex('\\')+1:]

	with open(f"re_{filename}", "w") as new_file:
		new_file.write("")
	with open(f"re_{filename}", "a") as new_file:
		with open(filepath, "r") as old_file:
			for old_line in old_file:
				old_dict = json.loads(old_line)
				labels = [[mention["start_offset"], mention["end_offset"], "LOCATION"] for mention in old_dict["location_mentions"]]
				new_dict = {"text": old_dict["text"], "label": labels}
				new_line = json.dumps(new_dict) + "\n"
				new_file.write(new_line)
	print(f"Operation complete.")

except Exception as e:
	print(e)
