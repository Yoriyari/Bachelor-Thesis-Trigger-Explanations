'''
v1.0.0 -- 11 Jan 2024 -- Yori Jonkman
Takes a labelled JSONL file from Doccano as input.
Converts the given file to the tokenized JSON label format read by TriggerNER.
Which is to say, from:

{"text": "Man in Belgium.", "location_mentions": [{"text": "Belgium", "type": "LOCATION", "start_offset": 7, "end_offset": 14}]

to:

{"text": "Man in Belgium .", "label": "O O B-LOC O"}

Does not add explanation values to the final result.
Can convert every JSONL file in a folder if given a folder.
'''
import copy, json, sys, os, re
from nltk import tokenize

def reformat_file(filepath, location=""):
	filename = filepath
	if "\\" in filepath:
		filename = filepath[filepath.rindex('\\')+1:]
	if location:
		location += "\\"
	if not re.search("\.[\w\d]+$", filename):	# Folders
		os.mkdir(f"{location}tokenized_{filename}")
		for subfile in os.listdir(filepath):
			reformat_file(f"{filepath}\\{subfile}", location=f"{location}tokenized_{filename}")
	elif re.search("\.jsonl$", filename):		# JSONL files
		new_json = []
		with open(filepath, "r", encoding="utf8") as old_file:
			first_line = old_file.readline()
			old_dict = json.loads(first_line)
			if "text" not in old_dict:
				return
			is_test_data = False
			if "location_mentions" not in old_dict:
				is_test_data = True
			for old_line in old_file:
				old_dict = json.loads(old_line)							# Read old jsonl file & prepare variables
				old_text = old_dict["text"]
				if not is_test_data:
					old_labels = old_dict["location_mentions"]
				tokens = list(tokenize.word_tokenize(old_text))
				spans = list(tokenize.WhitespaceTokenizer().span_tokenize(old_text))
				len_spans = len(spans)
				named_tags = ["O"] * len(tokens)
				if not is_test_data:
					for label in old_labels:
						new_label = copy.deepcopy(label)
						for i, (span_start, span_end) in enumerate(spans):	# Subtract preceding whitespaces from label indices
							if i == 0:
								continue
							prev_span_end = spans[i-1][1]
							space_width = span_start - prev_span_end
							if label["end_offset"] >= span_start:
								new_label["end_offset"] -= space_width
							else:
								break
							if label["start_offset"] >= span_start:
								new_label["start_offset"] -= space_width
						for i, token in enumerate(tokens):					# Subtract preceding tokens from label indices & label tokens in "O B-LOC O" format
							len_token = len(token)
							if new_label["start_offset"] < len_token:
								if new_label["end_offset"] >= len_token:
									if new_label["start_offset"] == 0:
										named_tags[i] = "B-LOC"
									else:
										named_tags[i] = "I-LOC"
								else:
									break
							new_label["start_offset"] -= len_token
							new_label["end_offset"] -= len_token
				new_text = " ".join(tokens)								# Add converted entry to list
				if is_test_data:
					new_dict = {"text": new_text}
				else:
					new_labels = " ".join(named_tags)
					new_dict = {"text": new_text, "label": new_labels}
				new_json.append(new_dict)
		filename = filename[:-1]										# Write fully converted json to new file
		with open(f"{location}tokenized_{filename}", "w") as new_file:
			new_file.write("")
		with open(f"{location}tokenized_{filename}", "a", encoding="utf8") as new_file:
			json.dump(new_json, new_file, indent=2)

try:
	filepath = sys.argv[1]
	reformat_file(filepath)
	print(f"Operation complete.")
except Exception as e:
	print(e)
