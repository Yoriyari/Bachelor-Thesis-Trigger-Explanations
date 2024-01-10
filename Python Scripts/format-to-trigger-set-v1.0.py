'''
v1.0.0 -- 01 Jan 2024 -- Yori Jonkman
Takes a labelled JSONL file from Doccano as input.
Converts the given file to the JSON label format read by TriggerNER.
Which is to say, from:

{"text": "Man in Belgium.", "label": [[4, 7, "TRIGGER"], [7, 14, "LOCATION"]]}

to:

{"text": "Man in Belgium .", "label": "O O B-LOC O", "explanation": "O T-0 O O"}

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
		os.mkdir(f"{location}explanation_{filename}")
		for subfile in os.listdir(filepath):
			reformat_file(f"{filepath}\\{subfile}", location=f"{location}explanation_{filename}")
	elif re.search("\.jsonl$", filename):		# JSONL files
		new_json = []
		with open(filepath, "r", encoding="utf8") as old_file:
			first_line = old_file.readline()
			old_dict = json.loads(first_line)
			if "label" not in old_dict:
				return
			if "text" not in old_dict:
				return
			for old_line in old_file:
				old_dict = json.loads(old_line)							# Read old jsonl file & prepare variables
				old_text = old_dict["text"]
				old_labels = old_dict["label"]
				tokens = list(tokenize.word_tokenize(old_text))
				spans = list(tokenize.WhitespaceTokenizer().span_tokenize(old_text))
				len_spans = len(spans)
				named_tags = ["O"] * len(tokens)
				trigger_tags = ["O"] * len(tokens)
				trigger_enum = 0
				for label in old_labels:
					new_label = copy.deepcopy(label)
					for i, (span_start, span_end) in enumerate(spans):	# Subtract preceding whitespaces from label indices
						if i == 0:
							continue
						prev_span_end = spans[i-1][1]
						space_width = span_start - prev_span_end
						if label[1] >= span_start:
							new_label[1] -= space_width
						else:
							break
						if label[0] >= span_start:
							new_label[0] -= space_width
					for i, token in enumerate(tokens):					# Subtract preceding tokens from label indices & label tokens in "B-LOC O T-0" format
						len_token = len(token)
						if new_label[2] == "TRIGGER":
							if new_label[0] < len_token:
								if new_label[1] >= len_token:
									trigger_tags[i] = f"T-{trigger_enum}"
								else:
									break
						else:
							if new_label[0] < len_token:
								if new_label[1] >= len_token:
									if new_label[0] == 0:
										named_tags[i] = "B-LOC"
									else:
										named_tags[i] = "I-LOC"
								else:
									break
						new_label[0] -= len_token
						new_label[1] -= len_token
					if new_label[2] == "TRIGGER":						# Token enumeration
						trigger_enum += 1
				new_text = " ".join(tokens)								# Add converted entry to list
				new_labels = " ".join(named_tags)
				explanation = " ".join(trigger_tags)
				new_dict = {"text": new_text, "label": new_labels, "explanation": explanation}
				new_json.append(new_dict)
		filename = filename[:-1]										# Write fully converted json to new file
		with open(f"{location}explanation_{filename}", "w") as new_file:
			new_file.write("")
		with open(f"{location}explanation_{filename}", "a", encoding="utf8") as new_file:
			json.dump(new_json, new_file, indent=2)

try:
	filepath = sys.argv[1]
	reformat_file(filepath)
	print(f"Operation complete.")
except Exception as e:
	print(e)
