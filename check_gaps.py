'''
v1.0.0 -- 22 Oct 2023 -- Yori Jonkman
Checks if there are gaps in the enumeration of trigger entities in supplied
TriggerNER data.
'''
import sys

def detect_enum_gap(sentence):
	missing_enum = None
	entities = list(set([word.split()[-1] for word in sentence]))
	for enum in range(0, 17):
		t = f"T-{enum}"
		if t not in entities:
			missing_enum = enum
		elif missing_enum != None:
			print(f"Trigger enumeration gap from T-{missing_enum} to T-{enum} detected in:\n{sentence}\n")
			return True
	return False

try:
	filepath = sys.argv[1]
	with open(filepath, "r") as file:
		sentence = []
		checked = 0
		gaps = 0
		for line in file:
			if not line.strip():
				if detect_enum_gap(sentence):
					gaps += 1
				checked += 1
				sentence = []
				continue
			sentence.append(line)
		if sentence:
			if detect_enum_gap(sentence):
				gaps += 1
			checked += 1
		print(f"Sentences checked: {checked}\n Enumeration gaps: {gaps}")
	print(f"Operation complete.")

except Exception as e:
	print(e)
