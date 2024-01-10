'''
v1.0.0 -- 22 Oct 2023 -- Yori Jonkman
Checks for true/false positive/negatives for trigger labels and unmarked tokens
between manually marked data and corresponding labelled TriggerNER data.
'''
import sys

try:
	file_manual = sys.argv[1]
	file_prelabelled = sys.argv[2]
	lines_m = []
	with open(file_manual, "r") as file:
		lines_m = file.readlines()
	lines_p = []
	with open(file_prelabelled, "r") as file:
		lines_p = file.readlines()
	sentence = []
	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0
	for i in range(len(lines_m)):
		line_m = lines_m[i].strip()
		line_p = lines_p[i].strip()
		if not line_m or not line_p:
			continue
		label_m = line_m.split()[-1]
		label_p = line_p.split()[-1]
		if label_p == "O":
			if label_m == "O":
				true_negatives += 1
			else:
				false_positives += 1
		elif label_p.startswith("T-"):
			if label_m.startswith("T-"):
				true_positives += 1
			else:
				false_negatives += 1
	print(f" True positives: {true_positives}\nFalse positives: {false_positives}\n True negatives: {true_negatives}\nFalse negatives: {false_negatives}")
	print(f"Operation complete.")

except Exception as e:
	print(e)
