import spacy 
import json

nlp = spacy.load("en_core_web_lg")

with open('Corona2.json', 'r') as f:
    data = json.load(f)

print(data['examples'][0]['annotations'][0])

training_data = []
for example in data['examples']:
  temp_dict = {}
  temp_dict['text'] = example['content']
  temp_dict['entities'] = []
  for annotation in example['annotations']:
    start = annotation['start']
    end = annotation['end']
    label = annotation['tag_name'].upper()
    temp_dict['entities'].append((start, end, label))
  training_data.append(temp_dict)

print(training_data[0]['text'])

print(training_data[0]['entities'])

print(training_data[0]['text'][360:371])





