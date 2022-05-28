from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

labels = ["animals", "food", "locations", "names"]
hypothesis_template = 'This text is about {}.'
inputs = ["golden retriever", "london", "hot dog stand", "joe's restaurant"]

for i in range(len(inputs)):
    prediction = classifier(inputs[i], labels, hypothesis_template=hypothesis_template, multi_label=True)
    print(prediction)
