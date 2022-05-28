from transformers import pipeline

labels = ["animals", "food", "locations", "person", "education", "entertainment", "dance", "politics", "economics", "science", "history", "transportation", "tools", "plants", "beliefs", "nutrition", "culture", "countries", "video games", "sports", "technology", "music"]

print("Setting up classifier (This takes ~ 1 min but only needs to be setup once if kept running)")
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

repeat = "y"
while(repeat == "y"):
    word = input("Word/Phrase to be classified: ").lower()

    print("Classifying word (This takes ~ 10 sec)\n")
    prediction = classifier(word, labels, multi_label=True)


    labels_ordered = prediction.get("labels")
    scores = prediction.get("scores")
    print("{: <15} Score".format("Label"))
    for i in range(5):
        print("{: <15} {:.4f}".format(labels_ordered[i], scores[i]))

    repeat = input("Would you like to classify another word? [Y/N]: ").lower()