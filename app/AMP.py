import itertools
from datasets import Dataset
import json
import joblib

def preprocess_data(filexaif):
    idents = []
    idents_comb = []
    propositions = {}
    data = {'text': [], 'text2': []}

    for node in filexaif['nodes']:
        if node['type'] == 'I':
            propositions[node['nodeID']] = node['text']
            idents.append(node['nodeID'])

    for p in itertools.combinations(idents, 2):
        idents_comb.append(p)
        data['text'].append(propositions[p[0]])
        data['text2'].append(propositions[p[1]])

    final_data = Dataset.from_dict(data)

    return final_data, idents_comb, propositions

def data_engineering(sentences):

 text = sentences["text"]
 text2 = sentences["text2"]

 X = [t1 + " " + t2 for t1, t2 in zip(text, text2)]
 # vectorizer = TfidfVectorizer(
 #     stop_words="english", max_features=None)
 # X = vectorizer.fit_transform(X)

 return X


# def tokenize_sequence(samples):
#     return TOKENIZER(samples["text"], samples["text2"], padding="max_length", truncation=True)
#
#
# def make_predictions(trainer, tknz_data):
#     predicted_logprobs = trainer.predict(tknz_data)
#     predicted_labels = np.argmax(predicted_logprobs.predictions, axis=-1)
#
#     return predicted_labels


def output_xaif(idents, labels, fileaif):
    newnodeId = 90000
    newedgeId = 80000
    for i in range(len(labels)):
        lb = labels[i]


        if lb == 0:
            continue

        elif lb == 'RA':
            # Add the RA node
            fileaif["AIF"]["nodes"].append({"nodeID": newnodeId, "text": "Default Inference", "type": "RA", "timestamp": "", "scheme": "Default Inference", "schemeID": "72"})

            # Add the edges from ident[0] to RA and from RA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 'CA':
            # Add the CA node
            fileaif["AIF"]["nodes"].append({"nodeID": newnodeId, "text": "Default Conflict", "type": "CA", "timestamp": "", "scheme": "Default Conflict", 'schemeID': "71"})

            # Add the edges from ident[0] to MA and from MA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1



    return fileaif


def relation_identification(xaif):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    # genereate pair of propositions and their ids from xaif data
    dataset, ids, propositions = preprocess_data(xaif['AIF'])

    # create the Dataset for vectorizing.
    vectorized_data = data_engineering(dataset)

    # call the training model.
    model = joblib.load('app/svm_pipeline.joblib')

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = model.predict(vectorized_data)
    # count_no_rel = (labels == 'CA').sum()
    # print(count_no_rel)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open('../data.json', 'r')
    content = json.load(ff)
    print(content)
    out = relation_identification(content)
    with open("../data_out.json", "w") as outfile:
        json.dump(out, outfile, indent=4)
