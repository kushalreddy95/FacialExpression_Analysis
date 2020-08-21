import os
from textblob import TextBlob
import subprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy
import re



os.chdir("../..")
os.chdir("Voice_Memos/")
users = subprocess.check_output('ls').splitlines()
dict ={}
for user in users:
    os.chdir("{}/".format(user.decode("utf-8")))
    folders =  subprocess.check_output('ls').splitlines()
    if user.decode("utf-8") not in dict.keys():
        dict[user.decode("utf-8")] ={}

    for folder in folders:
        if folder.decode("utf-8").startswith("AR"):
            os.chdir("{}/".format(folder.decode("utf-8")))
            if folder.decode("utf-8") not in dict[user.decode("utf-8")].keys():
                dict[user.decode("utf-8")][folder.decode("utf-8")] = 0;
            if not os.path.isfile('IntSpeechToText.txt'):
                print("File does notexist")
                os.chdir("..")
                continue

            f = open("IntSpeechToText.txt", "r")
            if f.mode == 'r':
                contents = f.read()

                # stop_words = set(stopwords.words('english'))
                #
                # word_tokens = word_tokenize(contents)

                content = contents.split('.')
                total_polarity = 0
                for sentence in content:
                    blob1 = TextBlob(sentence)
                    total_polarity += blob1.sentiment.polarity

                dict[user.decode("utf-8")][folder.decode("utf-8")] = total_polarity



            f.close()

            os.chdir("..")
    os.chdir("..")

print(dict)


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words
def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text
def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))

def remstopwords(sentences):
    filtered_words = []
    for word in sentences:
        filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    return filtered_words

# para = text.split('.')
# total_polarity = 0
# for item in para:
#     print(item)
#
#     blob1 = TextBlob(item)
#     total_polarity += blob1.sentiment.polarity
#     print(total_polarity)
#     print("\n")
# print(total_polarity)
