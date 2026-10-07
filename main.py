import os
import math

def read_reviews( pos_folder , neg_folder , test_folder , pos_files , neg_files , test_files ):

    for i in pos_folder:
        file = open( "./data/imdb1/pos/" + i )
        pos_files.append( file.read() )
        file.close()

    for i in neg_folder:
        file = open( "./data/imdb1/neg/" + i )
        neg_files.append( file.read() )
        file.close()

    for i in test_folder:
        file = open( "./data/imdb1/test/" + i )
        test_files.append( file.read() )   
        file.close() 


def preprocess_text( pos_files , neg_files , test_files ):

    for i in range(len(pos_files)):
        pos_files[i] = pos_files[i].lower()

    for i in range(len(neg_files)):
        neg_files[i] = neg_files[i].lower()

    for i in range(len(test_files)):
        test_files[i] = test_files[i].lower()

    punctuation = [ '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                    ':', ';', '<','=', '>', '?', '@', '[', '\\', ']', '^', '_','`', '{', '|', '}', '~' ]

    for i in range(len(pos_files)):
        for j in punctuation:
            pos_files[i] = pos_files[i].replace(j,"")

    for i in range(len(neg_files)):
        for j in punctuation:
            neg_files[i] = neg_files[i].replace(j,"")  

    for i in range(len(test_files)):
        for j in punctuation:
            test_files[i] = test_files[i].replace(j,"")


    for i in range(len(pos_files)):
        pos_files[i] = pos_files[i].split()

    for i in range(len(neg_files)):
        neg_files[i] = neg_files[i].split()

    for i in range(len(test_files)):
        test_files[i] = test_files[i].split()


def load_stopwords( pos_files , neg_files , test_files ):

    file = open("./data/english.stop")
    stop_words = file.read()
    stop_words = stop_words.split()    

    for i in range(len(pos_files)):
        j = 0
        while j < len(pos_files[i]):
            if pos_files[i][j] in stop_words:
                pos_files[i].pop(j)
            else:
                j += 1

    for i in range(len(neg_files)):
        j = 0
        while j < len(neg_files[i]):
            if neg_files[i][j] in stop_words:
                neg_files[i].pop(j)
            else:
                j += 1 

    for i in range(len(test_files)):
        j = 0
        while j < len(test_files[i]):
            if test_files[i][j] in stop_words:
                test_files[i].pop(j)
            else:
                j += 1  

    file.close()            


def build_vocabulary( pos_vocab , neg_vocab , pos_words , neg_words , pos_files , neg_files ):

    for i in range(len(pos_files)):
        for j in pos_files[i]:
            pos_words.append(j)
            if pos_vocab.get( j , "none" ) == "none":
                pos_vocab.update({j:1})
            else:
                pos_vocab[j] += 1  

    for i in range(len(neg_files)):
        for j in neg_files[i]:
            neg_words.append(j)
            if neg_vocab.get( j , "none" ) == "none":
                neg_vocab.update({j:1})
            else:
                neg_vocab[j] += 1  

    vocab_size = set(pos_words + neg_words)  

    return len(vocab_size)                     


def train_naive_bayes( test_files , pos_vocab , neg_vocab , neg_score_list , pos_score_list , pos_len , neg_len , V , pos_prior , neg_prior ):

    for i in range(len(test_files)):
        pos_score = math.log(pos_prior)
        for j in test_files[i]:
            if pos_vocab.get( j , "none" ) == "none":
                pos_score += math.log( 1 / ( pos_len + V ))
            else:    
                pos_score += math.log( ( pos_vocab[j] + 1 ) / ( pos_len + V ))
        pos_score_list.append(pos_score)  

    for i in range(len(test_files)):
        neg_score = math.log(neg_prior)
        for j in test_files[i]:
            if neg_vocab.get( j , "none" ) == "none":
                neg_score += math.log( 1 / ( neg_len + V ))
            else:    
                neg_score += math.log( ( neg_vocab[j] + 1 ) / ( neg_len + V ))
        neg_score_list.append(neg_score)            


def predict_review( test_folder , pos_score_list , neg_score_list , prediction ):

    for i in range(len(test_folder)):
        if pos_score_list[i] > neg_score_list[i]:
            print(test_folder[i]," -> POSITIVE")
            prediction.append("POSITIVE")
        else:
            print(test_folder[i]," -> NEGATIVE")
            prediction.append("NEGATIVE")        


def predict_test_reviews( test_folder , prediction ):

    file = open("./predictions.txt", "w")
    for i in range(len(test_folder)):
        file.write(test_folder[i] + "," + prediction[i] + "\n") 
    file.close()        


# Main Block

pos_folder = os.listdir("./data/imdb1/pos")
neg_folder = os.listdir("./data/imdb1/neg")
test_folder = os.listdir("./data/imdb1/test")

pos_files = []
neg_files = []
test_files = [] 

read_reviews(pos_folder , neg_folder , test_folder , pos_files , neg_files , test_files )

print("Positive Reviews :",len(pos_files))
print("Negative Reviews :",len(neg_files))
print("Test Reviews :",len(test_files))

preprocess_text( pos_files , neg_files , test_files )

load_stopwords( pos_files , neg_files , test_files )

pos_prior = len(pos_files) / ( len(pos_files) + len(neg_files) )
neg_prior = len(neg_files) / ( len(pos_files) + len(neg_files) )

pos_vocab = {}
neg_vocab = {}
pos_words = []
neg_words = []

V = build_vocabulary( pos_vocab , neg_vocab , pos_words , neg_words , pos_files , neg_files )

print("\nTotal Positive Class Words:",len(pos_words))
print("Total Negative Class Words:",len(neg_words))
print("Total Unique Words in Vocabulary :",V)

neg_score_list = []
pos_score_list = []
prediction = []

train_naive_bayes( test_files , pos_vocab , neg_vocab , neg_score_list , pos_score_list , len(pos_words) , len(neg_words) , V , pos_prior , neg_prior )

predict_review( test_folder , pos_score_list , neg_score_list , prediction)

predict_test_reviews( test_folder , prediction )
