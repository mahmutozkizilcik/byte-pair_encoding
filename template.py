import sys
from collections import Counter

def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):
    file = open(file_name, "r", encoding="utf-8")
    text = file.read()
    file.close()
    text = text.split()

    for i in range(len(text)):     # add _ to the beginning and end of each word
        text[i] = "_" + text[i] + "_"
        text[i] = list(text[i])     #convert each word to a list of characters

    corpus = text.copy()
    vocabulary = init_vocab[:]
    merges =[]



    for count in range(max_merge_count):

        adjacent_dict ={}
        for c in corpus:  # count adjacent pairs
            for i in range(len(c)- 1):  # iterate through the word
                adjacent_pair = (c[i], c[i+1]) 
                if adjacent_pair in adjacent_dict:    # if the pair is already in the dictionary, increment its count
                    adjacent_dict[(adjacent_pair)] = adjacent_dict[(adjacent_pair)] + 1
                else:   # if the pair is not in the dictionary, add it with a count of 1
                    adjacent_dict[adjacent_pair] = 1
        
        if not adjacent_dict:  # if there are no more adjacent pairs alghorithm stops
            break
        
        
        adjacent_dict = dict(sorted(adjacent_dict.items(), key=lambda x: (-x[1], len(x[0][0]+x[0][1]), x[0][0]+x[0][1]))) # sort the dictionary by frequency, length and alphabetical order
        adjacent_dict_list = list(adjacent_dict.items()) # convert the dictionary to a list

        adjacent_dict_topK = adjacent_dict_list[0:topK] # get the topK most frequent pairs

        found = False # flag to check if the choosen word is found
        choosen_word = ()
        choosen_freq = 0
        
        for adj in adjacent_dict_topK: # check if the choosen word is found for ones with starts _
            if adj[0][0].startswith("_"):
                choosen_word = adj[0]
                choosen_freq = adj[1]
                found = True
                break
        
        if found == False:  # check if the choosen word is found for ones with ends _
            for adj in adjacent_dict_topK:
                if adj[0][1].endswith("_"):
                    choosen_word = adj[0]
                    choosen_freq = adj[1]
                    found = True
                    break

        if found == False:  # if the choosen word is not found for ones with starts _ and ends _
            choosen_word = adjacent_dict_topK[0][0]
            choosen_freq = adjacent_dict_topK[0][1]

            
        new_corpus = [] # create a new corpus
        for c in corpus: # iterate through the corpus
            new_word = [] # create a new word
            i = 0
            while i < len(c): # iterate through the word
                if i < len(c) - 1 and (c[i], c[i+1]) == choosen_word: # if the pair is found
                    new_word.append(c[i] + c[i+1])
                    i += 2
                else:
                    new_word.append(c[i])
                    i += 1
            new_corpus.append(new_word)
        corpus = new_corpus
                    
        merges.append((*choosen_word, choosen_freq)) # add the choosen word to the merges list
        vocabulary.append(choosen_word[0] + choosen_word[1]) # add the choosen word to the vocabulary list
        

        
    return merges, vocabulary


def test_BPE(file_name, merges, vocabulary):
    file = open(file_name, "r", encoding="utf-8")
    text = file.read()
    file.close()
    text = text.split()

    for i in range(len(text)):     # add _ to the beginning and end of each word
        text[i] = "_" + text[i] + "_"
        text[i] = list(text[i])     #convert each word to a list of characters

    test_corpus = text[:]

    # focus_pair= None
    for p1, p2, freq in merges: # iterate through the merges list
        focus_pair = (p1, p2)
        new_test_corpus = []

        for c in test_corpus: 
            new_word = [] # create a new word
            i = 0
            while i < len(c): 
                if i < len(c) - 1 and (c[i], c[i+1]) == focus_pair: # if the pair is found
                    new_word.append(c[i] + c[i+1])
                    i += 2
                else:
                    new_word.append(c[i])
                    i += 1
            new_test_corpus.append(new_word)
        test_corpus = new_test_corpus
    
    tokenized_corpus = []
    # convert the test corpus to a list of token
    for c in test_corpus: 
        for t in c:
            tokenized_corpus.append(t)

    input_ids = []
    # add the index to the input_ids list
    for t in tokenized_corpus: 
        token_index = vocabulary.index(t) 
        input_ids.append(token_index) 


    return tokenized_corpus, input_ids


def print_truncated(lst, file=sys.stdout):
    if len(lst) > 100:
        print(f"{lst[:50]} ... {lst[-50:]}", file=file)
    else:
        print(lst, file=file)


if __name__ == "__main__":
    init_vocab = list(
        "abcçdefgğhıijklmnoöprsştuüvyzqwx"
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
        "0123456789"
        ".,;:!?\"'()[]{}<>-_=/+*%<>^~@#$&|\\`"
    )
    init_vocab.sort()

    merges, vocabulary = train_BPE("train.txt", init_vocab)
    test_BPE("test.txt", merges, vocabulary)

    # ------------------------------------------------------------
    # You can use the following configurations to test your code:
    # ------------------------------------------------------------

    configs = [
        ("train.txt", "test.txt", 10, 1, "myoutput.txt"),
        ("train.txt", "test.txt", 20, 3, "myoutput.txt"),
        ("train1.txt", "test1.txt", 250, 1, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 5, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 10, "myoutput1.txt"),
        ("train2.txt", "test2.txt", 250, 1, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 5, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 10, "myoutput2.txt"),
    ]

    # Initialize or clear the output files
    for filename in ["myoutput.txt", "myoutput1.txt", "myoutput2.txt"]:
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            pass

    for i, (train_file, test_file, max_merges, top_k, out_file) in enumerate(
        configs, 1
    ):
        print(f"Running Configuration {i}: {train_file} -> {out_file}...")

        merges, vocab = train_BPE(
            train_file, init_vocab, max_merge_count=max_merges, topK=top_k
        )
        tokenized, ids = test_BPE(test_file, merges, vocab)

        # Write to consolidated file in append mode with cross-platform compatibility
        with open(out_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(
                f"----- Configuration {i}: Testing with {train_file} and {test_file}: maxMerge={max_merges}, topK={top_k} -----\n"
            )
            print("Merge List:", file=f)
            print_truncated(merges, file=f)
            print("\nUpdated Vocabulary:", file=f)
            print_truncated(vocab, file=f)
            print("\nTokenized Corpus:", file=f)
            print_truncated(tokenized, file=f)
            print("\nInput IDs:", file=f)
            print_truncated(ids, file=f)
            print(f"\nToken Count: {len(tokenized)}", file=f)
            f.write("\n" + "=" * 50 + "\n\n")
