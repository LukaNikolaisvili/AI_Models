import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import string 
from nltk.corpus import gutenberg
import time


start_time = time.time()
#using nltk package to download important things such as stopwords, gutenberg texts and etc...
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('gutenberg')

#as needed something unique to destinguish books I will be using their names (ID's)
book_ids = gutenberg.fileids()

for book_id in book_ids:
    
    # Load the book text
    book_text = gutenberg.raw(book_id)

    # Print all books information before the top 10 words (NOTE# 
    #NLTK might not have the updated information about the books some of them might be missing)
    
    #I am using the rstrip to make the name more beautiful and not to have the format ending at the end
    
    print("Name of the Book: ", book_id.rstrip('.txt'))
    print("Textbook Length (Words): ", len(book_text))
   
    #populating the variable matrix with booktext data
    gutenberg_texts = [book_text]
    #declaring the empty variable to use it after to hold as already processed texts
    processed_texts = []

    #Define the number of top words to keep(so it will be my variable that I can re-use if needed)
    top_n_words = 10

    #Preprocessing the data, couting words and their frequencies for each document downloaded from the gutenberg
    for text in gutenberg_texts:
        #Tokenize and preprocess text
        
        tokens = word_tokenize(text.lower())
        #Remove punctuation and select alphanumeric words
        tokens = [word for word in tokens if word not in stopwords.words('english') and word.isalnum()]
    
        #Count word frequencies
        word_counts = Counter(tokens)
    
        #Get the top words
        top_words = [word for word, _ in word_counts.most_common(top_n_words)]
    
        #Store processed text
        processed_texts.append(top_words)

        #Create a DataFrame for the term-document matrix
        term_document_matrix = pd.DataFrame(processed_texts)

        #Fill NaN values with zeros (for the words which are not present in a document)
        term_document_matrix = term_document_matrix.fillna(0)

        #Print the term-document matrix
        print(term_document_matrix)
        
          # Separation line between books to look nice formated output and it to be easy to determine the other books
        print("=" * 50)

# End of the execution time
end_time = time.time()

# total time needed to execute
elapsed_time = end_time - start_time

#printing total time needed for execution
print("\nTotal time elapsed:", elapsed_time, "seconds")

