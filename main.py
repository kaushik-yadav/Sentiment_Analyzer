import os
import re
import time

import threading
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openpyxl import load_workbook

syllable_dict = nltk.corpus.cmudict.dict()


def read_from_input(path):
    df = pd.read_excel(path)
    # Reading the data into a dictionary as key:value pairs of id:link
    urls = {}
    ## Extracting first 2 files as a sample
    for i in range(100):
        urls[df["URL_ID"][i]] = df["URL"][i]
    # making a directory if not already made
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    ## deleting because they give 404 error
    if urls.get("blackassign0036"):
        del urls["blackassign0036"]
    if urls.get("blackassign0049"):
        del urls["blackassign0049"]
    return urls


def get_article(file_name, url):
    # scraping article and title using bs4
    req=requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")
    if soup.find("h1", {"class": "entry-title"}):
        title = soup.find("h1", {"class": "entry-title"}).text
    else:
        title = soup.find("h1", {"class": "tdb-title-text"}).text
    # Iterate through each <div> element
    article=""
    if(soup.find(class_="td-post-content tagdiv-type")):
        article=soup.find(class_="td-post-content tagdiv-type").text
    else:
        article=soup.find(class_="tdb-block-inner td-fix-index").text
        if(len(article)<100):
            div_elements = soup.find_all('div', class_='tdb-block-inner td-fix-index')
            if(div_elements):
                for div_element in div_elements:
                    # Find all <p> tags within the current <div> element
                    paragraphs = div_element.find_all('p')
                    # Iterate through each <p> tag and print its text
                    for p in paragraphs:
                        article+=p.text+"\n"
    content = title + "\n\n" + article
    content=content.replace("%","")
    
    # Write content to file in 'tmp' folder
    with open(f"tmp/{file_name}", "w", encoding="utf-8") as f:
        f.write(content)

def write_in_file(file_name, url):
    # writing the data from pandas dataframe to files
    sample_file = file_name
    get_article(file_name, url)
    return sample_file


def extract_data(sample_file):
    # Extracting data from files
    with open(f"tmp/{sample_file}", "r", encoding="utf-8") as f:
        lines = f.readlines()
    ## excluding the last line as it contains the reference for this text (not useful for analysis)
    content = "\n".join(lines[:-1])
    content = content.strip().split()
    return content


def generator(path):
    for row in open(path, "r"):
        yield row


def stop_words_extract(stop_words):
    # using generators to find and store rows of StopWords file and saving it in stop_words variable
    directories = os.listdir("data\StopWords")
    for i in directories:
        i = "data/StopWords/" + i
        for j in generator(i):
            x = j[: len(j) - 1]
            stop_words[x.lower()] = 1
    return stop_words


def pos_neg_words_extract(positive_words, negative_words):
    # using generators to store rows of positive and negative words from files and saving it in positive_words and negative words respectively
    directories = os.listdir("data\MasterDictionary")
    count = 0
    for i in directories:
        i = "data/MasterDictionary/" + i
        count += 1
        for j in generator(i):
            x = j[: len(j) - 1]
            if count == 2:
                positive_words[x] = 1
            else:
                negative_words[x] = -1
    return positive_words, negative_words


def clean_text(content, stop_words,tokenized_stop_words):
    # cleaning the text by removing stop words and checking words by converting into lowercase and removing punctuations
    cleaned_string = ""
    for i in range(len(content)):
        if content[i].isupper():
            continue
        working = True
        for j in stop_words:
            if content[i] == j:
                working = False
        if working:
            cleaned_string += " " + content[i]
    cleaned_string = cleaned_string.strip()
    word_tokens = word_tokenize(cleaned_string)
    filtered_sentence = []
    for w in word_tokens:
        if w not in tokenized_stop_words and w.isalpha():
            filtered_sentence.append(w.lower())
    return cleaned_string, filtered_sentence


def len_of_sentences(cleaned_string):
    # calculating length of sentences (this sentence has punctuations)
    sentence = nltk.sent_tokenize(cleaned_string)
    sentences = len(sentence)
    return sentences


def count_syllables(word):
    if word.lower() not in syllable_dict:
        return 0
    pronunciation = syllable_dict[word.lower()][0]
    return sum(1 for phoneme in pronunciation if phoneme[-1].isdigit())


def count_complex_words(word_list):
    return sum(1 for word in word_list if count_syllables(word) > 2)


def count_syllables(word):
    word = word.lower()
    num_vowels = len([char for char in word if char in "aeiou"])
    if word.endswith("es") or word.endswith("ed"):
        num_vowels -= 1  # do not include 'es' and 'ed' of found
    return num_vowels


def count_syllables_per_word(text):
    words = nltk.word_tokenize(text)
    syllable_counts = [count_syllables(word) for word in words]
    return syllable_counts


def count_pronouns(text):
    pattern = r"\b(I|we|my|ours|us|We|My|Our|Ours|our)\b"
    # making sure not to include US in it
    pattern = r"(?<!\bUS\b)" + pattern
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches)

def calculate_output(
    filtered_sentence,
    cleaned_string,
    negative_words,
    positive_words,
    output,
    file_name,
    url,
):
    # all of the output variables are calculated here
    positive_score, negative_score, polarity_score, subjectivity_score = 0, 0, 0, 0
    for i in filtered_sentence:
        if i in positive_words:
            positive_score += 1
        if i in negative_words:
            negative_score += 1

    if positive_score + negative_score != 0:
        polarity_score = (positive_score - negative_score) / (
            positive_score + negative_score
        )
    polarity_score += 0.000001
    subjectivity_score = (positive_score + negative_score) / len(filtered_sentence)
    subjectivity_score += 0.000001
    word_count = len(filtered_sentence)
    sentences = len_of_sentences(cleaned_string)
    average_sentence_length = word_count / sentences
    new_s = " ".join(filtered_sentence)
    token = word_tokenize(new_s)
    complex_word_count = count_complex_words(token)
    percent_complex = complex_word_count / word_count
    fog_index = 0.4 * (average_sentence_length + (complex_word_count / word_count))
    avg_words_per_sent = average_sentence_length
    syllable_per_word = count_syllables_per_word(cleaned_string)
    avg_syllable_per_word = sum(syllable_per_word) / len(syllable_per_word)
    personal_pronouns = count_pronouns(cleaned_string)
    avg_word_length = 0
    x = "".join(filtered_sentence)
    avg_word_length = len(x) / word_count
    output["FILE_NAME"] = [file_name]
    output["URL"] = [url]
    output["POSITIVE_SCORE"] = [positive_score]
    output["NEGATIVE_SCORE"] = [negative_score]
    output["POLARITY_SCORE"] = [polarity_score]
    output["SUBJECTIVITY_SCORE"] = [subjectivity_score]
    output["AVG_SENTENCE_LENGTH"] = [average_sentence_length]
    output["PERCENTAGE_OF_COMPLEX_WORDS (%)"] = [percent_complex*100]
    output["FOG_INDEX"] = [fog_index]
    output["AVG_NUMBER_OF_WORDS_PER_SENTENCE"] = [avg_words_per_sent]
    output["COMPLEX_WORD_COUNT"] = [complex_word_count]
    output["WORD_COUNT"] = [word_count]
    output["SYLLABLE_PER_WORD"] = [avg_syllable_per_word]
    output["PERSONAL_PRONOUNS"] = [personal_pronouns]
    output["AVG_WORD_LENGTH"] = [avg_word_length]
    return output

def process_urls(file_name,url,list_of_outputs,tokenized_stop_words):
        sample_file = write_in_file(file_name, url)
        content = extract_data(sample_file)
        positive_words, negative_words, stop_words = {}, {}, {}
        stop_words = stop_words_extract(stop_words)
        positive_words, negative_words = pos_neg_words_extract(
            positive_words, negative_words
        )
        cleaned_string, filtered_sentence = clean_text(content, stop_words,tokenized_stop_words)
        output = {}
        calculated_output = calculate_output(
            filtered_sentence,
            cleaned_string,
            negative_words,
            positive_words,
            output,
            file_name,
            url,
        )
        print(f"-- {file_name} is done ")
        list_of_outputs.append(calculated_output)

def calculate_all_variables(urls,tokenized_stop_words):
    list_of_outputs = []
    threads=[]
    for file_name, url in urls.items():
        thread =threading.Thread(target=process_urls,args=(file_name,url,list_of_outputs,tokenized_stop_words))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return list_of_outputs

def saving_file(list_of_outputs, output_file):
    result_df = pd.DataFrame()
    # Loop through each dictionary
    for dicto in list_of_outputs:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(dicto, index=[0])
        # Append the DataFrame to the result DataFrame
        result_df = pd.concat([result_df, df], ignore_index=True)
    # Write the result DataFrame to an Excel file
    sorted_result_df=result_df.sort_values(by="FILE_NAME")
    sorted_result_df.to_excel(output_file, index=False)


def changing_column_widths(output_file):
    # loading the workbook
    workbook = load_workbook(output_file)
    sheet = workbook.active

    column_widths = {
        "A": 15,
        "B": 116,
        "C": 16,
        "D": 16,
        "E": 16,
        "F": 20,
        "G": 24,
        "H": 36,
        "I": 14,
        "J": 40,
        "K": 24,
        "L": 14,
        "M": 22,
        "N": 22,
        "O": 22,
    }

    # Set column widths
    for column, width in column_widths.items():
        sheet.column_dimensions[column].width = width

    # Save the modified workbook
    workbook.save(output_file)


path = "data\Input.xlsx"
urls = read_from_input(path)
list_of_outputs = []
output_file = "output.xlsx"
tokenized_stop_words=set(stopwords.words("english"))
list_of_outputs=calculate_all_variables(urls,tokenized_stop_words)
saving_file(list_of_outputs, output_file)
changing_column_widths(output_file)
