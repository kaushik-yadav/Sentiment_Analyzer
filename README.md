# Sentiment_Analyzer

--- This sentiment analyzer is designed to evaluate the sentiment of textual data extracted from URLs provided in an Excel input file. 
--- The process involves several steps including web scraping, text preprocessing, sentiment analysis, and output generation.
---The script is multithreaded for efficient processing of multiple URLs concurrently.
---This script will generate an Excel file named output.xlsx containing sentiment analysis results for each URL based on the desired Input.xlsx Excel file.

### Components

- **Web Scraping**: Utilizes BeautifulSoup to extract article content from provided URLs.
- **Text Preprocessing**: Cleans the text data by removing stop words, punctuation, and non-alphabetic characters.
- **Sentiment Analysis**: Calculates various sentiment-related metrics including positive and negative scores, polarity score, subjectivity score, average sentence length, percentage of complex words, Fog index, and more.
- **Output Generation**: Organizes the sentiment analysis results into a structured Excel file for further analysis and visualization.
- **Threading**: is utilized in various components of the sentiment analyzer to improve efficiency by allowing multiple tasks to be performed concurrently.
